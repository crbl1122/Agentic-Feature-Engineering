"""
Research nodes — ReAct loop to discover predictive features via web search + LLM.
"""
import json

from langchain_core.messages import HumanMessage, SystemMessage

from feature_engineer.llm.setup import llm
from feature_engineer.llm.tools import serper_tool
from feature_engineer.state import AgentState
from feature_engineer.storage.parquet import path_to_df

from langgraph.prebuilt import ToolNode

# LLM with Serper tool bound
research_llm = llm.bind_tools([serper_tool])

# ToolNode reads/writes from research_messages key
research_tool_node = ToolNode(tools=[serper_tool], messages_key="research_messages")


def research_features(state: AgentState) -> dict:
    """ReAct node — calls research_llm which may invoke web_search tool."""
    objective = state.get("objective", "")
    existing  = state.get("research_messages", [])

    if not existing:
        if not objective:
            print("[research_features] No objective — skipping.")
            return {"feature_candidates": [], "research_messages": []}

        system = SystemMessage(content=(
            "You are a feature engineering expert. "
            "Use the web_search tool to find predictive features for the given ML use case. "
            "Make 1-3 targeted searches, then summarise the best feature types. "
            "Focus on feature CONCEPTS (e.g. 'recency', 'frequency', 'regional trend'), "
            "not pandas implementations."
        ))
        human = HumanMessage(content=(
            f"Research the most predictive feature types for this objective:\n{objective}\n\n"
            "Search for best practices, then list 8-12 high-value feature types."
        ))
        messages = [system, human]
    else:
        messages = existing

    response = research_llm.invoke(messages)
    return {"research_messages": [response]}


def research_tools_condition(state: AgentState) -> str:
    """Route: tool_calls present → execute tools, else → extract_candidates."""
    messages = state.get("research_messages", [])
    if not messages:
        return "done"
    last = messages[-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return "done"


def extract_candidates(state: AgentState) -> dict:
    """Extract feature candidates from the final research LLM message."""
    messages = state.get("research_messages", [])
    if not messages:
        return {"feature_candidates": []}

    last_text = ""
    for msg in reversed(messages):
        if hasattr(msg, "content") and isinstance(msg.content, str) and msg.content.strip():
            last_text = msg.content
            break

    if not last_text:
        return {"feature_candidates": []}

    extract_prompt = (
        f"From this feature engineering research summary, extract a clean list of "
        f"feature type names (8-12 items). Return ONLY a JSON array of strings.\n\n{last_text}"
    )
    response = llm.invoke(extract_prompt).content.strip()
    try:
        clean      = response.replace("```json", "").replace("```", "").strip()
        candidates = json.loads(clean)
        if not isinstance(candidates, list):
            candidates = []
    except Exception:
        candidates = [
            line.strip().lstrip("-•123456789. ")
            for line in last_text.splitlines()
            if line.strip()
        ][:12]

    print(f"[research_features] {len(candidates)} candidates extracted:")
    for c in candidates:
        print(f"  • {c}")

    return {"feature_candidates": candidates}


def map_to_columns(state: AgentState) -> dict:
    """Map feature candidates to what's feasible given the CSV schema."""
    candidates = state.get("feature_candidates", [])
    if not candidates:
        print("[map_to_columns] No candidates — skipping.")
        return {"feasible_features": []}

    df           = path_to_df(state["df"])
    schema_lines = [
        f"  {col!r}: dtype={df[col].dtype}, sample={df[col].dropna().head(3).tolist()}"
        for col in df.columns
    ]
    candidates_str = "\n".join(f"  - {c}" for c in candidates)

    prompt = f"""You are a feature engineering expert.

Available CSV columns:
{chr(10).join(schema_lines)}

Feature candidates to evaluate:
{candidates_str}

Return ONLY the feasible features as a JSON array of strings.
Include which columns are needed for each.
Example: ["recency: days since last order (needs: date, customer_id)"]
Exclude any feature requiring columns not present.
Return only the JSON array, no other text.
"""
    response = llm.invoke(prompt).content.strip()
    try:
        clean    = response.replace("```json", "").replace("```", "").strip()
        feasible = json.loads(clean)
        if not isinstance(feasible, list):
            feasible = []
    except Exception:
        feasible = candidates

    print(f"[map_to_columns] {len(feasible)}/{len(candidates)} features feasible:")
    for f in feasible:
        print(f"  ✓ {f}")

    return {"feasible_features": feasible}
