"""
Research nodes — web search ReAct loop + structured output everywhere.

No manual JSON parsing — all LLM outputs use Pydantic structured output.
"""
import json
import re
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from feature_engineer.llm.setup import (
    llm, research_base_llm, research_structured_llm,
    evaluation_llm, mapping_llm,
)
from feature_engineer.llm.tools import serper_tool
from feature_engineer.state import (
    AgentState, ResearchFeature, ResearchFeatureList,
    EvaluationResult, FeasibleFeatureList,
)
from feature_engineer.storage.parquet import path_to_df

from langgraph.prebuilt import ToolNode

MAX_RESEARCH_ATTEMPTS = 1

_research_llm_forced = research_base_llm.bind_tools([serper_tool], tool_choice="required")
_research_llm_auto   = research_base_llm.bind_tools([serper_tool])
research_tool_node   = ToolNode(tools=[serper_tool], messages_key="research_messages")

# path to custom arXiv MCP server
_ARXIV_SERVER_PATH = Path(__file__).parent.parent / "mcp_servers" / "arxiv_server.py"


def _expand_queries(objective: str, domain_hint: str, n: int = 3) -> list[str]:
    """Generate n diverse search queries for the same domain using LLM."""
    prompt = (
        f"Generate {n} different search queries to find relevant notebooks, papers or tutorials.\n\n"
        f"Domain: {domain_hint}\n"
        f"Objective: {objective[:300]}\n\n"
        f"Rules:\n"
        f"- Keep queries SHORT: 4-6 words\n"
        f"- Use the most specific domain keywords (dataset name, disease, technique)\n"
        f"- Each query must use a different angle:\n"
        f"  1. dataset name + feature engineering\n"
        f"  2. dataset name + EDA or data analysis\n"
        f"  3. dataset name + prediction or machine learning\n"
        f"- Return only the {n} queries, one per line, no numbering or explanation"
    )
    response = llm.invoke(prompt)
    queries  = [q.strip() for q in response.content.strip().splitlines() if q.strip()]
    return queries[:n]


def _generate_arxiv_query(objective: str, domain_hint: str) -> str:
    """Generate a single arXiv-specific query from the objective."""
    prompt = (
        f"Generate ONE short arXiv search query to find academic papers "
        f"about feature engineering for this specific ML task.\n"
        f"Domain: {domain_hint}\n"
        f"Objective: {objective[:200]}\n\n"
        f"Rules:\n"
        f"- Start with 'abs:' followed by 4-6 words\n"
        f"- Use the MOST SPECIFIC domain keyword (dataset name or disease name)\n"
        f"- Do NOT include generic terms like 'machine learning', 'tabular', 'preprocessing'\n"
        f"- Return ONLY the query string, nothing else"
    )
    response = llm.invoke(prompt)
    query = response.content.strip()
    # ensure it starts with abs:
    if not query.startswith("abs:"):
        query = f"abs:{query}"
        query = query.replace("abs: ", "abs:")
    return query


def _extract_arxiv_features(papers: list[dict], objective: str, column_schema: dict) -> list[ResearchFeature]:
    """Extract features from arXiv papers using LLM directly with schema-aware comparison."""
    features = []

    # build schema description for LLM
    schema_lines = []
    for col, info in column_schema.items():
        sem  = info.get("semantic_type", "unknown")
        samp = info.get("sample_values", [])[:3]
        schema_lines.append(f"  '{col}': {sem}, sample={samp}")
    schema_str = "\n".join(schema_lines)

    for paper in papers:
        text = paper.get("full_text") or paper.get("abstract", "")
        if not text:
            continue

        # pre-filter — skip papers clearly unrelated to feature engineering or ML
        abstract_lower = paper.get("abstract", "").lower()
        relevant_keywords = ["feature", "preprocess", "transform", "encode",
                             "predict", "machine learning", "data", "model"]
        if not any(kw in abstract_lower for kw in relevant_keywords):
            print(f"[arxiv] Skipping irrelevant paper: {paper['title'][:60]}")
            continue

        print(f"[arxiv] Extracting features from: {paper['title'][:60]}")
        try:
            prompt = (
                f"You are a feature engineering expert.\n"
                f"Objective: {objective[:200]}\n\n"
                f"Available dataset columns with their semantic types and sample values:\n"
                f"{schema_str}\n\n"
                f"Use this paper text to inspire for data transformation to use in features.\n"
                f"If nothing relevant is found, return [].\n"
                f"For each feature:\n"
                f"  - If it CAN be computed from the available columns (semantic match, "
                f"even if column names differ), return it with formula using actual column names\n"
                f"  - If it CANNOT be computed (requires data not in the dataset), still return it "
                f"with requires_external_data=true and describe what data is needed\n\n"
                f"Return a JSON array with objects containing:\n"
                f"  name, description, formula (pandas), requires_external_data (bool), required_data (str if external)\n"
                f"Return ONLY the JSON array. If nothing relevant found, return [].\n\n"
                f"AVOID: simple dtype casts of single existing columns (e.g. Y/N → 0/1 without combining with other columns)\n"
                f"AVOID: features that use target columns directly\n\n"
                f"Paper text:\n{text[:10000]}"
            )
            response = llm.invoke(prompt)
            raw = response.content.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            extracted = json.loads(raw.strip())
            for f in extracted:
                # normalize name to snake_case
                raw_name = f.get("name", "")
                snake_name = re.sub(r'(?<=[a-z0-9])(?=[A-Z])', '_', raw_name).lower()
                snake_name = re.sub(r'[^a-z0-9_]', '_', snake_name)
                snake_name = re.sub(r'_+', '_', snake_name).strip('_')
                features.append(ResearchFeature(
                    name=snake_name,
                    description=f.get("description", ""),
                    formula_hint=f.get("formula", ""),
                    source="arxiv",
                ))
                ext = " [external data required]" if f.get("requires_external_data") else ""
                print(f"[arxiv] Feature found: {snake_name}{ext} (source: arxiv)")
        except Exception as e:
            print(f"[arxiv] Extraction error for {paper['id']}: {e}")

    return features


async def research_features(state: AgentState) -> dict:
    """ReAct node — Kaggle-focused multi-search with chain-of-thought."""
    objective         = state.get("objective", "")
    existing          = state.get("research_messages", [])
    research_feedback = state.get("research_feedback", "")

    if not existing:
        if not objective:
            print("[research_features] No objective — skipping.")
            return {"feature_candidates": [], "research_messages": []}

        domain_hint = (await llm.ainvoke(
            f"Extract a short 5-10 word domain title from this ML objective. "
            f"Return ONLY the title as plain words separated by spaces, no hyphens or underscores.\n\n{objective[:500]}"
        )).content.strip()
        domain_hint = domain_hint.replace("-", " ").replace("_", " ")
        print(f"[research_features] Domain: {domain_hint}")
        print(f"[research_features] Searching for domain-specific features...")

        df           = path_to_df(state["df"]) if state.get("df") else None
        columns_info = ""
        if df is not None:
            col_details = [
                f"  {col} ({df[col].dtype}): sample={df[col].dropna().head(3).tolist()}"
                for col in df.columns
            ]
            columns_info = "Available columns:\n" + "\n".join(col_details)

        system = SystemMessage(content=(
            "You are a feature engineering expert specializing in competitive ML.\n\n"
            "CHAIN OF THOUGHT — follow these steps:\n"
            "  Step 1: Extract the ML domain from the objective.\n"
            "  Step 2: Review the available column names and types.\n"
            "  Step 3: Search for feature engineering best practices 2-3 times.\n"
            "  Step 4: Synthesise a curated list of specific, computable features "
            "with formula hints based on the ACTUAL columns available.\n\n"
            "SEARCH RULES:\n"
            "  - Make 2-3 searches with different angles\n"
            "  - Search for notebooks, papers, or tutorials relevant to the domain\n"
            "  - Focus on features that use the ACTUAL columns listed below\n\n"
            "OUTPUT: after searching, produce a structured list of features.\n"
            "Each feature must have a name, description, and formula_hint "
            "using the actual column names.\n\n"
            f"{columns_info}"
        ))

        queries     = _expand_queries(objective, domain_hint, n=3)
        queries_str = "\n".join(f"  - {q}" for q in queries)
        print(f"[research_features] Expanded queries:\n{queries_str}")
        human = HumanMessage(content=(
            f"Objective:\n{objective}\n\n"
            "Follow the chain-of-thought steps:\n"
            "  1. Extract the domain.\n"
            "  2. Review the available columns.\n"
            f"  3. Search using these queries (use each one):\n{queries_str}\n"
            "  4. After all searches, produce the structured feature list."
        ))

        messages          = [system, human]
        is_first_or_retry = True

        # ── arXiv MCP — connects to HTTP server started by main.py ──
        if _ARXIV_SERVER_PATH.exists():
            arxiv_query = _generate_arxiv_query(objective, domain_hint)
            print(f"[arxiv] Query: {arxiv_query}")
            try:
                from langchain_mcp_adapters.client import MultiServerMCPClient
                client = MultiServerMCPClient({
                    "arxiv": {
                        "url": "http://127.0.0.1:8000/mcp",
                        "transport": "http",
                    }
                })
                tools = await client.get_tools()
                tools = {t.name: t for t in tools}

                    # search
                search_result = await tools["search_arxiv"].ainvoke({
                        "query": arxiv_query, "max_results": 5
                    })
                print(f"[arxiv] search_result type: {type(search_result)}")
                print(f"[arxiv] search_result[:200]: {str(search_result)[:200]}")

                # unwrap LangChain tool output format
                if isinstance(search_result, list) and search_result and "text" in search_result[0]:
                    papers = [json.loads(item["text"]) for item in search_result]
                else:
                    papers = json.loads(search_result) if isinstance(search_result, str) else search_result

                print(f"[arxiv] Found {len(papers)} papers")

                    # download + read each paper
                enriched = []
                for paper in papers:
                    try:
                        await tools["download_paper"].ainvoke({
                                "paper_id": paper["id"], "pdf_url": paper["pdf_url"]
                            })
                        full_text = await tools["read_paper"].ainvoke({
                                "paper_id": paper["id"]
                            })
                        # unwrap
                        if isinstance(full_text, list) and full_text and "text" in full_text[0]:
                            full_text = full_text[0]["text"]

                    except Exception as e:
                        print(f"[arxiv] Could not read {paper['id']}: {e}")
                        paper["full_text"] = paper.get("abstract", "")
                    enriched.append(paper)

                if enriched:
                    arxiv_features = _extract_arxiv_features(enriched, objective, state.get("column_schema", {}))
                    if arxiv_features:
                        print(f"[arxiv] Added {len(arxiv_features)} features to research context")
                        arxiv_lines = "\n".join(
                            f"  - {f.name}: {f.description} → {f.formula_hint} [SOURCE: arxiv]"
                            for f in arxiv_features
                        )
                        messages.append(AIMessage(content=f"Features from arXiv papers:\n{arxiv_lines}"))
                    else:
                        print("[arxiv] No features extracted from papers")
                else:
                    print("[arxiv] No papers found — continuing with Serper only")
            except Exception as e:
                import traceback
                print(f"[arxiv] MCP client error: {e}")
                traceback.print_exc()
        else:
            print(f"[arxiv] Server not found at {_ARXIV_SERVER_PATH} — skipping")

        messages_update = {"research_messages": messages}

    else:
        messages          = existing
        is_first_or_retry = False
        messages_update   = None

    research_llm = _research_llm_forced if is_first_or_retry else _research_llm_auto

    response = await research_llm.ainvoke(messages)

    if hasattr(response, "tool_calls") and response.tool_calls:
        for tc in response.tool_calls:
            query = tc.get("args", {}).get("query", "")
            if query:
                print(f"[research] LLM searched: {query}")

    if messages_update:
        messages_update["research_messages"] = messages + [response]
        return messages_update
    return {"research_messages": [response]}


def research_tools_condition(state: AgentState) -> str:
    """Route: tool_calls → tools, else → extract_candidates."""
    messages = state.get("research_messages", [])
    if not messages:
        return "done"
    last = messages[-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return "done"


def extract_candidates(state: AgentState) -> dict:
    """Extract feature candidates using structured output — no JSON parsing."""
    messages = state.get("research_messages", [])
    if not messages:
        return {"feature_candidates": [], "research_formula_hints": {}, }

    # extract Serper results as structured candidates
    serper_candidates = []
    for msg in messages:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                query = tc.get("args", {}).get("query", "")
                if query:
                    print(f"[research] Serper query: {query}")
        if hasattr(msg, "type") and msg.type == "tool":
            raw = msg.content if isinstance(msg.content, str) else str(msg.content)
            if raw.strip():
                lines = raw.splitlines()
                print(f"[research] Serper results:")
                i = 0
                count = 0
                while i < len(lines) and count < 6:
                    line = lines[i].strip()
                    if line and not line.startswith("URL:"):
                        url = ""
                        if i + 1 < len(lines) and lines[i+1].strip().startswith("URL:"):
                            url = lines[i+1].strip()[4:].strip()
                        print(f"  • {line[:100]}")
                        if url:
                            print(f"    {url}")
                        serper_candidates.append({
                            "name": line[:80],
                            "description": line[:200],
                            "formula_hint": "",
                            "source": "serper",
                            "url": url,
                        })
                        count += 1
                    i += 1

    # build context from Serper messages
    conversation = []
    for msg in messages:
        if hasattr(msg, "content") and isinstance(msg.content, str) and msg.content.strip():
            conversation.append(msg.content)
    context = "\n\n---\n\n".join(conversation)

    if not context:
        return {"feature_candidates": [], "research_formula_hints": {}}

    df           = path_to_df(state["df"]) if state.get("df") else None
    columns_info = f"Available columns: {df.columns.tolist()}" if df is not None else ""
    objective    = state.get("objective", "")

    target_cols  = state.get("target_columns", [])
    target_note  = ""
    if target_cols:
        target_note = (
            f"\nNEVER extract candidates that use these TARGET COLUMNS: {target_cols}\n"
            "These are prediction targets — features derived from them cause leakage.\n"
        )

    # arXiv features injected as AIMessage in research_messages — already in context
    # extract them explicitly for the prompt
    arxiv_section = ""
    for msg in messages:
        if hasattr(msg, "type") and msg.type == "ai":
            raw = msg.content if isinstance(msg.content, str) else ""
            if "[SOURCE: arxiv]" in raw:
                arxiv_section = f"\n{raw}\n"
                break

    prompt = (
        f"Based on this research:\n\n{context}\n\n"
        f"{arxiv_section}"
        f"Objective: {objective}\n"
        f"{columns_info}\n"
        f"{target_note}\n"
        "The context above contains TWO types of information:\n"
        "  1. Web search results (Serper) — use these as INSPIRATION to propose new features with pandas formulas\n"
        "  2. Features from arXiv papers marked [SOURCE: arxiv] — INCLUDE these DIRECTLY in your list as-is, do not rephrase or replace them\n\n"
        "Extract a curated list of 8-10 specific, computable feature candidates.\n"
        "Include relevant features from BOTH the web research AND the arXiv papers above.\n"
        "If the research context is limited, use your domain knowledge to propose additional "
        "relevant features based on the available columns.\n"
        "Each feature must:\n"
        "  1. Be a concrete transformation with a clear pandas formula\n"
        "  2. Be relevant to the objective domain\n"
        "  3. Use columns that actually exist in the dataset\n"
        "  4. Have a formula_hint you could turn into a pandas expression\n\n"
        "IMPORTANT: propose a MIX of feature types. At least 4 must be NUMERIC:\n"
        "  - Numeric examples: groupby().transform('mean'), .eq('Y').groupby().transform('mean'),\n"
        "    groupby().transform('count'), groupby().transform('nunique'),\n"
        "    binary encoding (map Y→1/N→0) combined with another column\n"
        "  - Categorical examples: col1 + '_' + col2 (only for low-cardinality columns)\n\n"
        "PREFER features that achieve one of these goals:\n"
        "  - Add meaningful interactions (combine 2+ columns to capture joint effects)\n"
        "  - Summarize cohort structure (count, frequency, or ratio per group)\n"
        "  - Improve separability (encode categories that distinguish the target)\n"
        "  - Inject prior statistical information safely (without using target columns)\n\n"
        "AVOID: simple binary encodings of a single existing column "
        "(e.g. Y→1/N→0 is trivial if the column already exists as Y/N).\n"
        "AVOID: methodology names like 'feature selection' or 'normalization'.\n"
        "AVOID: features with no transformation beyond dtype casting."
    )

    result: ResearchFeatureList = research_structured_llm.invoke(prompt)

    candidates     = []
    formula_hints  = {}

    print(f"[extract_candidates] {len(result.features)} candidates extracted (structured):")
    for f in result.features:
        label = f"{f.name}: {f.description}"
        candidates.append(label)
        if f.formula_hint:
            formula_hints[label] = f.formula_hint
        hint_str   = f"  → {f.formula_hint}" if f.formula_hint else ""
        source_str = f"  [source: {f.source}]" if hasattr(f, "source") and f.source else ""
        print(f"  • {label}{hint_str}{source_str}")

    return {
        "feature_candidates":     candidates,
        "research_formula_hints": formula_hints,
    }


def evaluate_research(state: AgentState) -> dict:
    """Evaluate candidates using structured EvaluationResult — no JSON parsing."""
    candidates      = state.get("feature_candidates", [])
    good_candidates = list(state.get("good_candidates", []))
    objective       = state.get("objective", "")
    attempts        = state.get("research_attempts", 0) + 1

    if not candidates:
        return {
            "research_attempts":    attempts,
            "research_is_specific": True,
            "research_feedback":    "",
            "research_messages":    [],
        }

    already_good   = set(good_candidates)
    new_candidates = [c for c in candidates if c not in already_good]

    if not new_candidates:
        return {
            "research_attempts":    attempts,
            "research_is_specific": True,
            "good_candidates":      good_candidates,
            "research_feedback":    "",
            "research_messages":    [],
        }

    df           = path_to_df(state["df"])
    column_names = df.columns.tolist()
    target_cols    = state.get("target_columns", [])
    candidates_str = "\n".join(f"  - {c}" for c in new_candidates)

    target_block = ""
    if target_cols:
        target_block = (
            f"TARGET COLUMNS — these are prediction targets, NEVER use them in features: {target_cols}\n"
            "Reject any candidate that uses a target column directly or indirectly "
            "(binning, groupby, ratio, etc.).\n\n"
        )

    prompt = f"""You are a feature engineering quality reviewer.

Objective:
{objective}

{target_block}Available columns: {column_names}

New feature candidates to evaluate:
{candidates_str}

For EACH candidate classify as specific (ACCEPT) or generic (REJECT):

ACCEPT — specific and computable:
  - Names a concrete transformation: ratio, lag, bin, rolling, interaction, count, combination
  - You could write a pandas expression for it without guessing
  - Relevant to the domain described in the objective
  - NOT identical to an existing column (renaming/retyping)
  - Combines 2+ columns OR applies a non-trivial transformation to 1 column
  NOTE: lag/shift/rolling of existing columns are ALWAYS ACCEPT ✓
  NOTE: ratios between two meaningful columns are ALWAYS ACCEPT ✓
  NOTE: rolling mean/std/sum over time are ALWAYS ACCEPT ✓
  NOTE: growth rate (lag-based) features are ALWAYS ACCEPT ✓

REJECT — generic, vague, off-domain, or trivially simple:
  REJECT_GENERIC: methodology name, process name, category name
  REJECT_VAGUE: names a concept where NO pandas formula is imaginable
    Signs: "historical X", "X factors", "X conditions", "X patterns"
    NOT vague: ratio of A to B, rolling mean of X, growth rate of Y — these are clear
  REJECT_OFFDOMAIN: clearly belongs to a different domain than the objective
  REJECT_IDENTICAL: same concept as an existing column with no new computation
  REJECT_TRIVIAL: simple dtype cast of a SINGLE existing column — adds no new information
    e.g. "cna_flag: map Y→1, N→0" when CNA already exists → trivial dtype cast
    NOT trivial: ratio between two columns, normalized metric, per-capita calculation
    EXCEPTION: binary encoding is ACCEPT if combined with another column or used in interaction

Fill the reasons dict with the rejection reason for each generic candidate.
"""
    result: EvaluationResult = evaluation_llm.invoke(prompt)

    newly_good      = result.specific
    rejected        = result.generic
    reasons         = {r.candidate: r.reason for r in result.rejection_reasons}
    good_candidates = good_candidates + [c for c in newly_good if c not in already_good]
    n_rejected      = len(rejected)

    print(f"[evaluate_research] Attempt {attempts}/{MAX_RESEARCH_ATTEMPTS}")
    print(f"  Objective: {objective[:80]}{'...' if len(objective) > 80 else ''}")
    if newly_good:
        print(f"  accepted ({len(newly_good)}):")
        for c in newly_good:
            print(f"      • {c}")
    if rejected:
        print(f"  rejected ({n_rejected}):")
        for c in rejected:
            print(f"      • {c}  <- {reasons.get(c, '')}")
    print(f"  Total good so far: {len(good_candidates)}")

    if n_rejected == 0 or attempts >= MAX_RESEARCH_ATTEMPTS:
        return {
            "research_attempts":    attempts,
            "research_is_specific": True,
            "good_candidates":      good_candidates,
            "research_feedback":    "",
            "research_messages":    [],
        }

    # MAX_RESEARCH_ATTEMPTS=1 so we never reach here, but kept for safety
    return {
        "research_attempts":    attempts,
        "research_is_specific": True,
        "good_candidates":      good_candidates,
        "research_feedback":    "",
        "research_messages":    [],
    }


def after_evaluate_research(state: AgentState) -> str:
    """Route: retry if generic candidates remain and budget allows, else continue."""
    if state.get("research_is_specific", False):
        return "continue"
    if state.get("research_attempts", 0) >= MAX_RESEARCH_ATTEMPTS:
        print("[evaluate_research] Max attempts reached — continuing.")
        return "continue"
    print("[evaluate_research] Requesting replacements for rejected candidates.")
    return "retry"


def map_to_columns(state: AgentState) -> dict:
    """Map good_candidates to feasible features using structured FeasibleFeatureList."""
    candidates    = state.get("good_candidates") or state.get("feature_candidates", [])
    formula_hints = state.get("research_formula_hints", {})

    if not candidates:
        print("[map_to_columns] No candidates — skipping.")
        return {"feasible_features": []}

    df           = path_to_df(state["df"])
    print(f"[map_to_columns] df.columns: {df.columns.tolist()}")
    print(f"[map_to_columns] column_schema keys: {list(state.get('column_schema', {}).keys())}")

    schema_lines = [
        f"  {col!r}: dtype={df[col].dtype}, sample={df[col].dropna().head(3).tolist()}"
        for col in df.columns
    ]

    # detect natural grouping columns
    grouping_cols = [
        col for col in df.columns
        if df[col].dtype == object and 2 <= df[col].nunique() <= 100
    ]
    grouping_hint = ""
    if grouping_cols:
        grouping_hint = (
            f"\nNATURAL GROUPING COLUMNS: {grouping_cols}\n"
            "For each numeric feature candidate, also consider a per-group variant "
            "using these columns (e.g. mean per country, sum per region).\n"
            "Per-group features are often more informative than global stats.\n"
        )

    hints_section = ""
    if formula_hints:
        hints_section = "\nFormula hints from research:\n"
        for label, hint in formula_hints.items():
            hints_section += f"  {label} → {hint}\n"

    target_cols  = state.get("target_columns", [])
    target_block = ""
    if target_cols:
        target_block = (
            f"\nTARGET COLUMNS — never use these in feature formulas: {target_cols}\n"
            "EXCLUDE any candidate whose formula uses a target column directly or indirectly.\n"
        )

    candidates_str = "\n".join(f"  - {c}" for c in candidates)

    prompt = f"""You are a feature engineering expert.

Available CSV columns:
{chr(10).join(schema_lines)}
{hints_section}{grouping_hint}{target_block}
Feature candidates to evaluate for feasibility:
{candidates_str}

For each candidate, decide if it can be computed from the available columns.

INCLUDE if:
  - There is a direct or semantic column match
  - The column contains data of the RIGHT NATURE for this feature
  - A meaningful pandas expression can be written

EXCLUDE if:
  - No column in the dataset relates to this concept
  - The column type is wrong (e.g. feature needs text but column is numeric)
  - The mapping would be forced or unrelated

For each feasible feature provide:
  - name: short snake_case name
  - description: what it measures
  - needs: list of required column names
  - formula_hint: pandas expression hint using actual column names
"""
    result: FeasibleFeatureList = mapping_llm.invoke(prompt)

    feasible_strs   = []
    feasible_names_lower = set()
    for f in result.features:
        needs_str = ", ".join(f.needs)
        label = f"{f.name}: {f.description} (needs: {needs_str})"
        feasible_strs.append(label)
        feasible_names_lower.add(f.name.lower())

    print(f"[map_to_columns] feasible_names_lower: {feasible_names_lower}")
    for c in candidates:
        matches = [n for n in feasible_names_lower if c.lower().startswith(n)]
        print(f"[map_to_columns] candidate: {c[:50]} → matches: {matches}")

    # collect infeasible candidates for recommendations
    infeasible_strs = [c for c in candidates if not any(
        c.lower().startswith(n) for n in feasible_names_lower
    )]

    print(f"[map_to_columns] {len(result.features)}/{len(candidates)} features feasible:")
    for f in result.features:
        print(f"  ✓ {f.name}: {f.description} (needs: {', '.join(f.needs)})")
    for c in infeasible_strs:
        print(f"  ✗ {c[:80]} → infeasible (missing columns)")

    return {"feasible_features": feasible_strs, "infeasible_features": infeasible_strs}
