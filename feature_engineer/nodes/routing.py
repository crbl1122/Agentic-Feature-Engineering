"""
Routing nodes and pure router functions.
next_feature and save_csv live here alongside the conditional edge functions.
"""
import json
import os

import pandas as pd

from feature_engineer.config import OUTPUT_DIR
from feature_engineer.llm.setup import llm
from feature_engineer.state import AgentState, FeaturePlan
from feature_engineer.storage.parquet import path_to_df


# ── generate_recommendations ─────────────────────────────────────────────────────

def generate_recommendations(state: AgentState) -> dict:
    """Generate top-5 recommended features from research context that need additional data."""
    messages        = state.get("research_messages", [])
    objective       = state.get("objective", "")
    completed       = state.get("completed_features", [])
    original_cols   = state.get("original_columns", [])

    # build context from Serper results
    context_lines = []
    for msg in messages:
        if hasattr(msg, "type") and msg.type == "tool":
            raw = msg.content if isinstance(msg.content, str) else str(msg.content)
            if raw.strip():
                context_lines.append(raw[:500])
    context = "\n\n".join(context_lines[:6])

    if not context:
        return {"feature_recommendations": []}

    prompt = f"""You are a feature engineering expert in pharmacogenomics.

Objective: {objective[:300]}

Available columns: {original_cols}

Features already generated: {completed}

Research context (from literature):
{context}

Based on this literature, identify the TOP 5 most important features that:
1. Would be highly predictive for the objective
2. CANNOT be computed from the available columns above
3. Would require additional data (specify what data)

Return a JSON array with exactly 5 objects, each with:
- "name": short feature name
- "description": biological rationale (1-2 sentences)
- "required_data": what additional data would be needed
- "example_formula": example pandas formula if data were available

Return ONLY the JSON array, no other text."""

    try:
        response = llm.invoke(prompt)
        raw      = response.content.strip()
        # strip markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        recommendations = json.loads(raw.strip())
        print(f"[recommendations] Generated {len(recommendations)} feature recommendations")
        return {"feature_recommendations": recommendations[:5]}
    except Exception as e:
        print(f"[recommendations] Error: {e}")
        return {"feature_recommendations": []}


# ── save_csv ────────────────────────────────────────────────────────────────────

def save_csv(state: AgentState) -> dict:
    """Write enriched DataFrame (original + validated features only) to output/."""
    df        = path_to_df(state["df"])
    completed = state.get("completed_features", [])
    original  = state.get("original_columns", [])

    keep = original + [c for c in completed if c not in original]
    df   = df[[c for c in keep if c in df.columns]]

    print("\n── Current state ───────────────────────────")
    print(f"  input_path         : {state['input_path']}")
    print(f"  output_path        : {state['output_path']}")
    print(f"  objective          : {state['objective']!r}")
    print(f"  max_features       : {state.get('max_features', 3)}")
    print(f"  attempts           : {state.get('attempts', 0)}")
    print(f"  errors             : {state['errors']}")
    print(f"  completed_features : {completed}")
    print(f"  df.shape           : {df.shape}")
    print(f"  df.columns         : {df.columns.tolist()}")
    print("────────────────────────────────────────────\n")

    if not completed:
        print("[save_csv] No features passed validation — saving original DataFrame.")
    else:
        for col in completed:
            print(f"[save_csv] Feature added: '{col}'")

    df.to_csv(state["output_path"], index=False)
    abs_path = os.path.abspath(state["output_path"])
    print(f"[save_csv] Written to '{abs_path}'")
    print(f"[save_csv] {len(completed)} feature(s) added: {completed or 'none'}")
    return {}


# ── next_feature ────────────────────────────────────────────────────────────────

def next_feature(state: AgentState) -> dict:
    """Pop next feature from queue, skip duplicates, request replacement if duplicate."""
    completed_formulas = state.get("completed_formulas", [])
    completed_features = state.get("completed_features", [])
    failed_formulas    = state.get("failed_formulas", [])
    global_bl_formulas = set(completed_formulas) | set(failed_formulas)

    queue = list(state["feature_queue"])

    while queue:
        item    = queue[0]
        formula = item["pandas_code"]
        name    = item["feature_name"]

        if formula in global_bl_formulas:
            print(f"[next_feature] Skipping — formula in blacklist: {name}")
            queue = queue[1:]
        elif name in completed_features:
            print(f"[next_feature] Duplicate feature_name '{name}' — requesting replacement.")
            return {
                "plan":          FeaturePlan(**item),
                "feature_queue": queue[1:],
                "attempts":      0,
                "errors":        [f"__duplicate__: '{name}' already completed"],
            }
        else:
            break

    if not queue:
        print("[next_feature] Queue empty → save_csv")
        return {
            "feature_queue": [],
            "errors":        ["__queue_exhausted__"],
        }

    plan = FeaturePlan(**queue[0])
    print(f"[next_feature] Processing: {plan.feature_name}")
    return {
        "plan":          plan,
        "feature_queue": queue[1:],
        "attempts":      0,
        "errors":        [],
    }


# ── Pure routers ────────────────────────────────────────────────────────────────

def should_execute(state: AgentState) -> str:
    """Route after validate_code.
    'run'    → AST passed.
    'revise' → SyntaxError — fixable, not dangerous.
    'next'   → Security block — skip, no revision.
    'save'   → Security block + queue empty.
    """
    if not state["errors"]:
        return "run"

    error = state["errors"][0]
    if "Syntax error" in error:
        print(f"[router] Syntax error in '{state['plan'].feature_name}' — routing to revise_plan.")
        return "revise"

    print(f"[router] AST blocked '{state['plan'].feature_name}' — skipping (unsafe code).")
    return "next" if state["feature_queue"] else "save"


def should_retry(state: AgentState) -> str:
    """Route after validate: revise / record / next / save."""
    if state["errors"] and state.get("attempts", 0) < 3:
        print(f"[router] Errors detected → revise_plan (attempt {state.get('attempts', 0)}/3)")
        return "revise"
    if state["errors"]:
        print(f"[router] Exhausted retries for '{state['plan'].feature_name}' — skipping.")
        return "next" if state["feature_queue"] else "save"
    return "record"


def after_record(state: AgentState) -> str:
    """Route after record_feature: next or save."""
    if state["feature_queue"]:
        print(f"[router] {len(state['feature_queue'])} feature(s) remaining → next_feature")
        return "next"
    print("[router] All features processed → save_csv")
    return "save"


def after_next(state: AgentState) -> str:
    """Route after next_feature: run / revise / save."""
    errors = state.get("errors", [])
    if errors and errors[0] == "__queue_exhausted__":
        return "save"
    if errors and errors[0].startswith("__duplicate__"):
        return "revise"
    return "run"
