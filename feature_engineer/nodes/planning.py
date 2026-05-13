"""
Planning nodes — feature_planner and validate_plan.
"""
from feature_engineer.llm.naming import derive_feature_name
from feature_engineer.llm.setup import planner_llm, verdict_llm
from feature_engineer.state import AgentState, FeaturePlan, VerdictList
from feature_engineer.storage.parquet import path_to_df


def feature_planner(state: AgentState) -> dict:
    """Ask LLM to plan up to max_features new columns guided by feasible_features."""
    df    = path_to_df(state["df"])
    max_f = state.get("max_features", 3)

    schema_lines = [
        f"  {col!r}: dtype={df[col].dtype}, sample={df[col].dropna().head(3).tolist()}"
        for col in df.columns
    ]
    schema_str = "\n".join(schema_lines)

    objective = state.get("objective", "")
    hint_block = (
        f"\nYour PRIMARY objective is: {objective}\n"
        f"All features MUST be directly related to this objective.\n"
        if objective
        else "\nDesign features that would add the most predictive value.\n"
    )

    feasible = state.get("feasible_features", [])
    feasible_block = (
        "\nPrioritise generating features from this researched list of high-value candidates:\n"
        + "\n".join(f"  - {f}" for f in feasible) + "\n"
        if feasible else ""
    )

    completed_formulas = state.get("completed_formulas", [])
    failed_formulas    = state.get("failed_formulas", [])

    blacklist_block = ""
    if completed_formulas:
        blacklist_block += (
            "\nAlready ACCEPTED formulas — do NOT reuse or propose semantically equivalent:\n"
            + "\n".join(f"  - {f}" for f in completed_formulas) + "\n"
        )
    if failed_formulas:
        blacklist_block += (
            "\nAlready FAILED formulas — do NOT reuse:\n"
            + "\n".join(f"  - {f}" for f in failed_formulas) + "\n"
        )

    prompt = f"""You are a feature engineering expert.

Given this CSV schema:
{schema_str}
{hint_block}{feasible_block}{blacklist_block}
Generate between 1 and {max_f} new feature columns.
For each feature return a FeaturePlan with:
- feature_name: short snake_case column name
- description: what it represents
- pandas_code: a single Python *expression* using only `df` that returns a pandas Series.
  Do NOT import anything. Do NOT assign variables.

GROUPBY RULE: if description says "per X", use groupby + transform.
  Example: "total sales per region" → df.groupby('region')['unit_price'].transform('sum')

DATETIME RULE: wrap BOTH sides with pd.to_datetime() when subtracting dates.
  RIGHT: pd.to_datetime(df['date']) - pd.to_datetime(df.groupby('x')['date'].transform('min'))

APPLY vs TRANSFORM RULE (CRITICAL):
  NEVER use groupby().apply() to create a column — it changes the index and produces NaN.
  ALWAYS use groupby().transform() or pre-multiply then transform:
  WRONG: df.groupby('x').apply(lambda x: x['a'].sum())          → NaN column
  RIGHT: df.groupby('x')['a'].transform('sum')                  → correct
  WRONG: df.groupby('x').apply(lambda x: (x['a']*x['b']).sum()) → NaN column
  RIGHT: (df['a'] * df['b']).groupby(df['x']).transform('sum')  → correct
"""
    result   = planner_llm.invoke(prompt)
    features = result.features[:max_f]

    for f in features:
        f.feature_name = derive_feature_name(f.pandas_code)

    print(f"[feature_planner] hint: {objective or '(none)'}")
    print(f"[feature_planner] Planned {len(features)} feature(s):")
    for f in features:
        print(f"  • {f.feature_name}: {f.pandas_code}")

    return {
        "plan":               features[0],
        "feature_queue":      [f.model_dump() for f in features[1:]],
        "completed_features": [],
        "completed_plans":    [],
        "completed_formulas": [],
        "failed_formulas":    [],
        "attempts":           0,
        "errors":             [],
    }


def validate_plan(state: AgentState) -> dict:
    """Filter planned features using chain-of-thought verdicts (hint + leakage)."""
    plan  = state["plan"]
    queue = state["feature_queue"]
    all_features = [plan.model_dump()] + queue

    features_str = "\n".join(
        f"  {i+1}. {f['feature_name']}: {f['pandas_code']}"
        for i, f in enumerate(all_features)
    )

    hint_section = (
        f"""HINT: "{state['objective']}"
Classification rule: classify by the GROUPING column, not the metric computed.
  - "location-based" → groupby on a geographic column (e.g. region)
  - "time-based"     → uses a date/time column (e.g. date)
  - "A OR B"         → match A, OR B, OR both
  - "A AND B"        → must match both

"""
        if state.get("objective") else "HINT: none — skip hint check.\n\n"
    )

    prompt = f"""You are a feature engineering reviewer. For EACH feature, reason step by step.

{hint_section}LEAKAGE RULES (always apply):
  UNSAFE — drop:
    shift(-N) with N > 0 → future rows
    transform('max') or transform('last') on a DATE column
    df['date'].max() or df['date'].last() globally
    expanding() windows

  SAFE — keep:
    transform('min') on date → first/earliest date per group → always past ✓
    transform('mean/sum/count') on numeric → cross-sectional ✓
    dt.dayofweek, dt.month, dt.day → current row only ✓
    shift(1) or shift(+N) → previous rows ✓

Features to review:
{features_str}

For each feature fill in:
  grouping_col, date_usage, leakage_check, hint_check, verdict ('keep'/'drop'), drop_reason
"""
    result: VerdictList = verdict_llm.invoke(prompt)

    kept    = []
    dropped = []

    for v in result.verdicts:
        print(f"[validate_plan] '{v.feature_name}':")
        print(f"  grouping   : {v.grouping_col}")
        print(f"  date_usage : {v.date_usage}")
        print(f"  leakage    : {v.leakage_check}")
        print(f"  hint       : {v.hint_check}")
        verdict_str = v.verdict.lower()
        if v.drop_reason:
            print(f"  verdict    : {verdict_str.upper()} ← {v.drop_reason}")
        else:
            print(f"  verdict    : {verdict_str.upper()} ✓")

        match = next((f for f in all_features if f["feature_name"] == v.feature_name), None)
        if match and verdict_str == "keep":
            kept.append(FeaturePlan(**match))
        elif match:
            dropped.append(f"{v.feature_name}: {v.drop_reason}")

    if not kept:
        print("[validate_plan] All features dropped — keeping original plan unchanged.")
        return {}

    print(f"\n[validate_plan] Kept {len(kept)}/{len(all_features)} features.")
    return {
        "plan":          kept[0],
        "feature_queue": [f.model_dump() for f in kept[1:]],
    }
