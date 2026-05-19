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

    target_cols_set = set(state.get("target_columns", []))
    schema_lines = [
        f"  {col!r}: dtype={df[col].dtype}, sample={df[col].dropna().head(3).tolist()}"
        for col in df.columns
        if col not in target_cols_set
    ]
    schema_str = "\n".join(schema_lines)

    objective = state.get("objective", "")
    hint_block = (
        f"\nYour PRIMARY objective is: {objective}\n"
        f"All features MUST be directly related to this objective.\n"
        if objective
        else "\nDesign features that would add the most predictive value.\n"
    )

    # use ranked_features if available, fallback to feasible_features
    feasible = state.get("ranked_features") or state.get("feasible_features", [])
    formula_hints = state.get("research_formula_hints", {})

    # clean feasible feature labels
    def _clean_feasible(f: str) -> str:
        base = f.split(" (")[0].strip()
        name = base.split(":")[0].strip()
        return name.lower().replace(" ", "_") + (": " + ":".join(base.split(":")[1:]).strip() if ":" in base else "")

    feasible_clean = [_clean_feasible(f) for f in feasible]

    # build feasible block with formula hints when available
    if feasible_clean:
        feasible_lines = []
        for f in feasible_clean:
            name = f.split(":")[0].strip()
            hint = next(
                (v for k, v in formula_hints.items()
                if k.split(":")[0].strip().lower().replace(" ", "_") == name),
                ""
            )
            if hint:
                feasible_lines.append(f"  - {f}  →  USE THIS FORMULA: {hint}")
            else:
                feasible_lines.append(f"  - {f}")
        feasible_block = (
            "\nPrioritise generating features from this researched list. "
            "Where a formula hint is provided, USE IT as the basis — do not invent a different formula:\n"
            + "\n".join(feasible_lines) + "\n"
        )
    else:
        feasible_block = ""

    # detect natural grouping columns
    df_cols_for_grouping = [
        col for col in df.columns
        if df[col].dtype == object and 2 <= df[col].nunique() <= 100
    ]
    grouping_block = ""
    if df_cols_for_grouping:
        grouping_block = (
            f"\nGROUPING COLUMNS DETECTED: {df_cols_for_grouping}\n"
            "Consider generating per-group variants of numeric features using these columns.\n"
            "Example: if 'country' has 50 unique values → "
            "df['energy_consumption'].groupby(df['country']).transform('mean') "
            "gives mean energy per country — more informative than a global stat.\n"
        )

    from feature_engineer.nodes.schema_analyzer import format_schema_for_prompt
    schema       = state.get("column_schema", {})
    target_cols  = state.get("target_columns", [])
    schema_block = ""
    if schema:
        schema_block = "\n" + format_schema_for_prompt(schema) + "\n"
    target_block = ""
    if target_cols:
        target_block = (
            f"\nTARGET COLUMNS — never use in any feature: {target_cols}\n"
        )

    completed_formulas = state.get("completed_formulas", [])
    failed_formulas    = state.get("failed_formulas", [])
    failed_formulas    = state.get("failed_formulas", [])

    # detect feature types in feasible_clean for diversity enforcement
    has_rolling = any("rolling" in f.lower() for f in feasible_clean)
    has_lag     = any(("lag" in f.lower() or "shift" in f.lower() or "previous" in f.lower()) for f in feasible_clean)
    has_ratio   = any(("ratio" in f.lower() or "per_capita" in f.lower() or "change" in f.lower()) for f in feasible_clean)

    diversity_lines = []
    if has_rolling:
        diversity_lines.append("  - at least 1 rolling window feature (rolling mean, rolling std, etc.)")
    if has_lag:
        diversity_lines.append("  - at least 1 lag/shift feature (previous value, lagged value, etc.)")
    if has_ratio:
        diversity_lines.append("  - at least 1 ratio/difference/per-capita feature")

    diversity_block = ""
    if diversity_lines:
        diversity_block = (
            "\nDIVERSITY RULE: if feasible_features contains rolling, lag, or ratio features, "
            "include at least one of each type present.\n"
        )

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
{schema_block}{target_block}{hint_block}{feasible_block}{grouping_block}{diversity_block}{blacklist_block}
Generate between 1 and {max_f} new feature columns.
For each feature return a FeaturePlan with:
- feature_name: short snake_case column name
- description: what it represents
- pandas_code: a single Python *expression* using only `df` that returns a pandas Series.
  Do NOT import anything. Do NOT assign variables.
  If a formula hint is provided above — USE IT. Do not substitute a different formula.

MATH CORRECTNESS RULE:
  The formula must compute exactly what the description says.
  WRONG: description='energy per urban pop', code=df['energy'] / df['urban_pop'] * df['energy']
  RIGHT: description='energy per urban pop', code=df['energy'] / df['urban_pop']
  Before writing the formula, verify: does this expression produce the described quantity?

GROUPBY RULE: if description says "per X", use groupby + transform.
  Example: "total sales per region" → df.groupby('region')['unit_price'].transform('sum')

DATETIME RULE: wrap BOTH sides with pd.to_datetime() when subtracting dates.
  RIGHT: pd.to_datetime(df['date']) - pd.to_datetime(df.groupby('x')['date'].transform('min'))

APPLY vs TRANSFORM RULE (CRITICAL — applies to ALL groupby aggregations):
  NEVER use groupby() without transform() to create a column — it changes the index and produces NaN.
  This applies to: .mean(), .sum(), .count(), .min(), .max(), .std(), .apply()
  ALWAYS use .transform() to keep the original DataFrame index:

  WRONG: df.groupby('x')['a'].mean()           → Series indexed by x → NaN when assigned
  RIGHT: df.groupby('x')['a'].transform('mean') → Series indexed by df → correct ✓

  WRONG: df.groupby('x')['a'].sum()            → NaN
  RIGHT: df.groupby('x')['a'].transform('sum') → correct ✓

  WRONG: df.groupby('x').apply(lambda x: x['a'].sum()) → NaN
  RIGHT: (df['a']).groupby(df['x']).transform('sum')   → correct ✓

  RULE: if you need a per-group statistic as a new column, ALWAYS use transform().

PD.CUT RULE: labels must be exactly len(bins) - 1.
  bins=[a, b, c, d] → 3 intervals → labels must have 3 elements.
  WRONG: bins=[-inf, 0, 15, 25, 35, +inf], labels=['a','b','c','d'] → 5 bins, 4 labels ← ERROR
  RIGHT: bins=[-inf, 0, 15, 25, 35, +inf], labels=['a','b','c','d','e'] → 5 bins, 5 labels ✓
  Always count: N bin edges → N-1 intervals → N-1 labels.

DAILY DATE RULE: if date column contains daily data (no time component),
  .dt.hour / .dt.minute / .dt.second will always be 0 — useless.
  Use .dt.dayofweek, .dt.month, .dt.quarter, .dt.dayofyear instead.
  Check sample values before choosing a date feature.
"""
    result   = planner_llm.invoke(prompt)
    features = result.features[:max_f]

    # Build lookup: snake_case name → original description from feasible_features
    # e.g. "total momentum of particles" → "total_momentum"
    feasible_name_map: dict[str, str] = {}
    for f_str in feasible_clean:
        concept = f_str.split(" (")[0].split(":")[0].strip().lower().replace(" ", "_")
        feasible_name_map[concept] = concept

    objective      = state.get("objective", "")
    original_cols  = state.get("original_columns", [])
    # used_names = original CSV columns + already completed features
    # → derive_feature_name will never return a name that collides with either
    already_used   = list(original_cols) + list(state.get("completed_features", []))

    for f in features:
        formula_lower = f.pandas_code.lower()
        matched_name  = None
        for concept in feasible_name_map:
            concept_words = concept.replace("_", " ").split()
            if any(w in formula_lower for w in concept_words if len(w) > 3):
                if concept not in already_used:
                    matched_name = concept
                break

        name = matched_name if matched_name else \
            derive_feature_name(f.pandas_code, context=objective, used_names=already_used)

        f.feature_name = name
        already_used.append(name)

    print(f"[feature_planner] hint: {objective or '(none)'}")

    # deduplicate by pandas_code before queueing
    seen_codes = set()
    unique_features = []
    for f in features:
        if f.pandas_code not in seen_codes:
            seen_codes.add(f.pandas_code)
            unique_features.append(f)
        else:
            print(f"[feature_planner] Duplicate formula removed: {f.feature_name} ({f.pandas_code})")
    features = unique_features

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

    df              = path_to_df(state["df"])
    existing_cols   = df.columns.tolist()
    schema          = state.get("column_schema", {})
    target_cols     = state.get("target_columns", [])
    n_rows          = len(df)
    low_card_thresh = max(10, n_rows // 100)   # e.g. 36540 rows → threshold = 365

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

{hint_section}TRIVIAL FEATURE RULE (always applies):
  Reject any feature whose pandas_code is just df['column_name'] with no transformation.
  Copying an existing column adds zero new information.

NAME COLLISION RULE (always applies):
  Reject any feature whose feature_name is identical to an existing column name.
  Existing columns: {existing_cols}
  The new feature must have a DIFFERENT name that reflects the transformation applied.

MATH CORRECTNESS RULE (always applies):
  The formula must compute exactly what the description says.
  Read the description, then verify: does the expression produce that exact quantity?
  WRONG: description='energy per urban pop', code=df['energy'] / df['urban_pop'] * df['energy']
         → multiplies energy twice — not energy per urban pop → DROP
  RIGHT: description='energy per urban pop', code=df['energy'] / df['urban_pop'] → correct ✓
  WRONG: description='ratio of A to B', code=df['A'] * df['B'] → that is a product not a ratio
  RIGHT: description='ratio of A to B', code=df['A'] / df['B'] → correct ✓
  If the formula does not match the description → DROP with reason.

HIGH CARDINALITY STRING RULE (always applies):
  String concatenation of high-cardinality columns produces features too sparse for ML.
  If formula is a string concatenation (col1 + '_' + col2):
    - If ALL concatenated columns have < 50 unique values → KEEP
    - If ANY concatenated column has >= 50 unique values → DROP
  Examples:
    msi (2 unique) + growth (3 unique) → 6 max combinations → KEEP ✓
    cna (2) + gene_expression (2) → 4 combinations → KEEP ✓
    target (185 unique) + tissue (19 unique) → too many → DROP
    drug_name (286) + tissue (19) → too many → DROP

  Reject features that produce very few unique values relative to dataset size ({n_rows} rows).
  If a groupby transform produces <= {low_card_thresh} unique values for {n_rows} rows → DROP.
  This indicates the feature has insufficient predictive variation.

  UNSAFE — drop:
    shift(-N) where N > 0 → FUTURE rows (NEGATIVE shift only)
      shift(-1) → tomorrow → LEAKAGE
      shift(-2) → 2 days ahead → LEAKAGE
    transform('max') or transform('last') on a DATE column
    df['date'].max() or df['date'].last() globally
    expanding() windows

  SAFE — keep:
    shift(N) or shift(+N) where N > 0 → PAST rows → ALWAYS SAFE ✓
      shift(1)  → yesterday → SAFE
      shift(2)  → 2 days ago → SAFE
      shift(7)  → last week → SAFE
    CRITICAL: shift with POSITIVE number is NEVER leakage — it looks backward.
    Only NEGATIVE shift (shift(-1), shift(-2)) is leakage.
    rolling(window=N) → looks backward only → SAFE ✓
    transform('min') on date → first/earliest date per group → SAFE ✓
    transform('mean/sum/count') on numeric → cross-sectional → SAFE ✓
    dt.dayofweek, dt.month, dt.day → current row only → SAFE ✓

Features to review:
{features_str}

For each feature fill in:
  grouping_col, date_usage, leakage_check, hint_check, verdict ('keep'/'drop'), drop_reason
"""
    result: VerdictList = verdict_llm.invoke(prompt)

    kept    = []
    dropped = []

    # build lookup for deterministic overrides
    feature_map = {f["feature_name"]: f for f in all_features}

    for v in result.verdicts:
        print(f"[validate_plan] '{v.feature_name}':")
        print(f"  grouping   : {v.grouping_col}")
        print(f"  date_usage : {v.date_usage}")
        print(f"  leakage    : {v.leakage_check}")
        print(f"  hint       : {v.hint_check}")
        verdict_str = v.verdict.lower()

        match = feature_map.get(v.feature_name)

        # DETERMINISTIC: target leakage — simple substring check
        if match:
            code         = match.get("pandas_code", "")
            used_targets = [c for c in target_cols if c in code]
            if used_targets:
                reason = f"Target leakage: formula contains target column(s) {used_targets}"
                print(f"  verdict    : OVERRIDE → DROP ← {reason}")
                dropped.append(f"{v.feature_name}: {reason}")
                continue

        # DETERMINISTIC: high cardinality string concat check
        # override LLM verdict based on actual cardinality from column_schema
        if match:
            code = match.get("pandas_code", "")
            import re as _re
            is_concat = " + '_' + " in code or "+'_'+" in code
            if is_concat:
                cols_in_concat = _re.findall(r"df\['([^']+)'\]", code)
                cardinalities  = [schema.get(c, {}).get("unique", 9999) for c in cols_in_concat]
                product_card   = 1
                for c in cardinalities:
                    product_card *= c
                threshold = max(100, n_rows // 100)
                if product_card <= threshold and verdict_str == "drop" and "cardinality" in (v.drop_reason or "").lower():
                    print(f"  verdict    : OVERRIDE → KEEP ✓ "
                          f"(product cardinality {product_card} ≤ {threshold}: "
                          f"{list(zip(cols_in_concat, cardinalities))})")
                    kept.append(FeaturePlan(**match))
                    continue
                elif product_card > threshold and verdict_str == "keep":
                    reason = (f"High cardinality concat: product={product_card} > threshold={threshold} "
                              f"(cols: {list(zip(cols_in_concat, cardinalities))})")
                    print(f"  verdict    : OVERRIDE → DROP ← {reason}")
                    dropped.append(f"{v.feature_name}: {reason}")
                    continue
        if verdict_str == "drop" and "low cardinality" in (v.drop_reason or "").lower():
            code = match.get("pandas_code", "") if match else ""
            # cohort statistics are valid even with few unique values
            cohort_patterns = [
                "nunique", "transform('mean')", "transform(\"mean\")",
                "transform('sum')", "transform(\"sum\")",
                "transform('count')", "transform(\"count\")",
                ".eq('Y')", '.eq("Y")', "value_counts",
            ]
            if any(p in code for p in cohort_patterns):
                print(f"  verdict    : OVERRIDE → KEEP ✓ (cohort statistic — low cardinality is expected)")
                kept.append(FeaturePlan(**match))
                continue
        if match and verdict_str == "drop" and "shift" in v.drop_reason.lower():
            code = match.get("pandas_code", "")
            import re as _re
            # check if shift argument is positive (no minus sign)
            shift_args = _re.findall(r"\.shift\(([^)]+)\)", code)
            if shift_args and not any(a.strip().startswith("-") for a in shift_args):
                print(f"  verdict    : OVERRIDE → KEEP ✓ (shift with positive N is never leakage)")
                kept.append(FeaturePlan(**match))
                continue

        if v.drop_reason:
            print(f"  verdict    : {verdict_str.upper()} ← {v.drop_reason}")
        else:
            print(f"  verdict    : {verdict_str.upper()} ✓")

        if match is None:
            print(f"  ⚠ WARNING: no match found for feature_name '{v.feature_name}' — "
                  f"available names: {[f['feature_name'] for f in all_features]}")
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
