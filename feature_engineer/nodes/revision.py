"""
revise_plan node — asks LLM to fix a failed expression with full context.
"""
from feature_engineer.llm.naming import derive_feature_name
from feature_engineer.llm.setup import structured_llm
from feature_engineer.state import AgentState, FeaturePlan
from feature_engineer.storage.parquet import path_to_df


def revise_plan(state: AgentState) -> dict:
    """Ask the LLM to fix the broken expression and return an updated plan."""
    plan               = state["plan"]
    df                 = path_to_df(state["df"])
    last_error         = " | ".join(state["errors"]) if state["errors"] else "unknown error"
    failed_formulas    = state.get("failed_formulas", [])
    completed_formulas = state.get("completed_formulas", [])

    all_failed = failed_formulas + [plan.pandas_code]
    print(f"[revise_plan] Error received: {last_error[:300]}")
    print(f"[revise_plan] Failed formulas blacklist ({len(all_failed)}): {all_failed}")

    # detect columns referenced in formula that don't exist in df
    import re
    referenced_cols = re.findall(r"df\['([^']+)'\]", plan.pandas_code)
    missing_cols    = [c for c in referenced_cols if c not in df.columns]

    if missing_cols:
        print(f"[revise_plan] ⚠ Missing columns: {missing_cols} — not in dataset")
        print(f"[revise_plan]   Available columns: {df.columns.tolist()}")

    # schema with group sizes + special detail for referenced columns
    schema_lines = []
    for col in df.columns:
        if col == plan.feature_name:
            continue
        dtype  = df[col].dtype
        counts = df[col].value_counts()
        if col in referenced_cols:
            # show more detail for columns used in the failing formula
            sample       = df[col].dropna().head(10).tolist()
            has_non_num  = df[col].apply(lambda x: not str(x).replace('.','').replace('-','').isdigit()).any()
            non_num_vals = df[col][df[col].apply(lambda x: not str(x).replace('.','').replace('-','').isdigit())].unique()[:5].tolist() if has_non_num else []
            schema_lines.append(
                f"  {col!r}: dtype={dtype}, sample={sample}"
                + (f", ⚠ NON-NUMERIC VALUES: {non_num_vals}" if non_num_vals else "")
            )
            if non_num_vals:
                print(f"[revise_plan] ⚠ Column '{col}' has non-numeric values: {non_num_vals}")
        elif df[col].nunique() <= 20:
            unique_vals = df[col].dropna().unique().tolist()
            min_c = int(counts.min()) if len(counts) > 0 else 0
            max_c = int(counts.max()) if len(counts) > 0 else 0
            schema_lines.append(
                f"  {col!r}: dtype={dtype}, unique values={unique_vals}, "
                f"rows per value: min={min_c} max={max_c}"
            )
        else:
            schema_lines.append(
                f"  {col!r}: dtype={dtype}, sample={df[col].dropna().head(3).tolist()}"
            )

    # determine action based on error type — priority order matters
    objective = state.get("objective", "")
    if missing_cols:
        action = (
            f"The formula references columns that DO NOT EXIST in the dataset: {missing_cols}\n\n"
            f"Available columns are: {df.columns.tolist()}\n\n"
            "You MUST rewrite the formula using ONLY the available columns listed above.\n"
            "Do NOT use fillna() or any workaround — the column simply does not exist.\n"
            "If the concept cannot be computed without the missing column, "
            "propose a COMPLETELY DIFFERENT feature that uses only available columns."
        )
    elif "__duplicate__" in last_error:
        hint_instruction = (
            f" The new feature MUST respect the objective: {objective}."
            if objective else ""
        )
        action = (
            "This feature is a duplicate of one already completed."
            f"{hint_instruction} Propose a COMPLETELY DIFFERENT feature with a new "
            "formula and new name. Do NOT reuse any formula from the blacklists below."
        )
    elif "invalid literal" in last_error or "could not convert" in last_error or "non-numeric" in last_error.lower():
        action = (
            "The column contains NON-NUMERIC values (e.g. 'N.V.', 'Unknown', text strings) "
            "that cannot be cast directly to int or parsed as dates.\n\n"
            "Use pd.to_numeric(df['col'], errors='coerce') to safely convert.\n\n"
            "EXAMPLE FIX:\n"
            "  WRONG: 2023 - df['year'].astype(int)  ← fails on 'N.V.'\n"
            "  RIGHT: 2023 - pd.to_numeric(df['year'], errors='coerce')\n\n"
            "Check the ⚠ NON-NUMERIC VALUES in the schema below to understand what values exist."
        )
    elif "constant" in last_error or "zero variance" in last_error:
        hint_instruction = (
            f" The new feature MUST respect the objective: {objective}."
            if objective else ""
        )
        action = (
            "The expression produces a constant column — the data does not support "
            f"this feature.{hint_instruction} Propose a COMPLETELY DIFFERENT feature.\n\n"
            "MANDATORY: Update feature_name AND description to match the new feature.\n\n"
            "Do NOT use transform('min'), transform('max'), or per-group subtraction "
            "when every group has only one row."
        )
    else:
        action = "Fix the expression so it returns a valid pandas Series."

    failed_block = (
        "\nAlready FAILED formulas — do NOT reuse:\n"
        + "\n".join(f"  - {c}" for c in all_failed) + "\n"
    )
    completed_block = (
        "\nAlready ACCEPTED formulas — do NOT reuse or propose semantically equivalent:\n"
        + "\n".join(f"  - {f}" for f in completed_formulas) + "\n"
        if completed_formulas else ""
    )
    hint_block = (
        f"\nObjective (the new feature MUST respect this): {objective}\n"
        if objective else ""
    )

    prompt = f"""You are a pandas expert.
{hint_block}
feature_name: {plan.feature_name}
Description : {plan.description}
Expression  : {plan.pandas_code}

Problem:
  {last_error}
{failed_block}{completed_block}
Action required:
  {action}

RULES:
  1. Must return a pandas Series (single column, not a DataFrame).
  2. Do NOT use pd.get_dummies(), pd.concat, or pd.DataFrame.
  3. Do NOT use placeholder strings — use real values from the schema below.
  4. DATETIME RULE: wrap BOTH sides with pd.to_datetime() when subtracting dates.
  5. APPLY vs TRANSFORM RULE: NEVER use groupby().apply() — it changes the index → NaN.
     ALWAYS use transform() or pre-multiply then transform:
     WRONG: df.groupby('x').apply(lambda x: (x['a']*x['b']).sum()) → NaN
     RIGHT: (df['a'] * df['b']).groupby(df['x']).transform('sum')  → correct

  6. PD.CUT RULE: labels must be exactly len(bins) - 1.
     bins has N edges → N-1 intervals → N-1 labels required.
     WRONG: bins=[-inf,0,15,25,35,+inf], labels=['a','b','c','d'] → 5 bins, 4 labels ← ERROR
     RIGHT: bins=[-inf,0,15,25,35,+inf], labels=['a','b','c','d','e'] → 5 bins, 5 labels ✓

  7. DAILY DATE RULE: if date sample shows no time (e.g. 2020-01-01 00:00:00),
     .dt.hour is always 0 — useless. Use .dt.dayofweek, .dt.month, .dt.quarter instead.

DataFrame schema with actual unique values and group sizes:
{chr(10).join(schema_lines)}

Return a corrected FeaturePlan. feature_name will be derived from the formula automatically.
"""
    revised: FeaturePlan = structured_llm.invoke(prompt)
    revised.feature_name = derive_feature_name(
        revised.pandas_code,
        context=state.get("objective", ""),
        used_names=list(state.get("completed_features", [])),
    )

    # deterministic guard — reject if LLM proposed an already used formula
    all_used = set(completed_formulas) | set(all_failed)
    if revised.pandas_code in all_used:
        print(f"[revise_plan] LLM proposed duplicate formula — rejecting: {revised.pandas_code}")
        return {
            "errors":          [f"Duplicate formula proposed: {revised.pandas_code}"],
            "attempts":        state["attempts"] + 1,
            "failed_formulas": [plan.pandas_code],
        }

    print(f"[revise_plan] Attempt {state['attempts'] + 1}/3 — "
          f"name: {revised.feature_name} code: {revised.pandas_code}")
    return {
        "plan":            revised,
        "errors":          [],
        "attempts":        state["attempts"] + 1,
        "failed_formulas": [plan.pandas_code],
    }
