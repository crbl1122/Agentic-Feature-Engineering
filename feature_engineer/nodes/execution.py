"""
Execution nodes — validate_code, create_feature, validate, record_feature.
These are the only nodes that touch the DataFrame directly.
"""
import traceback

import numpy as np
import pandas as pd

from feature_engineer.security.ast_validator import assert_safe
from feature_engineer.state import AgentState
from feature_engineer.storage.parquet import path_to_df, thread_id_from_path, df_to_path

_EVAL_NAMESPACE = {
    "__builtins__": {},
    "pd": pd,
    "np": np,
    "int": int,
    "float": float,
    "str": str,
    "bool": bool,
    "isinstance": isinstance,   # needed for None-safe lambdas
    "len": len,                  # needed for string length checks
}


def validate_code(state: AgentState) -> dict:
    """AST-check the LLM-generated expression before execution."""
    plan = state["plan"]
    try:
        assert_safe(plan.pandas_code)
        print(f"[validate_code] '{plan.feature_name}' passed AST check.")
        return {"errors": []}
    except ValueError:
        err = traceback.format_exc()
        print(f"[validate_code] Blocked:\n{err}")
        return {"errors": [err]}


def create_feature(state: AgentState) -> dict:
    """Execute the LLM-generated pandas expression and attach the new column."""
    df   = path_to_df(state["df"]).copy()
    plan = state["plan"]

    # hard block — never overwrite an original column
    original_cols = state.get("original_columns", [])
    if plan.feature_name in original_cols:
        err = (
            f"Name collision: '{plan.feature_name}' is an original CSV column. "
            f"Propose a different name that reflects the transformation applied."
        )
        print(f"[create_feature] Blocked: {err}")
        return {"errors": [err], "attempts": state["attempts"] + 1}

    # auto-convert object columns that look like dates
    for col in df.columns:
        if df[col].dtype == object:
            try:
                df[col] = pd.to_datetime(df[col], format="mixed")
            except Exception:
                pass

    try:
        new_col = eval(plan.pandas_code, {**_EVAL_NAMESPACE, "df": df})  # noqa: S307
        df[plan.feature_name] = new_col
        print(f"[create_feature] Added column '{plan.feature_name}'")
        df_to_path(df, thread_id_from_path(state["df"]))
        return {}
    except Exception:
        err = traceback.format_exc()
        print(f"[create_feature] Error:\n{err}")
        return {"errors": [err], "attempts": state["attempts"] + 1}


def validate(state: AgentState) -> dict:
    """Check the new column for nulls, dtype issues, and constant values."""
    df   = path_to_df(state["df"])
    plan = state["plan"]
    col  = plan.feature_name

    issues: list[str] = []

    if col not in df.columns:
        issues.append(f"Column '{col}' was not created (likely an execution error).")
        return {"errors": issues}

    null_pct = df[col].isna().mean()
    if null_pct > 0.5:
        issues.append(f"'{col}' has {null_pct:.0%} null values — too many.")

    if df[col].nunique() <= 1:
        actual_val = df[col].iloc[0]
        issues.append(
            f"'{col}' is constant (every value is {actual_val!r}) — zero variance."
        )

    if hasattr(df[col].dtype, 'name') and 'timedelta' in str(df[col].dtype):
        issues.append(
            f"'{col}' is timedelta64 — ML models need numeric. "
            f"Use .dt.days // 365 to convert to integer years."
        )

    if issues:
        print(f"[validate] Issues: {issues}")
        return {"errors": issues}

    print(f"[validate] '{col}' passed. dtype={df[col].dtype}, "
          f"nulls={null_pct:.1%}, unique={df[col].nunique()}")
    return {"errors": []}


def record_feature(state: AgentState) -> dict:
    """Append the validated feature name, plan and formula to completed lists."""
    plan = state["plan"]
    print(f"[record_feature] Recording '{plan.feature_name}' as completed.")
    return {
        "completed_features": [plan.feature_name],
        "completed_plans":    [plan.model_dump()],
        "completed_formulas": [plan.pandas_code],
    }
