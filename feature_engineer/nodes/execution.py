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
    "isinstance": isinstance,
    "len": len,
    "set": set,
    "list": list,
    "dict": dict,
    "tuple": tuple,
}


def validate_code(state: AgentState) -> dict:
    """AST-check + deterministic dtype compatibility checks. Security only."""
    plan          = state["plan"]
    schema        = state.get("column_schema", {})
    failed        = state.get("failed_formulas", [])
    import re as _re

    # hard block — never retry a formula that already failed
    if plan.pandas_code in failed:
        err = f"Formula already failed — skipping: {plan.pandas_code}"
        print(f"[validate_code] SKIP '{plan.feature_name}': formula in failed list")
        return {"errors": [err]}

    # block numeric operations on string/categorical columns
    numeric_ops = [".var()", ".mean()", ".std()", ".sum()", ".min()", ".max()"]
    referenced  = _re.findall(r"df\['([^']+)'\]", plan.pandas_code)
    for col in referenced:
        if col not in schema:
            continue
        sem = schema[col].get("semantic_type", "")
        if sem in ("binary_string", "categorical", "identifier", "text"):
            for op in numeric_ops:
                if f"df['{col}']{op}" in plan.pandas_code or \
                   f"df[\"{col}\"]{op}" in plan.pandas_code:
                    err = (
                        f"Column '{col}' is {sem} (not numeric) — "
                        f"operation {op} is invalid on this dtype."
                    )
                    print(f"[validate_code] Blocked incompatible op: {err}")
                    return {"errors": [err]}

    # block groupby().apply(lambda).transform() — apply reduces dims, transform fails
    import re as _re2
    if _re2.search(r"\.apply\(lambda.*\)\.transform\(", plan.pandas_code):
        err = (
            "groupby().apply(lambda).transform() fails — apply() reduces to scalar per group "
            "before transform() can run. "
            "Use groupby().transform(lambda) directly, or rewrite as: "
            "df['col'].eq('value').groupby(df['group']).transform('mean')"
        )
        print(f"[validate_code] BLOCKED '{plan.feature_name}': {err}")
        print(f"[validate_code]   formula: {plan.pandas_code}")
        return {"errors": [err]}

    # block .unstack().stack() — returns MultiIndex Series incompatible with df index
    if ".unstack(" in plan.pandas_code and ".stack(" in plan.pandas_code:
        err = (
            "unstack().stack() returns a MultiIndex Series — incompatible with DataFrame index. "
            "Use groupby().transform() instead to keep alignment with df."
        )
        print(f"[validate_code] BLOCKED '{plan.feature_name}': {err}")
        print(f"[validate_code]   formula: {plan.pandas_code}")
        return {"errors": [err]}

    try:
        assert_safe(plan.pandas_code)
        print(f"[validate_code] '{plan.feature_name}' passed AST check.")
        print(f"[validate_code]   formula: {plan.pandas_code}")
        return {"errors": []}
    except ValueError:
        err = traceback.format_exc()
        print(f"[validate_code] BLOCKED '{plan.feature_name}' (AST unsafe):")
        print(f"[validate_code]   formula: {plan.pandas_code}")
        print(f"[validate_code]   reason:  {err.splitlines()[-1]}")
        return {"errors": [err]}


def _fix_groupby_transform(code: str) -> str:
    """Deterministically convert groupby().agg() → groupby().transform('agg')."""
    import re
    for agg in ("mean", "sum", "count", "min", "max", "std", "median", "nunique"):
        code = re.sub(
            rf"(df\.groupby\([^)]+\)\[[^\]]+\])\.{agg}\(\)",
            rf"\1.transform('{agg}')",
            code,
        )
    # fix .transform('nunique').transform('sum') → just .transform('nunique')
    code = re.sub(
        r"(df\.groupby\([^)]+\)\[[^\]]+\])\.transform\('nunique'\)\.transform\('[^']+'\)",
        r"\1.transform('nunique')",
        code,
    )
    # fix groupby([list]).size() → groupby([list])['first_col'].transform('count')
    # multi-key groupby .size() returns MultiIndex Series — incompatible with DataFrame index
    def _fix_multikey_size(m):
        groupby_part = m.group(1)
        # extract columns from groupby([...])
        cols_match = re.search(r"\[([^\]]+)\]", groupby_part)
        if cols_match:
            cols_str  = cols_match.group(1)
            # get first column name
            first_col = re.findall(r"['\"]([^'\"]+)['\"]", cols_str)
            if first_col:
                return f"{groupby_part}['{first_col[0]}'].transform('count')"
        return m.group(0)
    code = re.sub(
        r"(df\.groupby\(\[[^\]]+\]\))\.size\(\)(\.transform\(['\"][^'\"]*['\"]\))?",
        _fix_multikey_size,
        code,
    )
    # fix groupby('col')['col2'].apply(lambda x: x.eq('Y').mean()).transform(...)
    # → df['col2'].eq('Y').groupby(df['col']).transform('mean')
    def _fix_apply_eq_transform(m):
        groupby_col = m.group(1)
        target_col  = m.group(2)
        agg         = "mean" if "mean" in m.group(3) else "sum"
        return f"df['{target_col}'].eq('Y').groupby(df['{groupby_col}']).transform('{agg}')"
    code = re.sub(
        r"df\.groupby\(['\"]([^'\"]+)['\"]\)\[['\"]([^'\"]+)['\"]\]"
        r"\.apply\(lambda x: x\.eq\(['\"]Y['\"]\)\.(mean|sum)\(\)\)"
        r"\.transform\(['\"](?:mean|sum)['\"]\)",
        _fix_apply_eq_transform,
        code,
    )
    return code


def _fix_str_contains(code: str) -> str:
    """Add na=False to str.contains() calls that don't already have it."""
    import re
    # match .str.contains('...') or .str.contains("...") without na= already present
    def add_na_false(m):
        inner = m.group(1)
        if "na=" in inner:
            return m.group(0)
        return f".str.contains({inner}, na=False)"
    code = re.sub(r'\.str\.contains\(([^)]+)\)', add_na_false, code)
    return code


def _fix_case_sensitivity(code: str, schema: dict) -> str:
    """Fix string comparisons to use exact case from schema exact_values."""
    import re
    for col, info in schema.items():
        if info.get("semantic_type") not in ("categorical", "binary_string"):
            continue
        exact_vals = info.get("exact_values", [])
        if not exact_vals:
            continue
        # build map: lowercase → exact case
        case_map = {v.lower(): v for v in exact_vals if v and v != "None"}
        # find patterns like df['col'] == 'value' or df['col'] == "value"
        col_escaped = re.escape(col)
        pattern = rf"(df\[[\'\"]{{col_escaped}}[\'\"]\]\s*==\s*['\"])([^'\"]+)(['\"])"
        def fix_case(m, case_map=case_map):
            val = m.group(2)
            fixed = case_map.get(val.lower(), val)
            return m.group(1) + fixed + m.group(3)
        code = re.sub(
            rf"(df\['{{col_escaped}}'\]\s*==\s*['\"])([^'\"]+)(['\"])",
            fix_case, code
        )
        code = re.sub(
            rf'(df\["{{col_escaped}}"\]\s*==\s*["\'])([^"\']+)(["\'])',
            fix_case, code
        )
    return code


def _fix_string_concat(code: str, schema: dict) -> str:
    """Auto-add .astype(str) when concatenating a non-string column with strings."""
    import re
    def replace_col(m):
        col = m.group(1)
        info = schema.get(col, {})
        sem = info.get("semantic_type", "")
        if sem in ("numeric_continuous", "categorical_numeric", "binary_numeric"):
            return f"df['{col}'].astype(str)"
        return m.group(0)
    if " + '_' + " in code or "+'_'+" in code or " + \" \" + " in code:
        code = re.sub(r"df\['([^']+)'\]", replace_col, code)
    return code


def create_feature(state: AgentState) -> dict:
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
        return {"errors": [err], "attempts": state.get("attempts", 0) + 1}

    # hard block — never use target columns in formula
    target_cols = state.get("target_columns", [])
    used_targets = [c for c in target_cols if c in plan.pandas_code]
    if used_targets:
        err = (
            f"Target leakage: formula references target column(s) {used_targets}. "
            f"These columns must never be used in feature engineering."
        )
        print(f"[create_feature] Blocked: {err}")
        return {"errors": [err], "attempts": state.get("attempts", 0) + 1}

    # auto-convert object columns that look like dates
    for col in df.columns:
        if df[col].dtype == object:
            try:
                df[col] = pd.to_datetime(df[col], format="mixed")
            except Exception:
                pass

    schema = state.get("column_schema", {})

    # semantic block — get_dummies returns DataFrame not Series → revise_plan
    if "get_dummies" in plan.pandas_code:
        err = (
            "pd.get_dummies() returns a DataFrame not a Series — "
            "cannot assign as a single column. "
            "Use df['col'].astype('category').cat.codes for integer label encoding, "
            "or df['col'].map({val: idx, ...}) for explicit mapping."
        )
        print(f"[create_feature] Blocked get_dummies: {err}")
        return {"errors": [err], "attempts": state.get("attempts", 0) + 1}

    try:
        code = plan.pandas_code
        # apply deterministic fixes
        fixed_code = _fix_groupby_transform(code)
        fixed_code = _fix_str_contains(fixed_code)
        fixed_code = _fix_case_sensitivity(fixed_code, schema)
        fixed_code = _fix_string_concat(fixed_code, schema)
        if fixed_code != code:
            print(f"[create_feature] Auto-fixed: {code}")
            print(f"                         → {fixed_code}")
        new_col = eval(fixed_code, {**_EVAL_NAMESPACE, "df": df})  # noqa: S307
        df[plan.feature_name] = new_col
        print(f"[create_feature] Added column '{plan.feature_name}'")
        df_to_path(df, thread_id_from_path(state["df"]))
        return {}
    except Exception:
        err   = traceback.format_exc()
        short = err.splitlines()[-1]
        print(f"[create_feature] FAILED '{plan.feature_name}': {short}")
        print(f"[create_feature]   formula: {plan.pandas_code}")
        return {"errors": [err], "attempts": state.get("attempts", 0) + 1}


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

    # low variance — numeric feature with only 2 unique values is likely trivial
    # exception: binary features from Y/N encoding are valid
    if (df[col].dtype in ['int64', 'int32', 'float64'] and
            df[col].nunique() == 2 and
            plan.pandas_code and
            'nunique' in plan.pandas_code):
        issues.append(
            f"'{col}' has only 2 unique values from a nunique aggregation — "
            f"likely all groups have the same count. Zero predictive variance."
        )

    if hasattr(df[col].dtype, 'name') and 'timedelta' in str(df[col].dtype):
        issues.append(
            f"'{col}' is timedelta64 — ML models need numeric. "
            f"Use .dt.days // 365 to convert to integer years."
        )

    # high cardinality warning for string/object features
    if df[col].dtype == object and df[col].nunique() > len(df) * 0.05:
        print(f"[validate] Warning: '{col}' has high cardinality "
              f"({df[col].nunique()} unique / {len(df)} rows = "
              f"{df[col].nunique()/len(df):.1%}) — consider encoding before ML.")

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
