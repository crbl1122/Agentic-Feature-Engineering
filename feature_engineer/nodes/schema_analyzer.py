"""
schema_analyzer node — pure Python, zero LLM, zero cost.

Reads the DataFrame and builds a structured column profile:
  - dtype
  - semantic_type (inferred deterministically)
  - sample_values
  - exact_values (for low-cardinality columns)
  - unique count
  - null_pct
  - is_target (from objective text parsing)

Also parses target_columns from the objective text.
Injects column_schema and target_columns into state for all downstream nodes.
"""
import re

import pandas as pd

from feature_engineer.state import AgentState
from feature_engineer.storage.parquet import path_to_df


# ── Semantic type inference ─────────────────────────────────────────────────────

def _infer_semantic_type(col: pd.Series) -> str:
    """Deterministically infer semantic type from a pandas Series."""
    dtype = col.dtype

    if pd.api.types.is_datetime64_any_dtype(col):
        return "datetime"

    if pd.api.types.is_numeric_dtype(col):
        nunique = col.nunique()
        if nunique == 2:
            return "binary_numeric"
        if nunique <= 20:
            return "categorical_numeric"
        return "numeric_continuous"

    if pd.api.types.is_object_dtype(col) or pd.api.types.is_string_dtype(col):
        non_null = col.dropna()
        if len(non_null) == 0:
            return "unknown"

        nunique = col.nunique(dropna=True)
        sample  = non_null.unique()[:10].tolist()

        # binary string: Y/N, Yes/No, True/False, 0/1 as strings
        binary_sets = [
            {"y", "n"}, {"yes", "no"}, {"true", "false"},
            {"0", "1"}, {"positive", "negative"},
        ]
        sample_lower = {str(v).lower().strip() for v in sample if v is not None}
        if sample_lower <= {"y", "n"} or sample_lower <= {"yes", "no"} or \
           sample_lower <= {"true", "false"} or sample_lower <= {"0", "1"}:
            return "binary_string"

        # date-like strings
        date_pattern = re.compile(r"\d{4}-\d{2}-\d{2}")
        if any(date_pattern.match(str(v)) for v in sample[:3] if v):
            return "date_string"

        # high cardinality → free text or identifier
        total = len(non_null)
        if nunique > total * 0.9:
            return "identifier"

        return "categorical"

    return "other"


# ── Target column extraction from objective ─────────────────────────────────────

def _parse_target_columns(objective: str, column_names: list[str]) -> list[str]:
    """Extract target columns from objective text by matching known column names.

    Looks for patterns like:
      - "TARGET VARIABLE"
      - "do not use"
      - "target column"
      - "target:"
    """
    targets = []
    obj_lower = objective.lower()

    # find lines that mention "target", "do not use", "never use"
    trigger_patterns = [
        r"target\s+variable",
        r"do\s+not\s+use",
        r"never\s+use",
        r"target\s+column",
        r"predict\s+.*?:",
    ]

    for col in column_names:
        col_lower = col.lower()
        # find the position of the column name in objective
        pos = obj_lower.find(col_lower)
        if pos == -1:
            continue
        # check surrounding context (±100 chars) for target indicators
        context = obj_lower[max(0, pos - 100): pos + len(col) + 100]
        for pattern in trigger_patterns:
            if re.search(pattern, context):
                targets.append(col)
                break

    return targets


# ── Main node ───────────────────────────────────────────────────────────────────

def schema_analyzer(state: AgentState) -> dict:
    """Build column schema profile and extract target columns."""
    df        = path_to_df(state["df"])
    objective = state.get("objective", "")

    schema = {}
    for col in df.columns:
        series          = df[col]
        dtype           = str(series.dtype)
        semantic_type   = _infer_semantic_type(series)
        null_pct        = round(series.isna().mean() * 100, 1)
        nunique         = int(series.nunique(dropna=True))
        sample          = series.dropna().head(5).tolist()

        entry: dict = {
            "dtype":         dtype,
            "semantic_type": semantic_type,
            "null_pct":      float(round(null_pct, 1)),
            "unique":        int(nunique),
            "sample_values": [v.item() if hasattr(v, 'item') else v for v in sample],
            "is_target":     False,
        }

        if nunique <= 30:
            entry["exact_values"] = sorted(
                [str(v) for v in series.dropna().unique().tolist()]
            )

        schema[col] = entry

    # use target_columns from state (set by user via UI/CLI)
    all_targets = list(state.get("target_columns", []))

    for col in all_targets:
        if col in schema:
            schema[col]["is_target"] = True

    print(f"[schema_analyzer] Profiled {len(schema)} columns:")
    for col, info in schema.items():
        target_flag = " ← TARGET" if info["is_target"] else ""
        print(f"  {col!r}: {info['semantic_type']} ({info['unique']} unique, "
              f"{info['null_pct']}% null){target_flag}")

    if all_targets:
        print(f"[schema_analyzer] Target columns (excluded from features): {all_targets}")

    return {
        "column_schema":  schema,
        "target_columns": all_targets,
    }


def format_schema_for_prompt(schema: dict, max_cols: int = 20) -> str:
    """Format column schema as a compact prompt-friendly string."""
    lines = ["Column schema:"]
    for col, info in list(schema.items())[:max_cols]:
        target_note = " [TARGET — do not use]" if info.get("is_target") else ""
        type_note   = info["semantic_type"]
        null_note   = f", {info['null_pct']}% null" if info["null_pct"] > 0 else ""

        if info["semantic_type"] in ("binary_string", "categorical") and \
                "exact_values" in info and info["unique"] <= 10:
            vals = info["exact_values"][:6]
            vals_str = f", values: {vals}"
        else:
            vals_str = f", sample: {info['sample_values'][:3]}"

        lines.append(f"  {col!r}: {type_note}{null_note}{vals_str}{target_note}")

    return "\n".join(lines)
