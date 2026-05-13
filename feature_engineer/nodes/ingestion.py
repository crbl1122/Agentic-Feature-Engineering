"""
load_csv node — reads CSV, caches to parquet, resets state.
"""
import os

import pandas as pd

from feature_engineer.state import AgentState
from feature_engineer.storage.parquet import df_to_path


def load_csv(state: AgentState) -> dict:
    """Read CSV from disk, write to parquet in df_store/, return path."""
    df        = pd.read_csv(state["input_path"])
    thread_id = os.path.splitext(os.path.basename(state["output_path"]))[0]
    path      = df_to_path(df, thread_id)

    print(f"[load_csv] Loaded {len(df)} rows × {len(df.columns)} cols")
    print(f"[load_csv] DataFrame cached → {path}")

    return {
        "df":               path,
        "original_columns": df.columns.tolist(),
        "errors":           [],
        "attempts":         0,
        "plan":             None,
    }
