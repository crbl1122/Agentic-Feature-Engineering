"""
DataFrame persistence — parquet files in df_store/.
State carries only the file path; data never serialised into SQLite.
"""
import os

import pandas as pd

from feature_engineer.config import PARQUET_DIR, OUTPUT_DIR


def init_dirs() -> None:
    """Create df_store/ and output/ if they don't exist."""
    os.makedirs(PARQUET_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR,  exist_ok=True)


def df_to_path(df: pd.DataFrame, thread_id: str) -> str:
    """Write DataFrame to a per-thread parquet file and return the path."""
    init_dirs()
    path = os.path.join(PARQUET_DIR, f"{thread_id}.parquet")
    df.to_parquet(path, index=False)
    return path


def path_to_df(path: str) -> pd.DataFrame:
    """Read DataFrame from a parquet file."""
    return pd.read_parquet(path)


def thread_id_from_path(path: str) -> str:
    """Extract thread_id from a parquet path (filename without extension)."""
    return os.path.splitext(os.path.basename(path))[0]
