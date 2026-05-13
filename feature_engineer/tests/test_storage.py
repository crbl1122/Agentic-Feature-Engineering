"""
Unit tests for storage layer — in-memory SQLite, temp parquet files.
"""
import os
import tempfile

import pandas as pd
import pytest

from feature_engineer.storage.database import init_db, save_run, load_history, load_failed_runs
from feature_engineer.storage.parquet import df_to_path, path_to_df


# ── Database tests ───────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_db(tmp_path):
    db = str(tmp_path / "test.db")
    init_db(db)
    return db


def test_init_db_creates_table(tmp_db):
    import sqlite3
    with sqlite3.connect(tmp_db) as con:
        tables = [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")]
    assert "runs" in tables


def test_save_and_load_run(tmp_db):
    state = {
        "input_path":      "sales.csv",
        "objective":       "predict sales",
        "completed_features": ["total_revenue"],
        "completed_plans": [{"feature_name": "total_revenue", "description": "Revenue",
                              "pandas_code": "df['a'] * df['b']"}],
        "attempts":        1,
        "plan":            None,
    }
    save_run("thread-001", state, "success", db_path=tmp_db)
    df = load_history(db_path=tmp_db)
    assert len(df) == 1
    assert df.iloc[0]["thread_id"] == "thread-001"
    assert df.iloc[0]["status"] == "success"


def test_load_failed_runs(tmp_db):
    state = {"input_path": "x.csv", "objective": "", "completed_features": [],
             "completed_plans": [], "attempts": 0, "plan": None}
    save_run("thread-fail", state, "failed", db_path=tmp_db)
    save_run("thread-ok",   state, "success", db_path=tmp_db)
    failed = load_failed_runs(db_path=tmp_db)
    assert len(failed) == 1
    assert "thread-fail" in failed[0]


# ── Parquet tests ────────────────────────────────────────────────────────────────

def test_df_round_trip(tmp_path):
    df   = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
    path = df_to_path(df, "test-thread")
    result = path_to_df(path)
    pd.testing.assert_frame_equal(df, result)
    os.remove(path)
