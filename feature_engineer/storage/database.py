"""
SQLite persistence — runs history + feature memory.
LangGraph checkpoints are managed separately by AsyncSqliteSaver.
"""
import json
import os
import sqlite3
from datetime import datetime

import pandas as pd

from feature_engineer.config import DB_PATH


def init_db(db_path: str = DB_PATH) -> None:
    """Create runs table and feature_memory table, migrate schema if needed."""
    with sqlite3.connect(db_path) as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS runs (
                thread_id     TEXT PRIMARY KEY,
                timestamp     TEXT,
                input_file    TEXT,
                hint          TEXT,
                feature_name  TEXT,
                description   TEXT,
                pandas_code   TEXT,
                attempts      INTEGER,
                status        TEXT,
                features_json TEXT
            )
        """)
        cols = [r[1] for r in con.execute("PRAGMA table_info(runs)").fetchall()]
        if "features_json" not in cols:
            con.execute("ALTER TABLE runs ADD COLUMN features_json TEXT")

        # ── Feature memory table ────────────────────────────────────────────
        con.execute("""
            CREATE TABLE IF NOT EXISTS feature_memory (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                feature_name TEXT NOT NULL,
                formula      TEXT NOT NULL UNIQUE,
                objective    TEXT NOT NULL,
                created_at   TEXT DEFAULT (datetime('now'))
            )
        """)
        con.execute("CREATE INDEX IF NOT EXISTS idx_fm_formula ON feature_memory(formula)")


def save_run(thread_id: str, state: dict, status: str, db_path: str = DB_PATH) -> None:
    """Persist a completed run to the runs table."""
    completed       = state.get("completed_features", [])
    completed_plans = state.get("completed_plans", [])
    plan            = state.get("plan")

    all_descriptions = " | ".join(p["description"] for p in completed_plans) \
        if completed_plans else (plan.description if plan else "")
    all_codes = " | ".join(p["pandas_code"] for p in completed_plans) \
        if completed_plans else (plan.pandas_code if plan else "")

    with sqlite3.connect(db_path) as con:
        con.execute("""
            INSERT OR REPLACE INTO runs
              (thread_id, timestamp, input_file, hint,
               feature_name, description, pandas_code, attempts, status, features_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            thread_id,
            datetime.now().isoformat(timespec="seconds"),
            os.path.basename(state.get("input_path", "")),
            state.get("objective", ""),
            ", ".join(completed) if completed else (plan.feature_name if plan else ""),
            all_descriptions,
            all_codes,
            state.get("attempts", 0),
            status,
            json.dumps(completed_plans),
        ))


# ── Feature memory functions ────────────────────────────────────────────────────

def save_feature_memory(feature_name: str, formula: str, objective: str, db_path: str = DB_PATH) -> bool:
    """Save a validated feature to memory if formula not already present.

    Returns True if saved, False if formula already exists.
    """
    with sqlite3.connect(db_path) as con:
        exists = con.execute(
            "SELECT 1 FROM feature_memory WHERE formula = ?", (formula,)
        ).fetchone()
        if exists:
            return False
        con.execute(
            "INSERT INTO feature_memory (feature_name, formula, objective) VALUES (?, ?, ?)",
            (feature_name, formula, objective[:500])
        )
        return True


def get_similar_domain_features(current_objective: str, threshold: float = 0.3, db_path: str = DB_PATH) -> list[dict]:
    """Return features from past runs with similar objectives using TF-IDF cosine similarity.

    Returns list of dicts: {feature_name, formula, similarity}
    """
    with sqlite3.connect(db_path) as con:
        rows = con.execute(
            "SELECT feature_name, formula, objective FROM feature_memory"
        ).fetchall()

    if not rows:
        return []

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        past_objectives = [r[2] for r in rows]
        corpus = [current_objective] + past_objectives

        vectorizer = TfidfVectorizer(stop_words="english")
        tfidf      = vectorizer.fit_transform(corpus)
        scores     = cosine_similarity(tfidf[0:1], tfidf[1:]).flatten()

        results = []
        for i, score in enumerate(scores):
            if score >= threshold:
                results.append({
                    "feature_name": rows[i][0],
                    "formula":      rows[i][1],
                    "similarity":   round(float(score), 3),
                })
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results

    except ImportError:
        print("[feature_memory] sklearn not available — skipping TF-IDF match")
        return []


def load_history(db_path: str = DB_PATH) -> pd.DataFrame:
    """Return all past runs as a DataFrame."""
    init_db(db_path)
    with sqlite3.connect(db_path) as con:
        try:
            return pd.read_sql("SELECT * FROM runs ORDER BY timestamp DESC", con)
        except Exception:
            return pd.DataFrame()


def load_failed_runs(db_path: str = DB_PATH) -> list[str]:
    """Return failed thread_ids formatted for a dropdown."""
    init_db(db_path)
    with sqlite3.connect(db_path) as con:
        try:
            df = pd.read_sql(
                "SELECT thread_id, timestamp, input_file FROM runs "
                "WHERE status='failed' ORDER BY timestamp DESC",
                con,
            )
            if df.empty:
                return []
            return [
                f"{row['timestamp']}  |  {row['input_file']}  |  {row['thread_id']}"
                for _, row in df.iterrows()
            ]
        except Exception:
            return []


def load_features_breakdown(thread_id: str, db_path: str = DB_PATH) -> pd.DataFrame:
    """Return per-feature breakdown for a given thread_id."""
    init_db(db_path)
    with sqlite3.connect(db_path) as con:
        try:
            row = con.execute(
                "SELECT features_json FROM runs WHERE thread_id = ?", (thread_id,)
            ).fetchone()
            if not row or not row[0]:
                return pd.DataFrame()
            return pd.DataFrame(json.loads(row[0]))
        except Exception:
            return pd.DataFrame()
