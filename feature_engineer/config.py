"""
Central configuration — all constants and env vars in one place.
Secrets live in .env and are never hardcoded here.
"""
import os

from dotenv import load_dotenv

load_dotenv()

# ── Paths ───────────────────────────────────────────────────────────────────────
DB_PATH      = "memory.db"
PARQUET_DIR  = "df_store"
OUTPUT_DIR   = "output"

# ── API keys (read from .env) ───────────────────────────────────────────────────
SERPER_KEY   = os.getenv("SERPER_API_KEY", "")
# OPENAI_API_KEY is read automatically by LangChain from the environment

# ── Agent defaults ──────────────────────────────────────────────────────────────
DEFAULT_MAX_FEATURES  = 3
DEFAULT_RECURSION_LIMIT = 100
LLM_MODEL             = "gpt-4o-mini"
