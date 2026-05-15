# LangGraph Feature Engineering Agent

This is an agentic MLOps tool that automates feature engineering for machine learning projects. You give it a CSV file and describe what you want to predict, and it takes care of researching, planning, generating and validating new features.

## What it does

The agent follows a multi-step pipeline:

1. Searches the web (via Serper) for feature engineering best practices related to your ML objective
2. Maps the research findings to what is actually possible with your data columns
3. Plans and generates pandas expressions for each feature
4. Validates each feature for data leakage, statistical quality and hint compliance
5. Self-corrects when a feature fails, keeping a blacklist of tried formulas to avoid repeating mistakes

## Quick start

```bash
# clone and install
git clone <repo>
cd feature_engineer
pip install -e .

# add your keys to .env
cp .env.example .env
# edit .env and add OPENAI_API_KEY and optionally SERPER_API_KEY

# launch the UI
python -m feature_engineer.main --ui

# or use CLI
python -m feature_engineer.main \
  --input data.csv \
  --output output/enriched.csv \
  --objective "predict customer churn using time and location features" \
  --max-features 5
```

## Project structure

```
feature_engineer/
├── config.py          # constants, paths, reads keys from .env
├── state.py           # AgentState and Pydantic models
├── main.py            # CLI entry point
├── security/          # AST validator for generated code
├── storage/           # SQLite run history and parquet helpers
├── llm/               # LLM setup, Serper tool, naming
├── nodes/             # all LangGraph nodes
├── graph/             # graph assembly
├── ui/                # Gradio interface
└── tests/             # unit tests
```

## Environment variables

Create a `.env` file in the project root:

```
OPENAI_API_KEY=sk-...
SERPER_API_KEY=...     # optional, falls back to LLM knowledge if not set
```

## Running tests

```bash
pytest tests/ -v
```

## How the agent protects against bad features

There are three layers of protection before a feature gets saved:

- AST validator: deterministic check that blocks `open()`, `exec()`, dunder access and `shift(-N)` before any code runs
- Semantic validator: LLM reviews each planned feature for temporal leakage and hint compliance using chain-of-thought reasoning
- Statistical validator: checks for null values and zero variance after execution

If a feature fails, the agent retries up to 3 times with a growing blacklist of formulas that did not work.

## Requirements

Python 3.12 or higher. Main dependencies: `langgraph`, `langchain-openai`, `gradio`, `pandas`, `pyarrow`, `aiosqlite`.
