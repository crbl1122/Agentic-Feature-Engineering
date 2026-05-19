# Feature Engineering Agent

An agentic tool that automates feature engineering for machine learning projects. Give it a CSV file and describe what you want to predict — it researches, plans, generates and validates new features automatically.

## What it does

The agent follows a multi-step pipeline:

1. **Research** — searches the web (Serper) and academic papers (arXiv MCP) for feature engineering best practices relevant to your ML objective
2. **Extract** — synthesises concrete feature candidates from both sources, with arXiv features injected directly as structured candidates with pandas formulas
3. **Evaluate** — filters out vague, generic, off-domain or target-leaking candidates
4. **Map** — checks each candidate against your actual columns using semantic matching, excluding target columns
5. **Rank** — orders features by predicted predictive value
6. **Plan** — generates pandas expressions for each feature using research formula hints where available
7. **Validate** — checks for data leakage, statistical quality and hint compliance
8. **Execute** — runs the pandas expressions, auto-fixes common patterns (e.g. `nunique().transform()`)
9. **Self-correct** — retries up to 3 times with a growing blacklist of failed formulas
10. **Recommend** — generates top-5 features that would require additional data, informed by both Serper and arXiv context

## Research sources

### Serper (web search)
Searches for notebooks, papers and tutorials relevant to your domain. Results are used as context — the LLM in `extract_candidates` proposes features with formulas based on what it finds.

### arXiv MCP (academic papers)
A custom MCP server (`mcp_servers/arxiv_server.py`) that:
- Generates a domain-specific arXiv query from your objective
- Downloads and reads up to 5 PDF papers
- Extracts concrete feature candidates with pandas formulas using schema-aware comparison
- Deduplicates features across papers
- Marks features requiring external data (`[external data required]`)

arXiv features are injected directly into the `extract_candidates` prompt with `[SOURCE: arxiv]` tags, treated equally to Serper-derived features.

## Quick start

```bash
# clone and install
git clone <repo>
cd feature_engineer
uv sync   # or pip install -e .

# add your keys to .env
cp .env.example .env
# edit .env and add OPENAI_API_KEY and SERPER_API_KEY

# install arXiv MCP dependencies
uv add nest_asyncio pymupdf mcp

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
├── config.py               # constants, paths, reads keys from .env
├── state.py                # AgentState and Pydantic models
├── main.py                 # CLI entry point
├── mcp_servers/
│   └── arxiv_server.py     # custom arXiv MCP server (search, download, read PDFs)
├── security/               # AST validator for generated code
├── storage/
│   ├── arxiv_papers/       # downloaded PDF cache
│   ├── database.py         # SQLite run history
│   └── parquet.py          # parquet caching helpers
├── llm/                    # LLM setup, Serper tool, naming
├── nodes/
│   ├── research.py         # Serper ReAct loop + arXiv MCP integration
│   ├── planning.py         # feature_planner + validate_plan
│   ├── execution.py        # validate_code + create_feature + validate
│   ├── revision.py         # revise_plan with formula blacklist
│   ├── routing.py          # save_csv + generate_recommendations
│   ├── ranking.py          # rank_features
│   ├── schema_analyzer.py  # column profiling + target detection
│   └── ingestion.py        # load_csv + parquet caching
├── graph/                  # graph assembly
├── ui/                     # Gradio interface
└── tests/                  # unit tests
```

## Environment variables

```
OPENAI_API_KEY=sk-...
SERPER_API_KEY=...
```

## Running tests

```bash
pytest tests/ -v
```

## Protection layers against bad features

**1. Target leakage** — three checkpoints:
- `extract_candidates` prompt: explicitly excludes target columns
- `map_to_columns` prompt: excludes features whose `needs` list contains target columns (with exception for positive-shift lag features)
- `validate_plan` deterministic override: blocks any formula containing a target column name

**2. AST validator** — deterministic checks before any code runs:
- Blocks `open()`, `exec()`, dunder access
- Blocks `shift(-N)` (negative shift = future leakage)
- Blocks `.unstack().stack()` (MultiIndex incompatible)
- Blocks formulas already in the failed blacklist

**3. Semantic validator** — LLM chain-of-thought review:
- Temporal leakage detection
- High cardinality string concat detection (with deterministic override)
- Math correctness check (description vs formula)
- Name collision detection

**4. Statistical validator** — post-execution checks:
- Null rate > 50%
- Zero variance (constant column)
- Low variance from `nunique` aggregations (unique=2 from groupby nunique)
- High cardinality warning for object columns

**5. Auto-fix** — deterministic corrections in `create_feature`:
- `groupby().nunique().transform()` → `groupby().transform('nunique')`
- `groupby([list])['col'].size().transform()` → `groupby([list])['col'].transform('count')`

## Tested datasets

| Dataset | Domain | Features generated |
|---|---|---|
| GDSC drug sensitivity | Cancer genomics | Binary flags, cohort counts, tissue encodings |
| Global climate energy | Energy forecasting | Temporal, rolling, per-country aggregations |
| MultiJet CMS | Particle physics | Invariant mass, momentum ratios, razor variables |

## Requirements

Python 3.12+. Main dependencies: `langgraph`, `langchain-openai`, `langchain-mcp-adapters`, `gradio`, `pandas`, `pyarrow`, `pymupdf`, `mcp`, `nest_asyncio`, `aiosqlite`.