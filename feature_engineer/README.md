# Feature Engineering Agent

An agentic tool that automates feature engineering for machine learning projects. Give it a CSV file and describe what you want to predict — it researches, plans, generates and validates new features automatically.

## What it does

The agent follows a multi-step pipeline:

1. **Research** — searches the web (Serper) and academic papers (arXiv MCP HTTP server) for feature engineering best practices relevant to your ML objective
2. **Extract** — synthesises concrete feature candidates from both sources, with arXiv features injected directly as structured candidates with pandas formulas
3. **Evaluate** — filters out vague, generic, off-domain or target-leaking candidates
4. **Map** — checks each candidate against your actual columns using semantic matching, excluding target columns
5. **Rank** — orders features by predicted predictive value
6. **Plan** — generates pandas expressions for each feature; injects formula hints from research and from feature memory (past runs with similar objectives via TF-IDF)
7. **Validate** — checks for data leakage, statistical quality and hint compliance
8. **Execute** — runs the pandas expressions, auto-fixes common patterns (e.g. `nunique().transform()`)
9. **Self-correct** — retries up to 3 times with a growing blacklist of failed formulas
10. **Remember** — saves validated features to SQLite memory; future runs with similar objectives reuse them via TF-IDF cosine similarity
11. **Recommend** — generates top-5 features that would require additional data, informed by both Serper and arXiv context

## Research sources

### Serper (web search)
Searches for notebooks, papers and tutorials relevant to your domain. Results are used as context — the LLM in `extract_candidates` proposes features with formulas based on what it finds.

### arXiv MCP (academic papers)
A custom MCP server (`mcp_servers/arxiv_server.py`) that runs as a separate HTTP process:
- Started automatically by `main.py` at startup on `http://127.0.0.1:8000`
- Generates a domain-specific arXiv query from your objective
- Downloads and reads up to 5 PDF papers with retry + rate-limit handling
- Extracts concrete feature candidates with pandas formulas using schema-aware comparison
- Deduplicates features across papers
- Marks features requiring external data (`[external data required]`)

arXiv features are injected directly into the `extract_candidates` prompt with `[SOURCE: arxiv]` tags, treated equally to Serper-derived features.

Uses **streamable HTTP transport** (MCP spec recommended for web server context). `langchain-mcp-adapters` `MultiServerMCPClient` connects to the server from the async LangGraph node.

## Feature memory

The agent maintains a persistent feature memory in `memory.db` (`feature_memory` table):

- **Save** — after each validated feature, `record_feature` saves `(feature_name, formula, objective)` if the formula is not already present
- **Reuse** — at planning time, `feature_planner` queries memory for features from past runs with similar objectives using **TF-IDF cosine similarity** (threshold 0.3)
- **Deduplication** — exact formula match prevents duplicate saves across runs; formula assignment syntax (`df['col'] = expr`) is stripped before saving
- Features with only 2 unique values are not saved to memory (trivial binary encodings)

## Quick start

```bash
# clone and install
git clone <repo>
cd feature_engineer
uv sync   # or pip install -e .

# add your keys to .env
cp .env.example .env
# edit .env and add OPENAI_API_KEY and SERPER_API_KEY

# launch the UI (arXiv MCP server starts automatically)
python -m feature_engineer.main --ui

# or use CLI
python -m feature_engineer.main \
  --input data.csv \
  --output output/enriched.csv \
  --objective "predict customer churn using time and location features" \
  --max-features 10
```

## Project structure

```
feature_engineer/
├── config.py               # constants, paths, reads keys from .env
├── state.py                # AgentState and Pydantic models
├── main.py                 # CLI entry point; starts arXiv MCP server subprocess
├── mcp_servers/
│   └── arxiv_server.py     # FastMCP HTTP server (search, download, read PDFs)
├── security/               # AST validator for generated code
├── storage/
│   ├── arxiv_papers/       # downloaded PDF cache
│   ├── database.py         # SQLite: runs history + feature memory (TF-IDF reuse)
│   └── parquet.py          # parquet caching helpers
├── llm/                    # LLM setup, Serper tool, naming
├── nodes/
│   ├── research.py         # Serper ReAct loop + arXiv MCP HTTP integration
│   ├── planning.py         # feature_planner (+ memory injection) + validate_plan
│   ├── execution.py        # validate_code + create_feature + validate + record_feature
│   ├── revision.py         # revise_plan with formula blacklist
│   ├── routing.py          # save_csv + generate_recommendations + routing functions
│   ├── ranking.py          # rank_features
│   ├── schema_analyzer.py  # column profiling + semantic type inference + target detection
│   └── ingestion.py        # load_csv + parquet caching
├── graph/                  # LangGraph graph assembly
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
- `map_to_columns` prompt: excludes features whose `needs` list contains target columns
- `validate_plan` deterministic override: blocks any formula containing a target column name
- `create_feature` hard block: deterministic check before eval

**2. AST validator** — deterministic checks before any code runs:
- Blocks `open()`, `exec()`, dunder access
- Blocks `shift(-N)` (negative shift = future leakage)
- Blocks `.unstack().stack()` (MultiIndex incompatible)
- Blocks `groupby().apply(lambda).transform()` pattern
- Blocks formulas already in the session failed blacklist

**3. Semantic validator** (`validate_plan`) — LLM chain-of-thought review:
- Temporal leakage detection (with positive-shift override — shift(N>0) is never leakage)
- High cardinality string concat detection (with deterministic cardinality override)
- Math correctness check (description vs formula)
- Name collision detection

**4. Statistical validator** (`validate`) — post-execution checks:
- Null rate > 50%
- Zero variance (constant column)
- Low variance from `nunique` aggregations
- Timedelta dtype (needs `.dt.days` conversion)
- High cardinality warning for object columns

**5. Auto-fix** — deterministic corrections in `create_feature`:
- `groupby().nunique()` → `groupby().transform('nunique')`
- `groupby([list])['col'].size()` → `groupby([list])['col'].transform('count')`
- `str.contains()` without `na=False` → adds `na=False`
- String concat with numeric columns → adds `.astype(str)`

**6. Eval namespace** — restricted Python builtins:
- Only `pd`, `np`, `int`, `float`, `str`, `bool`, `abs`, `len`, `isinstance`, `set`, `list`, `dict`, `tuple` available
- `__builtins__` set to `{}` — no shell access, no imports

## Agentic patterns used

- **ReAct** — Serper research loop (reason → search → observe → reason)
- **Plan and Execute** — `feature_planner` plans all features upfront, executed one by one
- **Reflection / Self-Critique** — `validate_plan` and `validate` review the agent's own output
- **Self-Correction with Memory** — `revise_plan` retries with growing blacklist
- **Tool Use** — Serper via `bind_tools()`, arXiv via MCP HTTP + `MultiServerMCPClient`
- **Structured Output** — all LLM calls use Pydantic models (no manual JSON parsing)
- **Episodic Memory** — TF-IDF feature memory across runs (`feature_memory` table)
- **State Machine** — LangGraph stateful graph with conditional routing

## Tested datasets

| Dataset | Domain | Features generated |
|---|---|---|
| GDSC drug sensitivity (242k×19) | Cancer genomics | Genomic interactions, tissue encodings, cohort counts |
| Global climate energy (36k×10) | Energy forecasting | Temporal, rolling, per-country aggregations |
| MultiJet CMS (21k×17) | Particle physics | Invariant mass, momentum ratios, razor variables |

## Requirements

Python 3.12+. Main dependencies: `langgraph`, `langchain-openai`, `langchain-mcp-adapters==0.1.7`, `gradio`, `pandas`, `pyarrow`, `pymupdf`, `mcp`, `scikit-learn`, `aiosqlite`.