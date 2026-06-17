# Copilot Instructions

## Repository Overview

CS4603 course repository for learning LLM basics and LangChain. Notebooks connect to **Databricks model-serving endpoints** via an OpenAI-compatible API. MLflow is used for tracing and experiment tracking.

## Environment Setup

Uses **uv** for environment management (not pip/venv directly):

```bash
# Create and activate venv
uv venv -n .venv-cs4603
.venv-cs4603\Scripts\activate      # Windows
source .venv-cs4603/bin/activate   # macOS/Linux

# Install dependencies
uv pip install -r requirements.txt

# Start MLflow UI (required for tracing notebooks)
mlflow ui --port 5000
```

## Required `.env` File

A `.env` file must exist at the repo root (it is gitignored):

```
DATABRICKS_TOKEN=""
DATABRICKS_HOST="https://<your-workspace-id>.databricks.com"
DATABRICKS_MODEL="databricks-qwen35-122b-a10b"

# Required for wk2-databricks notebooks only:
DATABRICKS_VS_ENDPOINT="<your-vector-search-endpoint-name>"
DATABRICKS_VS_INDEX="cs4603.rag.docs_index"
```

Available models can be listed by running `wk1 - llm basics/0.endpoints.ipynb` or from **AI/ML → Models** in the Databricks workspace UI.

## Architecture

### Shared Bootstrap Pattern

Each week folder has a `*_common_notebook.py` helper that every notebook imports. Call `bootstrap_notebook()` at the top of a notebook to get configured clients:

**`shared_notebook_setup.py`** (wk1) — returns raw `openai.OpenAI` client:
```python
from shared_notebook_setup import bootstrap_notebook
DATABRICKS_TOKEN, DATABRICKS_HOST, DATABRICKS_MODEL, client = bootstrap_notebook()
```

**`wk2 - langchain/langchain_common_notebook.py`** (wk2) — returns LangChain `ChatOpenAI` + embeddings:
```python
from langchain_common_notebook import bootstrap_notebook
DATABRICKS_TOKEN, DATABRICKS_HOST, DATABRICKS_MODEL, (llm, llm_noreason), databricks_embeddings = bootstrap_notebook()
```

**`wk2-databricks/databricks_native_common.py`** (wk2-databricks) — returns native `ChatDatabricks` + `DatabricksEmbeddings` + vector search config:
```python
from wk2_databricks.databricks_native_common import bootstrap_notebook
DATABRICKS_TOKEN, DATABRICKS_HOST, DATABRICKS_MODEL, (llm, llm_noreason), embeddings, vs_config = bootstrap_notebook()
```

### Key Objects

- **`llm`** — `ChatOpenAI` / `ChatDatabricks` with default reasoning
- **`llm_noreason`** — same model with `reasoning_effort="none"` (faster/cheaper)
- **`databricks_embeddings`** / **`embeddings`** — `OpenAIEmbeddings` / `DatabricksEmbeddings` using `databricks-gte-large-en` (1024-dim vectors)
- **`vs_config`** — `VectorSearchConfig(endpoint_name, index_name)` (wk2-databricks only)
- **`DatabricksConfig`** — frozen dataclass holding `token`, `host`, `endpoint`, `vs_endpoint`, `vs_index`
- `MissingEnvironmentVariableError` — raised when any required env var is absent

### Databricks Vector Search (wk2-databricks)

Uses a **Direct Access Index** — you manage chunking/embedding, Databricks manages persistent ANN storage.

- Index must be created once via `VectorSearchClient.create_direct_access_index()`  
- Embedding dimension: `1024` (matches `databricks-gte-large-en`)
- Upsert documents: `vectorstore.add_texts(texts, metadatas, ids)`
- Retrieve: `vectorstore.as_retriever(search_kwargs={"k": 4})`

### Databricks Client Construction

Both `wk1` and `wk2` helpers connect to Databricks by pointing the OpenAI `base_url` at `{DATABRICKS_HOST}/serving-endpoints`. The token is passed as `api_key`.

## Conventions

- Notebooks are numbered by topic order (e.g., `0.endpoints.ipynb`, `1.tokens.ipynb`).
- `enable_logging()` / `disable_logging()` helpers are available in both common modules to toggle debug output from LangChain and vector store internals.
- MLflow artifacts (under `mlartifacts/`) are committed to the repo; `mlruns/` and `mlflow.db` are gitignored.
- Suppress Pydantic v2 deprecation warnings with the pattern already in both common modules.
