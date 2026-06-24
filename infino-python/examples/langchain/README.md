# LangChain + Infino examples

Build LangChain apps on [Infino](https://pypi.org/project/infino/) through the
[`langchain-infino`](https://pypi.org/project/langchain-infino/) integration —
**one** vector store that does vector, full-text (BM25), hybrid, self-query, and
SQL over a single copy of your data on object storage. No `EnsembleRetriever`
over two backends, no separate metadata database, no second cache to run.

Each example uses a **real public dataset** (HuggingFace Hub) and the same local
`all-MiniLM-L6-v2` embeddings as the rest of the suite, so the retrieval path
runs **locally and key-free**. Generating answers (and the agent example) needs
an LLM key; each notebook says so up front and degrades gracefully without one.

## Examples

| Notebook | What it shows | LLM |
| -------- | ------------- | --- |
| [`01_live_data_rag.ipynb`](01_live_data_rag.ipynb) | A vector store you can safely **write to**: append / upsert / delete, and retrieval *and* SQL reflect every change immediately and survive a reopen from disk. | optional |
| [`02_research_assistant.ipynb`](02_research_assistant.ipynb) | Natural-language filters compile to a SQL `WHERE` push-down (self-query), hybrid RRF retrieval beats vector-only and BM25-only — **measured** with recall@10 — plus conversational follow-ups with citations. | partial |
| [`03_support_ops_agent.ipynb`](03_support_ops_agent.ipynb) | A LangGraph agent whose tools are all backed by **one** Infino store — semantic docs, exact error-code lookup (BM25), order/inventory SQL, and product-scoped filtered retrieval — with a semantic LLM cache and multi-turn memory. | required |

## Setup

```sh
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

The first run downloads the embedding model (~90 MB) and a dataset sample, so
the first cell can take a minute; later runs use the cache.

### Optional: LLM answers

The notebooks pick up a key automatically (via `_shared/llm.py`'s env handling),
reading a local `.azure.env` / `.env` if present:

- **Azure AI Foundry** (preferred): `AZURE_AI_ENDPOINT`, `AZURE_AI_API_KEY`, `DEFAULT_AZURE_MODEL`.
- **OpenAI**: `OPENAI_API_KEY` (optionally `OPENAI_MODEL`).

Keep credentials in an untracked env file — never commit keys.

## How this maps to LangChain

These examples are idiomatic LangChain — `InfinoVectorStore` is a standard
`VectorStore`, `as_retriever()` returns a normal `BaseRetriever`, and the chains
are plain LCEL / LangGraph. The shared glue lives one level up in
[`../_shared/`](../_shared/): `lc.py` wraps the local embedder as a LangChain
`Embeddings` and exposes an optional `chat_model()`; `loaders.py`, `embedding.py`,
and `llm.py` are reused unchanged from the base suite.
