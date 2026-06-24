"""LangChain glue for the examples: a local embedder and an optional chat model.

`MiniLMEmbeddings` exposes the same local `all-MiniLM-L6-v2` model the rest of
the suite uses (384-dim, cosine) through LangChain's `Embeddings` interface, so
the examples stay key-free. `chat_model()` returns a configured LangChain chat
model (Azure AI Foundry or OpenAI, same env convention as `_shared/llm.py`) or
`None` when no key is set — callers degrade to printing retrieved context.
"""

import os

from langchain_core.embeddings import Embeddings

from _shared.embedding import embed, embed_query
from _shared.llm import _load_env_files  # reuse the suite's .env loader


class MiniLMEmbeddings(Embeddings):
    """LangChain `Embeddings` over the suite's local all-MiniLM-L6-v2 (dim 384)."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return embed(list(texts))

    def embed_query(self, text: str) -> list[float]:
        return embed_query(text)


def chat_model(temperature: float = 0.0):
    """A LangChain chat model from the environment, or `None` if no key is set.

    Prefers Azure AI Foundry (`AZURE_AI_ENDPOINT`, `AZURE_AI_API_KEY`,
    `DEFAULT_AZURE_MODEL`) over OpenAI (`OPENAI_API_KEY`, optional `OPENAI_MODEL`),
    matching `_shared/llm.py`. Imported lazily so the embeddings path needs no
    `langchain-openai` install.
    """
    _load_env_files()
    from langchain_openai import ChatOpenAI

    azure_endpoint = os.environ.get("AZURE_AI_ENDPOINT")
    azure_key = os.environ.get("AZURE_AI_API_KEY")
    if azure_endpoint and azure_key:
        model = os.environ.get("DEFAULT_AZURE_MODEL", "gpt-4o-mini")
        return ChatOpenAI(
            base_url=azure_endpoint, api_key=azure_key, model=model,
            temperature=temperature,
        )

    if os.environ.get("OPENAI_API_KEY"):
        model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        return ChatOpenAI(model=model, temperature=temperature)

    return None
