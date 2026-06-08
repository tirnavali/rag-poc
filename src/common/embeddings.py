from __future__ import annotations

import math
from typing import TYPE_CHECKING

from langchain_ollama import OllamaEmbeddings
from langchain_core.embeddings import Embeddings

from src.config import settings

if TYPE_CHECKING:
    from src.config.collections import CollectionSpec


def _l2_normalize(vecs: list[list[float]]) -> list[list[float]]:
    result = []
    for v in vecs:
        norm = math.sqrt(sum(x * x for x in v))
        result.append([x / norm for x in v] if norm > 0 else v)
    return result


class L2NormalizedEmbeddings(Embeddings):
    """Wraps any Embeddings backend and L2-normalizes output vectors."""

    def __init__(self, base: Embeddings):
        self._base = base

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return _l2_normalize(self._base.embed_documents(texts))

    def embed_query(self, text: str) -> list[float]:
        return _l2_normalize([self._base.embed_query(text)])[0]


def build_embedder(model: str = settings.EMBED_MODEL) -> Embeddings:
    """Factory for embedding backends — used by both Retriever and Trainer.

    When USE_LOCAL_LATE_CHUNKING=1, returns LocalLateChunkingEmbedder (Jina v3,
    loaded locally via HuggingFace). Otherwise returns OllamaEmbeddings.

    Note: LocalLateChunkingEmbedder is a trainer-layer class; it is imported
    lazily here to avoid a common→trainer circular dependency at module load time.
    """
    if settings.USE_LOCAL_LATE_CHUNKING:
        from src.trainer.ingestion.embedder import LocalLateChunkingEmbedder
        from src.config.collections import MODEL_SPECS
        jina_model = settings.JINA_LOCAL_MODEL
        _mspec = MODEL_SPECS.get(jina_model, {})
        return LocalLateChunkingEmbedder(
            model_name=jina_model,
            max_context_tokens=_mspec.get("max_context_tokens", 8192),
            overlap_tokens=_mspec.get("overlap_tokens", 128),
        )
    return L2NormalizedEmbeddings(OllamaEmbeddings(model=model, base_url=settings.OLLAMA_HOST))


def build_embedder_for_spec(spec: "CollectionSpec") -> Embeddings:
    """Per-spec embedder factory. Single source of truth for retriever + benchmark.

    Selects LocalLateChunkingEmbedder for Jina-family models (supports_late_chunking=True)
    and L2-wrapped OllamaEmbeddings for Nomic-family models.
    L2 wrapping ensures unit-norm query vectors for cosine collections.
    """
    if spec.supports_late_chunking:
        from src.trainer.ingestion.embedder import LocalLateChunkingEmbedder
        return LocalLateChunkingEmbedder(
            model_name=spec.embed_model,
            max_context_tokens=spec.max_context_tokens,
            overlap_tokens=spec.overlap_tokens,
        )
    return L2NormalizedEmbeddings(
        OllamaEmbeddings(model=spec.embed_model, base_url=settings.OLLAMA_HOST)
    )


def ensure_ollama_model(model: str = settings.EMBED_MODEL) -> bool:
    """Try to pre-pull an Ollama model. Returns True on success, False otherwise."""
    import ollama
    print(f"--- Ollama model kontrolü: {model} ---")
    try:
        client = ollama.Client(host=settings.OLLAMA_HOST)
        client.pull(model)
        return True
    except Exception as e:
        print(f"HATA: Ollama modeli çekilemedi: {e}")
        return False
