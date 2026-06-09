"""Test embedder loading without meta tensor errors."""

import pytest
import torch


@pytest.mark.skip(
    reason="Jina model cache incompatible with installed transformers version; USE_LOCAL_LATE_CHUNKING=0 by default"
)
def test_embedder_loads_without_meta_tensor():
    """Smoke test: model loads + first embed_query call succeeds."""
    from src.trainer.ingestion.embedder import LocalLateChunkingEmbedder

    embedder = LocalLateChunkingEmbedder(
        model_name="jinaai/jina-embeddings-v3",
        max_context_tokens=8192,
    )
    vec = embedder.embed_query("test sorgu")
    assert isinstance(vec, list)
    assert len(vec) == 1024  # Jina v3 dim
    assert all(isinstance(x, float) for x in vec)
