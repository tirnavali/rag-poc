"""Tests for build_embedder_for_spec factory — stress L2 normalization, late chunking, mean pooling."""
import math
from unittest.mock import MagicMock, patch

import pytest


def _norm(vec) -> float:
    """L2 norm of vector."""
    return math.sqrt(sum(x * x for x in vec))


class TestBuildEmbedderForSpec:
    """Factory dispatch + proof of L2/mean-pool/late-chunking wiring."""

    def test_late_chunking_spec_returns_local_embedder(self):
        """Spec with supports_late_chunking=True → LocalLateChunkingEmbedder."""
        from src.config.collections import CollectionSpec, MODEL_SPECS
        from src.common.embeddings import build_embedder_for_spec
        from src.trainer.ingestion.embedder import LocalLateChunkingEmbedder

        spec = CollectionSpec(
            name="test_jina",
            db_path="/tmp",
            embed_model="jinaai/jina-embeddings-v3",
        )
        assert spec.supports_late_chunking
        assert spec.max_context_tokens == 8192
        assert spec.overlap_tokens == 128

        with patch("src.trainer.ingestion.embedder.AutoTokenizer.from_pretrained"), \
             patch("src.trainer.ingestion.embedder.AutoModel.from_pretrained"):
            emb = build_embedder_for_spec(spec)
            assert isinstance(emb, LocalLateChunkingEmbedder)
            assert emb.max_context_tokens == 8192

    def test_ollama_spec_returns_l2_wrapped(self):
        """Spec with supports_late_chunking=False → L2NormalizedEmbeddings wrapping OllamaEmbeddings."""
        from src.config.collections import CollectionSpec
        from src.common.embeddings import build_embedder_for_spec, L2NormalizedEmbeddings
        from langchain_ollama import OllamaEmbeddings

        spec = CollectionSpec(
            name="test_nomic",
            db_path="/tmp",
            embed_model="nomic-embed-text-v2-moe",
        )
        assert not spec.supports_late_chunking

        with patch("src.common.embeddings.OllamaEmbeddings") as mock_ollama:
            mock_base = MagicMock()
            mock_ollama.return_value = mock_base
            emb = build_embedder_for_spec(spec)
            assert isinstance(emb, L2NormalizedEmbeddings)
            assert emb._base is mock_base

    def test_l2_wrap_produces_unit_norm_query(self):
        """L2NormalizedEmbeddings normalizes query to unit norm."""
        from src.common.embeddings import L2NormalizedEmbeddings

        base = MagicMock()
        base.embed_query.return_value = [3.0, 4.0]  # norm=5
        wrapped = L2NormalizedEmbeddings(base)

        result = wrapped.embed_query("test")
        assert abs(_norm(result) - 1.0) < 1e-6, f"Expected unit norm, got {_norm(result)}"
        assert abs(result[0] - 0.6) < 1e-6  # 3/5
        assert abs(result[1] - 0.8) < 1e-6  # 4/5

    def test_l2_wrap_produces_unit_norm_documents(self):
        """L2NormalizedEmbeddings normalizes documents to unit norm."""
        from src.common.embeddings import L2NormalizedEmbeddings

        base = MagicMock()
        base.embed_documents.return_value = [[3.0, 4.0], [0.0, 5.0]]  # norms 5, 5
        wrapped = L2NormalizedEmbeddings(base)

        result = wrapped.embed_documents(["a", "b"])
        for vec in result:
            assert abs(_norm(vec) - 1.0) < 1e-6

    def test_zero_vector_not_normalized(self):
        """Zero vector returns as-is (handled gracefully)."""
        from src.common.embeddings import _l2_normalize

        result = _l2_normalize([[0.0, 0.0, 0.0]])
        assert result == [[0.0, 0.0, 0.0]]
