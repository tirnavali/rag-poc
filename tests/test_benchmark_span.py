"""Integration test: span_overlap path in RetrievalBenchmark."""

import pytest
from unittest.mock import MagicMock, patch

from src.config.document_types import DocumentType
from src.evaluator.benchmark import RetrievalBenchmark


@pytest.fixture
def mock_spec():
    spec = MagicMock()
    spec.name = "test_collection"
    spec.embed_model = "test-embed"
    spec.doc_type = DocumentType.CUSTOM
    spec.min_chunk_chars = 500
    spec.max_chunk_chars = 1000
    return spec


class TestBenchmarkSpanOverlap:
    def test_span_overlap_path_dispatches_correctly(self, mock_spec):
        """Verify that a query with relevant_spans uses span_overlap path."""
        with patch("src.evaluator.benchmark.VectorSearch") as mock_vs_class:
            mock_vs = MagicMock()
            mock_collection = MagicMock()
            mock_vs.embedder = MagicMock()
            mock_vs.collection = mock_collection
            mock_vs_class.return_value = mock_vs
            bench = RetrievalBenchmark(mock_spec)

        # Mock search to return dummy chunk_ids
        bench.search = MagicMock(
            return_value=[
                {"id": "doc_0_0", "text": "chunk 0", "score": 0.9},
                {"id": "doc_0_1", "text": "chunk 1", "score": 0.8},
            ]
        )

        # Mock chunk_id_to_span to resolve to character spans
        def mock_resolve(chunk_id, spec):
            mapping = {
                "doc_0_0": {"document_id": "doc_0", "char_start": 0, "char_end": 100},
                "doc_0_1": {"document_id": "doc_0", "char_start": 100, "char_end": 200},
            }
            return mapping.get(chunk_id)

        query = {
            "id": "test-001",
            "query": "test question",
            "relevant_spans": [
                {"document_id": "doc_0", "char_start": 50, "char_end": 150},
            ],
        }

        with patch("src.evaluator.benchmark.chunk_id_to_span", side_effect=mock_resolve):
            report = bench.evaluate([query], k_values=[1, 2])

        # Verify result was added
        assert len(report["results"]) == 1
        result = report["results"][0]

        # Check matcher is set to span_overlap
        assert result["matcher"] == "span_overlap"

        # Check metrics exist
        assert "precision_1" in result["metrics"]
        assert "recall_1" in result["metrics"]
        assert "mrr" in result["metrics"]

        # With span overlap:
        # - doc_0_0 [0-100] vs golden [50-150]: IoU = 50/150 > 0 → hit
        # - doc_0_1 [100-200] vs golden [50-150]: IoU = 50/150 > 0 → hit
        # Both hit, so precision@1 = 1.0, precision@2 = 1.0, recall@2 = 1.0
        assert result["metrics"]["precision_1"] == 1.0
        assert result["metrics"]["precision_2"] == 1.0
        assert result["metrics"]["recall_2"] == 1.0
        assert result["metrics"]["mrr"] == 1.0

    def test_legacy_chunk_id_path_still_works(self, mock_spec):
        """Verify backward compatibility: queries with relevant_chunk_ids use old path."""
        with patch("src.evaluator.benchmark.VectorSearch") as mock_vs_class:
            mock_vs = MagicMock()
            mock_collection = MagicMock()
            mock_vs.embedder = MagicMock()
            mock_vs.collection = mock_collection
            mock_vs_class.return_value = mock_vs
            bench = RetrievalBenchmark(mock_spec)

        bench.search = MagicMock(
            return_value=[
                {"id": "chunk_001", "text": "chunk", "score": 0.9},
                {"id": "chunk_002", "text": "chunk", "score": 0.8},
            ]
        )

        query = {
            "id": "test-legacy",
            "query": "legacy question",
            "relevant_chunk_ids": ["chunk_001", "chunk_003"],
        }

        report = bench.evaluate([query], k_values=[1, 2])
        result = report["results"][0]

        # Matcher should be "chunk_id" for legacy path
        assert result["matcher"] == "chunk_id"

        # Precision: 1/2 (only chunk_001 matches)
        assert result["metrics"]["precision_2"] == 0.5

    def test_span_path_takes_precedence(self, mock_spec):
        """If both relevant_spans and relevant_chunk_ids exist, spans win."""
        with patch("src.evaluator.benchmark.VectorSearch") as mock_vs_class:
            mock_vs = MagicMock()
            mock_collection = MagicMock()
            mock_vs.embedder = MagicMock()
            mock_vs.collection = mock_collection
            mock_vs_class.return_value = mock_vs
            bench = RetrievalBenchmark(mock_spec)

        bench.search = MagicMock(
            return_value=[
                {"id": "doc_0_0", "text": "chunk", "score": 0.9},
            ]
        )

        def mock_resolve(chunk_id, spec):
            return {"document_id": "doc_0", "char_start": 0, "char_end": 100}

        query = {
            "id": "test-mixed",
            "query": "mixed",
            "relevant_spans": [{"document_id": "doc_0", "char_start": 50, "char_end": 150}],
            "relevant_chunk_ids": ["doc_0_99"],  # Different ground truth
        }

        with patch("src.evaluator.benchmark.chunk_id_to_span", side_effect=mock_resolve):
            report = bench.evaluate([query], k_values=[1])

        result = report["results"][0]
        # Should use span_overlap matcher, not chunk_id
        assert result["matcher"] == "span_overlap"

    def test_gold_evidence_spans_key_recognized(self, mock_spec):
        """Verify that the new gold_evidence_spans key is recognized and produces same results as relevant_spans."""
        with patch("src.evaluator.benchmark.VectorSearch") as mock_vs_class:
            mock_vs = MagicMock()
            mock_collection = MagicMock()
            mock_vs.embedder = MagicMock()
            mock_vs.collection = mock_collection
            mock_vs_class.return_value = mock_vs
            bench = RetrievalBenchmark(mock_spec)

        bench.search = MagicMock(
            return_value=[
                {"id": "doc_0_0", "text": "chunk 0", "score": 0.9},
                {"id": "doc_0_1", "text": "chunk 1", "score": 0.8},
            ]
        )

        def mock_resolve(chunk_id, spec):
            mapping = {
                "doc_0_0": {"document_id": "doc_0", "char_start": 0, "char_end": 100},
                "doc_0_1": {"document_id": "doc_0", "char_start": 100, "char_end": 200},
            }
            return mapping.get(chunk_id)

        query = {
            "id": "test-new-schema",
            "query": "test question",
            "gold_evidence_spans": [
                {"document_id": "doc_0", "char_start": 50, "char_end": 150},
            ],
            "intent": "factual",
            "abstain_required": False,
        }

        with patch("src.evaluator.benchmark.chunk_id_to_span", side_effect=mock_resolve):
            report = bench.evaluate([query], k_values=[1, 2])

        # Verify result was added
        assert len(report["results"]) == 1
        result = report["results"][0]

        # Check matcher is span_overlap
        assert result["matcher"] == "span_overlap"

        # Check metrics exist
        assert "precision_1" in result["metrics"]
        assert "recall_1" in result["metrics"]
        assert "evidence_coverage_1" in result["metrics"]
        assert "evidence_coverage_2" in result["metrics"]

        # Both chunks hit the golden span [50-150]
        assert result["metrics"]["precision_1"] == 1.0
        assert result["metrics"]["precision_2"] == 1.0

        # Evidence coverage: doc_0_0 covers [50-100] (50/100), doc_0_1 covers [100-150] (50/100)
        # Together they cover 100/100 of the golden span
        assert result["metrics"]["evidence_coverage_1"] == 0.5  # Only top-1 chunk_0_0
        assert result["metrics"]["evidence_coverage_2"] == 1.0  # Both chunks
