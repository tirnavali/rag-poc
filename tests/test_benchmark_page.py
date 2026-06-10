"""Integration tests for page-level evaluation path in RetrievalBenchmark."""
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

class TestBenchmarkPageOverlap:
    def test_page_overlap_path_dispatches_correctly(self, mock_spec):
        """Verify that a query with relevant_pages uses page_overlap path and computes metrics correctly."""
        with patch("src.evaluator.benchmark.VectorSearch") as mock_vs_class:
            mock_vs = MagicMock()
            mock_collection = MagicMock()
            mock_vs.embedder = MagicMock()
            mock_vs.collection = mock_collection
            mock_vs_class.return_value = mock_vs
            bench = RetrievalBenchmark(mock_spec)

        # Mock search to return dummy chunks with metadata
        bench.search = MagicMock(
            return_value=[
                {
                    "id": "doc_0_0",
                    "text": "chunk 0",
                    "score": 0.9,
                    "meta": {"document_id": "doc_0", "pages": [1, 2]}
                },
                {
                    "id": "doc_0_1",
                    "text": "chunk 1",
                    "score": 0.8,
                    "meta": {"document_id": "doc_0", "page": 3}
                },
                {
                    "id": "doc_0_2",
                    "text": "chunk 2",
                    "score": 0.7,
                    "meta": {"document_id": "doc_0", "pages": [4]}
                }
            ]
        )

        # relevant pages are page 2 and page 3 of doc_0
        query = {
            "id": "test-page-001",
            "query": "test page question",
            "relevant_pages": [
                {"document_id": "doc_0", "pages": [2]},
                {"document_id": "doc_0", "page": 3}
            ],
        }

        report = bench.evaluate([query], k_values=(1, 2, 3))

        # Verify result was added
        assert len(report["results"]) == 1
        result = report["results"][0]

        # Check matcher is set to page_overlap
        assert result["matcher"] == "page_overlap"

        # Check metrics exist
        assert "precision_1" in result["metrics"]
        assert "recall_1" in result["metrics"]
        assert "precision_2" in result["metrics"]
        assert "precision_3" in result["metrics"]
        assert "mrr" in result["metrics"]
        assert "ndcg_10" in result["metrics"]

        # Explanation of metrics:
        # retrieved chunks:
        # rank 1: chunk_0_0 (pages [1, 2]) -> covers "doc_0#page_2" -> HIT!
        # rank 2: chunk_0_1 (page 3) -> covers "doc_0#page_3" -> HIT!
        # rank 3: chunk_0_2 (pages [4]) -> no overlap -> MISS.
        #
        # relevant_keys = {"doc_0#page_2", "doc_0#page_3"} (size 2)
        #
        # k=1: hits=1, precision=1.0. covered_keys = {"doc_0#page_2"}, recall = 1/2 = 0.5. hit_rate=1.0
        # k=2: hits=2, precision=1.0. covered_keys = {"doc_0#page_2", "doc_0#page_3"}, recall = 2/2 = 1.0. hit_rate=1.0
        # k=3: hits=2, precision=2/3 = 0.6667. covered_keys = {"doc_0#page_2", "doc_0#page_3"}, recall = 1.0. hit_rate=1.0
        #
        # MRR: first hit at rank 1 -> 1.0
        #
        # NDCG@10:
        # gains = [1, 1, 0]
        # ideal = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0] (since there are 2 relevant keys)
        # actual_dcg = 1/log2(2) + 1/log2(3) = 1 + 0.6309 = 1.6309
        # ideal_dcg = 1/log2(2) + 1/log2(3) = 1.6309
        # NDCG = 1.0
        
        assert result["metrics"]["precision_1"] == 1.0
        assert result["metrics"]["recall_1"] == 0.5
        assert result["metrics"]["hit_rate_1"] == 1.0

        assert result["metrics"]["precision_2"] == 1.0
        assert result["metrics"]["recall_2"] == 1.0
        assert result["metrics"]["hit_rate_2"] == 1.0

        assert abs(result["metrics"]["precision_3"] - 2.0 / 3.0) < 1e-6
        assert result["metrics"]["recall_3"] == 1.0
        assert result["metrics"]["hit_rate_3"] == 1.0

        assert result["metrics"]["mrr"] == 1.0
        assert abs(result["metrics"]["ndcg_10"] - 1.0) < 1e-6
