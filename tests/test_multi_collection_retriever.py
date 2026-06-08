"""Tests for src.retriever.multi_source module with collection specs and attribution."""
import pytest
from unittest.mock import MagicMock, patch

from src.common.protocols import RetrievalResult
from src.config.collections import CollectionSpec
from src.config.document_types import DocumentType
from src.retriever.multi_source import MultiSourceRetriever, _rrf_fuse


# Mock CollectionSpec instances
def create_mock_spec(name: str, doc_type: DocumentType = DocumentType.CUSTOM) -> CollectionSpec:
    """Create a mock CollectionSpec for testing."""
    spec = MagicMock(spec=CollectionSpec)
    spec.name = name
    spec.doc_type = doc_type
    return spec


class TestMultiSourceRetrieverListSpecs:
    """Tests for MultiSourceRetriever accepting list[CollectionSpec]."""

    def test_multi_source_retriever_accepts_dict_legacy_mode(self):
        """Should accept dict[DocumentType, CollectionSpec] (legacy mode)."""
        specs = {
            DocumentType.GAZETE: create_mock_spec("gazete_arsivi", DocumentType.GAZETE),
            DocumentType.TUTANAK: create_mock_spec("tbmm_minutes", DocumentType.TUTANAK),
        }

        with patch("src.retriever.multi_source.VectorRetriever"):
            retriever = MultiSourceRetriever(specs)

            # Should have 2 retrievers, keyed by DocumentType
            assert len(retriever.retrievers) == 2
            assert DocumentType.GAZETE in retriever.retrievers
            assert DocumentType.TUTANAK in retriever.retrievers

    def test_multi_source_retriever_accepts_list_of_specs(self):
        """Should accept list[CollectionSpec] (new multi-collection mode)."""
        specs = [
            create_mock_spec("gazete_arsivi", DocumentType.GAZETE),
            create_mock_spec("tbmm_minutes", DocumentType.TUTANAK),
            create_mock_spec("custom_collection", DocumentType.CUSTOM),
        ]

        with patch("src.retriever.multi_source.VectorRetriever"):
            retriever = MultiSourceRetriever(specs)

            # Should have 3 retrievers, keyed by collection name
            assert len(retriever.retrievers) == 3
            assert "gazete_arsivi" in retriever.retrievers
            assert "tbmm_minutes" in retriever.retrievers
            assert "custom_collection" in retriever.retrievers

    def test_multi_source_retriever_defaults_to_legacy_mode(self):
        """Should default to legacy mode (dict) when specs=None."""
        with patch("src.config.collections.get_default_spec") as mock_default:
            mock_default.side_effect = lambda dt: create_mock_spec(f"default_{dt.value}", dt)
            with patch("src.retriever.multi_source.VectorRetriever"):
                retriever = MultiSourceRetriever(specs=None)

                # Should have retrievers for all default doc types
                assert len(retriever.retrievers) > 0
                # Keys should be DocumentType enums (legacy mode)
                assert any(isinstance(k, DocumentType) for k in retriever.retrievers.keys())


class TestRRFFuseCollectionAttribution:
    """Tests for _rrf_fuse adding collection metadata."""

    def test_rrf_fuse_adds_collection_to_metadata(self):
        """Should add 'collection' field to result metadata from source name."""
        per_source = {
            "gazete_arsivi": {
                "documents": [[
                    "Document 1 from gazete",
                    "Document 2 from gazete",
                ]],
                "metadatas": [[
                    {"document_id": "doc1", "chunk_index": 0},
                    {"document_id": "doc2", "chunk_index": 0},
                ]],
                "distances": [[0.1, 0.2]],
            },
            "tbmm_minutes": {
                "documents": [["Document 3 from tutanak"]],
                "metadatas": [[
                    {"document_id": "doc3", "chunk_index": 0},
                ]],
                "distances": [[0.15]],
            },
        }

        result = _rrf_fuse(per_source, "test query", top_k=10)

        # All metadata should have collection field
        for meta in result["metadatas"][0]:
            assert "collection" in meta
            assert meta["collection"] in ["gazete_arsivi", "tbmm_minutes"]

    def test_rrf_fuse_collection_attribution_in_order(self):
        """Collection field should match the source it came from."""
        per_source = {
            "gazete_arsivi": {
                "documents": [["Gazete doc"]],
                "metadatas": [[{"document_id": "g1", "chunk_index": 0}]],
                "distances": [[0.1]],
            },
        }

        result = _rrf_fuse(per_source, "test query", top_k=10)

        assert len(result["metadatas"][0]) == 1
        assert result["metadatas"][0][0]["collection"] == "gazete_arsivi"

    def test_rrf_fuse_deduplicates_across_collections(self):
        """Should deduplicate docs with same document_id+chunk_index across collections."""
        per_source = {
            "gazete_arsivi": {
                "documents": [["Same doc from gazete"]],
                "metadatas": [[{"document_id": "shared_doc", "chunk_index": 0}]],
                "distances": [[0.1]],
            },
            "tbmm_minutes": {
                "documents": [["Same doc from tutanak"]],
                "metadatas": [[{"document_id": "shared_doc", "chunk_index": 0}]],
                "distances": [[0.2]],
            },
        }

        result = _rrf_fuse(per_source, "test query", top_k=10)

        # Should only have 1 result (deduplicated by document_id+chunk_index)
        assert len(result["documents"][0]) == 1
        assert len(result["metadatas"][0]) == 1
        assert len(result["distances"][0]) == 1

        # The result should contain one of the documents (whichever was processed last)
        # and both should have the collection field
        meta = result["metadatas"][0][0]
        assert "collection" in meta
        assert meta["collection"] in ["gazete_arsivi", "tbmm_minutes"]

        # Distance semantics: when deduplicated, the distance from the last-processed
        # source is retained (since records dict overwrites per uid in iteration order)
        distance = result["distances"][0][0]
        assert distance in [0.1, 0.2], f"Expected distance from one of the sources, got {distance}"

    def test_rrf_fuse_respects_top_k(self):
        """Should return at most top_k results after fusion."""
        per_source = {
            "gazete_arsivi": {
                "documents": [[f"Doc {i}" for i in range(15)]],
                "metadatas": [[
                    {"document_id": f"doc_g{i}", "chunk_index": 0}
                    for i in range(15)
                ]],
                "distances": [[0.1 + i * 0.01 for i in range(15)]],
            },
            "tbmm_minutes": {
                "documents": [[f"Tutanak {i}" for i in range(10)]],
                "metadatas": [[
                    {"document_id": f"doc_t{i}", "chunk_index": 0}
                    for i in range(10)
                ]],
                "distances": [[0.15 + i * 0.01 for i in range(10)]],
            },
        }

        result = _rrf_fuse(per_source, "test query", top_k=5)

        # Should have exactly top_k results (or fewer if less available after dedup)
        assert len(result["documents"][0]) <= 5
        assert len(result["metadatas"][0]) <= 5
        assert len(result["distances"][0]) <= 5

    def test_rrf_fuse_preserves_original_metadata(self):
        """Should preserve all original metadata fields when adding collection."""
        per_source = {
            "test_collection": {
                "documents": [["Test doc"]],
                "metadatas": [[{
                    "document_id": "test_id",
                    "chunk_index": 0,
                    "custom_field": "custom_value",
                    "author": "Test Author",
                }]],
                "distances": [[0.1]],
            },
        }

        result = _rrf_fuse(per_source, "test query", top_k=10)

        meta = result["metadatas"][0][0]
        # Original fields should be preserved
        assert meta["document_id"] == "test_id"
        assert meta["chunk_index"] == 0
        assert meta["custom_field"] == "custom_value"
        assert meta["author"] == "Test Author"
        # Plus the new collection field
        assert meta["collection"] == "test_collection"


class TestMultiSourceRetrieverIntegration:
    """Integration tests for multi-collection retrieval."""

    def test_multi_source_retriever_retrieve_with_list_specs(self):
        """Should successfully retrieve from multiple collections passed as list."""
        specs = [
            create_mock_spec("gazete_arsivi", DocumentType.GAZETE),
            create_mock_spec("tbmm_minutes", DocumentType.TUTANAK),
        ]

        # Mock VectorRetriever.retrieve() to return test results
        mock_retriever_result: RetrievalResult = {
            "documents": [["Test doc"]],
            "metadatas": [[{"document_id": "test", "chunk_index": 0}]],
            "distances": [[0.1]],
            "is_minutes": False,
            "parsed_dates": {},
            "expanded_query": None,
            "fallback_level": None,
        }

        with patch("src.retriever.multi_source.VectorRetriever") as MockVectorRetriever:
            mock_instance = MagicMock()
            mock_instance.retrieve.return_value = mock_retriever_result
            MockVectorRetriever.return_value = mock_instance

            retriever = MultiSourceRetriever(specs)
            result = retriever.retrieve("test query", top_k=10)

            # Should return a valid RetrievalResult with collection metadata
            assert "documents" in result
            assert "metadatas" in result
            assert "distances" in result

            # Check that results have collection field
            for meta in result["metadatas"][0]:
                assert "collection" in meta


class TestRetrieveBalanced:
    """Tests for MultiSourceRetriever.retrieve_balanced()."""

    def test_retrieve_balanced_returns_equal_results_per_collection(self):
        """Should return per_collection_k results from each collection."""
        specs = [
            create_mock_spec("gazete_arsivi", DocumentType.GAZETE),
            create_mock_spec("tbmm_minutes", DocumentType.TUTANAK),
        ]

        def mock_retrieve(query, top_k):
            return {
                "documents": [[f"doc{i}" for i in range(top_k)]],
                "metadatas": [[
                    {"document_id": f"d{i}", "chunk_index": 0}
                    for i in range(top_k)
                ]],
                "distances": [[0.1 + i * 0.01 for i in range(top_k)]],
                "is_minutes": False,
                "parsed_dates": {},
                "expanded_query": None,
                "fallback_level": None,
            }

        with patch("src.retriever.multi_source.VectorRetriever") as MockVR:
            mock_gazete = MagicMock()
            mock_gazete.retrieve.side_effect = mock_retrieve
            mock_gazete.spec.doc_type = DocumentType.GAZETE

            mock_tutanak = MagicMock()
            mock_tutanak.retrieve.side_effect = mock_retrieve
            mock_tutanak.spec.doc_type = DocumentType.TUTANAK

            MockVR.side_effect = [mock_gazete, mock_tutanak]

            retriever = MultiSourceRetriever(specs)
            result = retriever.retrieve_balanced("test query", per_collection_k=3)

            # Should have 2 collections × 3 results = 6 total
            assert len(result["documents"][0]) == 6
            assert len(result["metadatas"][0]) == 6
            assert len(result["distances"][0]) == 6

    def test_retrieve_balanced_adds_collection_and_doc_type_to_metadata(self):
        """Should add collection and doc_type fields to metadata."""
        specs = [
            create_mock_spec("gazete_arsivi", DocumentType.GAZETE),
            create_mock_spec("tbmm_minutes", DocumentType.TUTANAK),
        ]

        mock_result = {
            "documents": [["gazete doc"]],
            "metadatas": [[{"document_id": "g1", "chunk_index": 0}]],
            "distances": [[0.1]],
            "is_minutes": False,
            "parsed_dates": {},
            "expanded_query": None,
            "fallback_level": None,
        }

        with patch("src.retriever.multi_source.VectorRetriever") as MockVR:
            mock_gazete = MagicMock()
            mock_gazete.retrieve.return_value = mock_result
            mock_gazete.spec.doc_type = DocumentType.GAZETE

            mock_tutanak = MagicMock()
            mock_tutanak.retrieve.return_value = mock_result
            mock_tutanak.spec.doc_type = DocumentType.TUTANAK

            MockVR.side_effect = [mock_gazete, mock_tutanak]

            retriever = MultiSourceRetriever(specs)
            result = retriever.retrieve_balanced("test query", per_collection_k=1)

            for meta in result["metadatas"][0]:
                assert "collection" in meta
                assert "doc_type" in meta
                assert meta["doc_type"] in ["gazete", "tutanak"]

    def test_retrieve_balanced_preserves_per_collection_order(self):
        """Results from each collection should maintain their own score order."""
        specs = [
            create_mock_spec("col_a", DocumentType.GAZETE),
            create_mock_spec("col_b", DocumentType.TUTANAK),
        ]

        def mock_retrieve(query, top_k):
            return {
                "documents": [[f"doc_{i}" for i in range(top_k)]],
                "metadatas": [[
                    {"document_id": f"did{i}", "chunk_index": 0}
                    for i in range(top_k)
                ]],
                "distances": [[0.1 + i * 0.1 for i in range(top_k)]],
                "is_minutes": False,
                "parsed_dates": {},
                "expanded_query": None,
                "fallback_level": None,
            }

        with patch("src.retriever.multi_source.VectorRetriever") as MockVR:
            mock_instance = MagicMock()
            mock_instance.retrieve.side_effect = mock_retrieve
            mock_instance.spec.doc_type = DocumentType.GAZETE
            MockVR.return_value = mock_instance

            retriever = MultiSourceRetriever(specs)
            result = retriever.retrieve_balanced("test query", per_collection_k=2)

            # Should have 4 results (2 per collection)
            assert len(result["documents"][0]) == 4
            # First 2 from col_a, next 2 from col_b
            assert result["metadatas"][0][0]["collection"] == "col_a"
            assert result["metadatas"][0][2]["collection"] == "col_b"

    def test_retrieve_balanced_uses_context_weight_when_no_override(self):
        """When per_collection_k is None, should use spec's context_weight."""
        spec_a = create_mock_spec("col_a", DocumentType.GAZETE)
        spec_a.context_weight = 7

        spec_b = create_mock_spec("col_b", DocumentType.TUTANAK)
        spec_b.context_weight = 3

        specs = [spec_a, spec_b]

        call_count = 0
        expected_ks = [7, 3]

        def mock_retrieve(query, top_k):
            nonlocal call_count
            assert top_k == expected_ks[call_count], f"Expected top_k={expected_ks[call_count]}, got {top_k}"
            call_count += 1
            return {
                "documents": [[f"doc_{i}" for i in range(top_k)]],
                "metadatas": [[
                    {"document_id": f"did{i}", "chunk_index": 0}
                    for i in range(top_k)
                ]],
                "distances": [[0.1 + i * 0.01 for i in range(top_k)]],
                "is_minutes": False,
                "parsed_dates": {},
                "expanded_query": None,
                "fallback_level": None,
            }

        def make_mock_instance(idx):
            mock_instance = MagicMock()
            mock_instance.retrieve.side_effect = mock_retrieve
            mock_instance.spec.context_weight = expected_ks[idx]
            mock_instance.spec.doc_type = DocumentType.GAZETE
            return mock_instance

        mock_instances = [make_mock_instance(0), make_mock_instance(1)]

        with patch("src.retriever.multi_source.VectorRetriever") as MockVR:
            MockVR.side_effect = lambda spec: mock_instances.pop(0)

            retriever = MultiSourceRetriever(specs)
            result = retriever.retrieve_balanced("test query")

            # col_a: 7 results, col_b: 3 results = 10 total
            assert len(result["documents"][0]) == 10
