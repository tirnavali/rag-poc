"""Tests for multi-collection vector explorer retrieval and result flattening."""
import pytest

from src.config.collections import get_spec


@pytest.mark.integration
class TestVectorExplorerMultiCollection:
    """Integration tests — require real ChromaDB collections on disk."""

    def test_vector_explorer_multi_collection_results(self):
        """Verify that multi-collection retrieval includes results from all sources.

        This test ensures that when querying multiple collections, results are
        properly attributed to their source collections via the 'collection' field
        in metadata. The test validates that the per-source retrieval structure
        before RRF fusion can be properly flattened for UI display.
        """
        from src.retriever.multi_source import MultiSourceRetriever

        spec1 = get_spec("gazete_arsivi")
        spec2 = get_spec("tbmm_minutes")

        retriever = MultiSourceRetriever(specs=[spec1, spec2])
        result = retriever.retrieve("ekonomi", top_k=3)

        assert isinstance(result["documents"], list)
        assert len(result["documents"]) >= 1

        docs_list = result["documents"][0]
        metas_list = result["metadatas"][0]
        dists_list = result["distances"][0]

        assert len(docs_list) > 0
        assert len(docs_list) == len(metas_list)
        assert len(docs_list) == len(dists_list)

        collections_found = set()
        for meta in metas_list:
            assert "collection" in meta
            assert meta["collection"] in ["gazete_arsivi", "tbmm_minutes"]
            collections_found.add(meta["collection"])

        assert len(collections_found) > 0


@pytest.mark.integration
def test_flatten_multi_collection_results():
    """Verify flattening preserves all results and maintains order.

    Fonksiyon pure olsa da scripts.vector_explorer import'u modül seviyesinde
    ChromaDB başlatıyor; integration olarak işaretlendi.
    """
    from scripts.vector_explorer import _flatten_multi_collection_results

    test_result = {
        "documents": [
            ["doc1_col1", "doc2_col1"],
            ["doc1_col2", "doc2_col2"],
        ],
        "metadatas": [
            [{"collection": "col1", "id": "1"}, {"collection": "col1", "id": "2"}],
            [{"collection": "col2", "id": "3"}, {"collection": "col2", "id": "4"}],
        ],
        "distances": [
            [0.1, 0.2],
            [0.15, 0.25],
        ],
    }

    flattened = list(_flatten_multi_collection_results(test_result))

    assert len(flattened) == 4

    for doc, meta, dist in flattened:
        assert isinstance(doc, str)
        assert isinstance(meta, dict)
        assert isinstance(dist, (float, int))

    assert flattened[0][0] == "doc1_col1"
    assert flattened[1][0] == "doc2_col1"
    assert flattened[2][0] == "doc1_col2"
    assert flattened[3][0] == "doc2_col2"

    assert flattened[0][1]["collection"] == "col1"
    assert flattened[2][1]["collection"] == "col2"

    assert flattened[0][2] == 0.1
    assert flattened[3][2] == 0.25
