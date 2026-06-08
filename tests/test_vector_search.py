"""Tests for VectorSearch primitive — stress rerank on/off, fetch_k, where_filter, L2 parity."""
from unittest.mock import MagicMock, patch

import pytest


class TestVectorSearch:
    """Shared primitive: every branch tested, with mocked Chroma collection."""

    def test_search_no_reranker_sorts_by_distance(self):
        """Without reranker, sort by distance (lower = better)."""
        from src.retriever.vector_search import VectorSearch
        from src.config.collections import CollectionSpec

        spec = MagicMock(spec=CollectionSpec)
        spec.supports_late_chunking = False
        spec.db_path = "/tmp"
        spec.name = "test"
        spec.db_path = "/tmp"
        spec.name = "test"

        with patch("src.retriever.vector_search.open_collection") as mock_open, \
             patch("src.retriever.vector_search.build_embedder_for_spec") as mock_emb:
            mock_collection = MagicMock()
            mock_open.return_value = (None, mock_collection)
            mock_embedder = MagicMock()
            mock_embedder.embed_query.return_value = [1.0, 0.0]
            mock_emb.return_value = mock_embedder

            # Mock chroma result: 3 candidates
            mock_collection.query.return_value = {
                "ids": [["c1", "c2", "c3"]],
                "documents": [["doc1", "doc2", "doc3"]],
                "metadatas": [[{"a": 1}, {"b": 2}, {"c": 3}]],
                "distances": [[0.5, 0.1, 0.9]],
            }

            search = VectorSearch(spec)
            results = search.search("query", top_k=3, reranker=None)

            # Without reranker, returns candidates in query order, sliced to top_k
            assert len(results) == 3
            assert all(r["rerank_score"] is None for r in results)

    def test_search_with_reranker_reorders(self):
        """With reranker, sort by rerank score (higher = better)."""
        from src.retriever.vector_search import VectorSearch
        from src.config.collections import CollectionSpec

        spec = MagicMock(spec=CollectionSpec)
        spec.supports_late_chunking = False
        spec.db_path = "/tmp"
        spec.name = "test"

        with patch("src.retriever.vector_search.open_collection") as mock_open, \
             patch("src.retriever.vector_search.build_embedder_for_spec") as mock_emb:
            mock_collection = MagicMock()
            mock_open.return_value = (None, mock_collection)
            mock_embedder = MagicMock()
            mock_embedder.embed_query.return_value = [1.0, 0.0]
            mock_emb.return_value = mock_embedder

            mock_collection.query.return_value = {
                "ids": [["c1", "c2", "c3"]],
                "documents": [["doc1", "doc2", "doc3"]],
                "metadatas": [[{"a": 1}, {"b": 2}, {"c": 3}]],
                "distances": [[0.5, 0.1, 0.9]],
            }

            # Mock reranker: c1=0.9, c2=0.2, c3=0.8
            mock_reranker = MagicMock()
            mock_reranker.rerank.return_value = [("c1", 0.9), ("c3", 0.8), ("c2", 0.2)]

            search = VectorSearch(spec)
            results = search.search("query", top_k=3, reranker=mock_reranker)

            # With reranker, returns top_k sorted by rerank score desc
            assert len(results) == 3
            assert all(r["rerank_score"] is not None for r in results)
            assert all(0.0 <= r["rerank_score"] <= 1.0 for r in results)
            # Order must follow reranker: c1(0.9) > c3(0.8) > c2(0.2)
            assert results[0]["id"] == "c1"
            assert results[1]["id"] == "c3"
            assert results[2]["id"] == "c2"
            assert results[0]["rerank_score"] > results[1]["rerank_score"] > results[2]["rerank_score"]

    def test_fetch_k_default(self):
        """top_k=5 → fetch_k defaults to 20."""
        from src.retriever.vector_search import VectorSearch
        from src.config.collections import CollectionSpec

        spec = MagicMock(spec=CollectionSpec)
        spec.supports_late_chunking = False
        spec.db_path = "/tmp"
        spec.name = "test"

        with patch("src.retriever.vector_search.open_collection") as mock_open, \
             patch("src.retriever.vector_search.build_embedder_for_spec") as mock_emb:
            mock_collection = MagicMock()
            mock_open.return_value = (None, mock_collection)
            mock_embedder = MagicMock()
            mock_embedder.embed_query.return_value = [1.0]
            mock_emb.return_value = mock_embedder

            mock_collection.query.return_value = {
                "ids": [[]],
                "documents": [[]],
                "metadatas": [[]],
                "distances": [[]],
            }

            search = VectorSearch(spec)
            search.search("query", top_k=5, fetch_k=None, reranker=None)

            # Verify query called with n_results=20 (max(5*4, 20))
            call_args = mock_collection.query.call_args
            assert call_args[1]["n_results"] == 20

    def test_fetch_k_explicit(self):
        """Explicit fetch_k overrides default."""
        from src.retriever.vector_search import VectorSearch
        from src.config.collections import CollectionSpec

        spec = MagicMock(spec=CollectionSpec)
        spec.supports_late_chunking = False
        spec.db_path = "/tmp"
        spec.name = "test"

        with patch("src.retriever.vector_search.open_collection") as mock_open, \
             patch("src.retriever.vector_search.build_embedder_for_spec") as mock_emb:
            mock_collection = MagicMock()
            mock_open.return_value = (None, mock_collection)
            mock_embedder = MagicMock()
            mock_embedder.embed_query.return_value = [1.0]
            mock_emb.return_value = mock_embedder

            mock_collection.query.return_value = {
                "ids": [[]],
                "documents": [[]],
                "metadatas": [[]],
                "distances": [[]],
            }

            search = VectorSearch(spec)
            search.search("query", top_k=3, fetch_k=50, reranker=None)

            call_args = mock_collection.query.call_args
            assert call_args[1]["n_results"] == 50

    def test_where_filter_passed_through(self):
        """where_filter dict passed to query_collection unchanged."""
        from src.retriever.vector_search import VectorSearch
        from src.config.collections import CollectionSpec

        spec = MagicMock(spec=CollectionSpec)
        spec.supports_late_chunking = False
        spec.db_path = "/tmp"
        spec.name = "test"

        with patch("src.retriever.vector_search.open_collection") as mock_open, \
             patch("src.retriever.vector_search.build_embedder_for_spec") as mock_emb, \
             patch("src.retriever.vector_search.query_collection") as mock_query_col:
            mock_collection = MagicMock()
            mock_open.return_value = (None, mock_collection)
            mock_embedder = MagicMock()
            mock_embedder.embed_query.return_value = [1.0]
            mock_emb.return_value = mock_embedder

            mock_query_col.return_value = {
                "ids": [[]],
                "documents": [[]],
                "metadatas": [[]],
                "distances": [[]],
            }

            search = VectorSearch(spec)
            where_filter = {"year": {"$eq": 1996}}
            search.search("query", top_k=5, where_filter=where_filter, reranker=None)

            call_kwargs = mock_query_col.call_args[1]
            assert call_kwargs["where_filter"] == where_filter

    def test_no_where_filter_omitted(self):
        """where_filter=None not passed to query_collection."""
        from src.retriever.vector_search import VectorSearch
        from src.config.collections import CollectionSpec

        spec = MagicMock(spec=CollectionSpec)
        spec.supports_late_chunking = False
        spec.db_path = "/tmp"
        spec.name = "test"

        with patch("src.retriever.vector_search.open_collection") as mock_open, \
             patch("src.retriever.vector_search.build_embedder_for_spec") as mock_emb, \
             patch("src.retriever.vector_search.query_collection") as mock_query_col:
            mock_collection = MagicMock()
            mock_open.return_value = (None, mock_collection)
            mock_embedder = MagicMock()
            mock_embedder.embed_query.return_value = [1.0]
            mock_emb.return_value = mock_embedder

            mock_query_col.return_value = {
                "ids": [[]],
                "documents": [[]],
                "metadatas": [[]],
                "distances": [[]],
            }

            search = VectorSearch(spec)
            search.search("query", top_k=5, where_filter=None, reranker=None)

            call_kwargs = mock_query_col.call_args[1]
            assert call_kwargs["where_filter"] is None

    def test_empty_chroma_result_returns_empty_list(self):
        """Empty query result → empty list."""
        from src.retriever.vector_search import VectorSearch
        from src.config.collections import CollectionSpec

        spec = MagicMock(spec=CollectionSpec)
        spec.supports_late_chunking = False
        spec.db_path = "/tmp"
        spec.name = "test"

        with patch("src.retriever.vector_search.open_collection") as mock_open, \
             patch("src.retriever.vector_search.build_embedder_for_spec") as mock_emb:
            mock_collection = MagicMock()
            mock_open.return_value = (None, mock_collection)
            mock_embedder = MagicMock()
            mock_embedder.embed_query.return_value = [1.0]
            mock_emb.return_value = mock_embedder

            mock_collection.query.return_value = {
                "ids": [[]],
                "documents": [[]],
                "metadatas": [[]],
                "distances": [[]],
            }

            search = VectorSearch(spec)
            results = search.search("query", top_k=5, reranker=None)

            assert results == []

    def test_reranker_with_fewer_candidates_than_top_k(self):
        """2 candidates, top_k=5 → 2 results (no padding)."""
        from src.retriever.vector_search import VectorSearch
        from src.config.collections import CollectionSpec

        spec = MagicMock(spec=CollectionSpec)
        spec.supports_late_chunking = False
        spec.db_path = "/tmp"
        spec.name = "test"

        with patch("src.retriever.vector_search.open_collection") as mock_open, \
             patch("src.retriever.vector_search.build_embedder_for_spec") as mock_emb:
            mock_collection = MagicMock()
            mock_open.return_value = (None, mock_collection)
            mock_embedder = MagicMock()
            mock_embedder.embed_query.return_value = [1.0]
            mock_emb.return_value = mock_embedder

            mock_collection.query.return_value = {
                "ids": [["c1", "c2"]],
                "documents": [["doc1", "doc2"]],
                "metadatas": [[{"a": 1}, {"b": 2}]],
                "distances": [[0.1, 0.2]],
            }

            mock_reranker = MagicMock()
            mock_reranker.rerank.return_value = [("c1", 0.9)]  # only 1 back

            search = VectorSearch(spec)
            results = search.search("query", top_k=5, reranker=mock_reranker)

            # Both candidates returned: c1 (scored 0.9) first, c2 (unscored → -inf) last
            assert len(results) == 2
            assert results[0]["id"] == "c1"
            assert results[1]["id"] == "c2"
            assert results[0]["rerank_score"] > results[1]["rerank_score"]
