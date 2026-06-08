"""Tests for VectorRetriever wrapper — stress date filter, post-process, settings flag, RetrievalResult shape."""
from unittest.mock import MagicMock, patch

import pytest

from src.config.document_types import DocumentType


class TestVectorRetriever:
    """Production wrapper: date parsing, post-process, settings flag, return shape."""

    def test_year_in_query_builds_where_filter(self):
        """Query with year → where_filter passed to VectorSearch."""
        from src.retriever.vector_retriever import VectorRetriever
        from src.config.collections import CollectionSpec

        spec = MagicMock(spec=CollectionSpec)
        spec.supports_late_chunking = False
        spec.doc_type = DocumentType.GAZETE

        with patch("src.retriever.vector_retriever.VectorSearch") as mock_vs_class:
            mock_vs = MagicMock()
            mock_vs.search.return_value = []
            mock_vs_class.return_value = mock_vs

            retriever = VectorRetriever(spec)
            retriever.retrieve("1996 yılında ne oldu?")

            # Verify where_filter with year 1996
            call_kwargs = mock_vs.search.call_args[1]
            where_filter = call_kwargs["where_filter"]
            assert where_filter == {"year": {"$eq": 1996}}

    def test_multiple_years_or_filter(self):
        """Query mentioning 1996 and 1997 → $or filter."""
        from src.retriever.vector_retriever import VectorRetriever
        from src.config.collections import CollectionSpec

        spec = MagicMock(spec=CollectionSpec)
        spec.supports_late_chunking = False
        spec.doc_type = DocumentType.GAZETE

        with patch("src.retriever.vector_retriever.VectorSearch") as mock_vs_class:
            mock_vs = MagicMock()
            mock_vs.search.return_value = []
            mock_vs_class.return_value = mock_vs

            retriever = VectorRetriever(spec)
            retriever.retrieve("1996 ve 1997 arasında neler oldu?")

            call_kwargs = mock_vs.search.call_args[1]
            where_filter = call_kwargs["where_filter"]
            assert "$or" in where_filter

    def test_no_date_no_filter(self):
        """Query without date → where_filter=None."""
        from src.retriever.vector_retriever import VectorRetriever
        from src.config.collections import CollectionSpec

        spec = MagicMock(spec=CollectionSpec)
        spec.supports_late_chunking = False
        spec.doc_type = DocumentType.GAZETE

        with patch("src.retriever.vector_retriever.VectorSearch") as mock_vs_class:
            mock_vs = MagicMock()
            mock_vs.search.return_value = []
            mock_vs_class.return_value = mock_vs

            retriever = VectorRetriever(spec)
            retriever.retrieve("Kardak kayalıkları")

            call_kwargs = mock_vs.search.call_args[1]
            assert call_kwargs["where_filter"] is None

    def test_use_reranker_flag_on(self):
        """USE_RERANKER=True → reranker passed."""
        from src.retriever.vector_retriever import VectorRetriever
        from src.config.collections import CollectionSpec
        from src.config import settings

        spec = MagicMock(spec=CollectionSpec)
        spec.supports_late_chunking = False
        spec.doc_type = DocumentType.GAZETE

        with patch("src.retriever.vector_retriever.VectorSearch") as mock_vs_class, \
             patch.object(settings, "USE_RERANKER", True), \
             patch("src.retriever.reranker.CrossEncoderReranker"):
            mock_vs = MagicMock()
            mock_vs.search.return_value = []
            mock_vs_class.return_value = mock_vs

            retriever = VectorRetriever(spec)
            retriever.retrieve("query")

            call_kwargs = mock_vs.search.call_args[1]
            assert call_kwargs["reranker"] is not None

    def test_use_reranker_flag_off(self):
        """USE_RERANKER=False → reranker=None."""
        from src.retriever.vector_retriever import VectorRetriever
        from src.config.collections import CollectionSpec
        from src.config import settings

        spec = MagicMock(spec=CollectionSpec)
        spec.supports_late_chunking = False
        spec.doc_type = DocumentType.GAZETE

        with patch("src.retriever.vector_retriever.VectorSearch") as mock_vs_class, \
             patch.object(settings, "USE_RERANKER", False):
            mock_vs = MagicMock()
            mock_vs.search.return_value = []
            mock_vs_class.return_value = mock_vs

            retriever = VectorRetriever(spec)
            retriever.retrieve("query")

            call_kwargs = mock_vs.search.call_args[1]
            assert call_kwargs["reranker"] is None

    def test_retrieval_result_shape(self):
        """Returns RetrievalResult TypedDict with correct shape."""
        from src.retriever.vector_retriever import VectorRetriever
        from src.config.collections import CollectionSpec

        spec = MagicMock(spec=CollectionSpec)
        spec.supports_late_chunking = False
        spec.doc_type = DocumentType.GAZETE

        with patch("src.retriever.vector_retriever.VectorSearch") as mock_vs_class:
            mock_vs = MagicMock()
            mock_vs.search.return_value = [
                {
                    "id": "c1",
                    "doc": "text",
                    "meta": {"date": "1996-01-15", "source_name": "Hürriyet"},
                    "dist": 0.1,
                    "rerank_score": None,
                }
            ]
            mock_vs_class.return_value = mock_vs

            retriever = VectorRetriever(spec)
            result = retriever.retrieve("query")

            # Verify shape
            assert "documents" in result
            assert "metadatas" in result
            assert "distances" in result
            assert "is_minutes" in result
            assert "parsed_dates" in result
            assert "expanded_query" in result

            # Verify list-of-lists convention
            assert isinstance(result["documents"], list) and len(result["documents"]) == 1
            assert isinstance(result["documents"][0], list)
            assert isinstance(result["metadatas"][0], list)
            assert isinstance(result["distances"][0], list)

    def test_is_minutes_flag_from_keywords(self):
        """Query with MINUTES_KEYWORDS → is_minutes=True."""
        from src.retriever.vector_retriever import VectorRetriever
        from src.config.collections import CollectionSpec
        from src.config import settings

        spec = MagicMock(spec=CollectionSpec)
        spec.supports_late_chunking = False
        spec.doc_type = DocumentType.GAZETE

        with patch("src.retriever.vector_retriever.VectorSearch") as mock_vs_class:
            mock_vs = MagicMock()
            mock_vs.search.return_value = []
            mock_vs_class.return_value = mock_vs

            retriever = VectorRetriever(spec)

            # Query with minutes keyword
            result = retriever.retrieve("tbmm meclis tutanakları")
            assert result["is_minutes"] is True

            # Query without minutes keyword
            result = retriever.retrieve("gazete kupürleri")
            assert result["is_minutes"] is False

    def test_post_process_adds_metadata_prefix(self):
        """Result doc prefixed with Kaynak|Tarih|Yazar|Başlık."""
        from src.retriever.vector_retriever import VectorRetriever
        from src.config.collections import CollectionSpec

        spec = MagicMock(spec=CollectionSpec)
        spec.supports_late_chunking = False
        spec.doc_type = DocumentType.GAZETE

        with patch("src.retriever.vector_retriever.VectorSearch") as mock_vs_class:
            mock_vs = MagicMock()
            mock_vs.search.return_value = [
                {
                    "id": "c1",
                    "doc": "Kardak meselesi...",
                    "meta": {
                        "source_name": "Hürriyet",
                        "date": "1996-01-15",
                        "author": "Yazar Adı",
                        "source_title": "Başlık",
                    },
                    "dist": 0.1,
                    "rerank_score": None,
                }
            ]
            mock_vs_class.return_value = mock_vs

            retriever = VectorRetriever(spec)
            result = retriever.retrieve("Kardak")

            doc = result["documents"][0][0]
            assert "Kaynak: Hürriyet" in doc
            assert "Tarih: 1996-01-15" in doc
            assert "Yazar: Yazar Adı" in doc
            assert "Başlık: " in doc  # Source title field name is "source_title"

    def test_inspect_record_returns_chunk_dict(self):
        """inspect_record → dict with content, chunk_id, metadata."""
        from src.retriever.vector_retriever import VectorRetriever
        from src.config.collections import CollectionSpec

        spec = MagicMock(spec=CollectionSpec)
        spec.supports_late_chunking = False
        spec.doc_type = DocumentType.GAZETE

        with patch("src.retriever.vector_retriever.VectorSearch") as mock_vs_class:
            mock_vs = MagicMock()
            mock_collection = MagicMock()
            mock_collection.get.return_value = {
                "ids": ["c1"],
                "documents": ["full text"],
                "metadatas": [{"date": "1996-01-15"}],
            }
            mock_vs.collection = mock_collection
            mock_vs_class.return_value = mock_vs

            retriever = VectorRetriever(spec)
            record = retriever.inspect_record("gazete", "c1")

            assert record["content"] == "full text"
            assert record["chunk_id"] == "c1"
            assert record["date"] == "1996-01-15"

    def test_inspect_record_missing_id_returns_none(self):
        """inspect_record with non-existent ID → None."""
        from src.retriever.vector_retriever import VectorRetriever
        from src.config.collections import CollectionSpec

        spec = MagicMock(spec=CollectionSpec)
        spec.supports_late_chunking = False
        spec.doc_type = DocumentType.GAZETE

        with patch("src.retriever.vector_retriever.VectorSearch") as mock_vs_class:
            mock_vs = MagicMock()
            mock_collection = MagicMock()
            mock_collection.get.return_value = {
                "ids": [],
                "documents": [],
                "metadatas": [],
            }
            mock_vs.collection = mock_collection
            mock_vs_class.return_value = mock_vs

            retriever = VectorRetriever(spec)
            record = retriever.inspect_record("gazete", "nonexistent")

            assert record is None

    def test_init_requires_spec(self):
        """VectorRetriever() without spec raises TypeError."""
        from src.retriever.vector_retriever import VectorRetriever

        with pytest.raises(TypeError):
            VectorRetriever()  # type: ignore
