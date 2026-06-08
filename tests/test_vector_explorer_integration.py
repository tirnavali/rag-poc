"""Integration tests for vector_explorer with multi-collection support."""
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock


def test_multi_collection_retriever_creates_retrievers():
    """MultiSourceRetriever can create retrievers from collection specs."""
    from src.retriever.multi_source import MultiSourceRetriever
    from src.config.collections import CollectionSpec
    from src.config.document_types import DocumentType

    # Mock VectorRetriever to avoid needing real collections
    with patch('src.retriever.multi_source.VectorRetriever') as mock_vr_class:
        # Create two specs with valid models
        spec1 = CollectionSpec(
            name="test_col1",
            db_path=Path("/tmp/test1"),
            embed_model="nomic-embed-text-v2-moe",
            doc_type=DocumentType.GAZETE,
        )
        spec2 = CollectionSpec(
            name="test_col2",
            db_path=Path("/tmp/test2"),
            embed_model="nomic-embed-text-v2-moe",
            doc_type=DocumentType.TUTANAK,
        )

        # Create multi-collection retriever
        retriever = MultiSourceRetriever(specs=[spec1, spec2])

        # Verify VectorRetriever was instantiated twice
        assert mock_vr_class.call_count == 2

        # Verify retrievers dict was created with collection names as keys
        assert len(retriever.retrievers) == 2
        assert "test_col1" in retriever.retrievers
        assert "test_col2" in retriever.retrievers


def test_multi_collection_retriever_query_workflow(mocker):
    """End-to-end: select collections → create retriever → query."""
    from src.retriever.multi_source import MultiSourceRetriever
    from src.config.collections import CollectionSpec
    from src.config.document_types import DocumentType
    from pathlib import Path

    # Create two specs with valid models
    spec1 = CollectionSpec(
        name="test_col1",
        db_path=Path("/tmp/test1"),
        embed_model="nomic-embed-text-v2-moe",
        doc_type=DocumentType.GAZETE,
    )
    spec2 = CollectionSpec(
        name="test_col2",
        db_path=Path("/tmp/test2"),
        embed_model="nomic-embed-text-v2-moe",
        doc_type=DocumentType.TUTANAK,
    )

    # Mock VectorRetriever to avoid needing real collections
    with patch('src.retriever.multi_source.VectorRetriever') as mock_vr_class:
        # Create mock retriever instances
        mock_retriever_1 = MagicMock()
        mock_retriever_2 = MagicMock()
        mock_vr_class.side_effect = [mock_retriever_1, mock_retriever_2]

        # Mock return values for retrieve() calls
        mock_retriever_1.retrieve.return_value = {
            "documents": [["doc1 from gazete"]],
            "metadatas": [[{"document_id": "d1", "chunk_index": 0}]],
            "distances": [[0.1]],
            "is_minutes": False,
            "parsed_dates": {},
            "expanded_query": None,
            "fallback_level": None,
        }
        mock_retriever_2.retrieve.return_value = {
            "documents": [["doc2 from tutanak"]],
            "metadatas": [[{"document_id": "d2", "chunk_index": 0}]],
            "distances": [[0.15]],
            "is_minutes": True,
            "parsed_dates": {},
            "expanded_query": None,
            "fallback_level": None,
        }

        # Create multi-collection retriever
        retriever = MultiSourceRetriever(specs=[spec1, spec2])

        # Verify retrievers were created for each spec
        assert len(retriever.retrievers) == 2
        assert "test_col1" in retriever.retrievers
        assert "test_col2" in retriever.retrievers

        # Query
        results = retriever.retrieve("test query", top_k=5)

        # Verify results structure
        assert "documents" in results
        assert "metadatas" in results
        assert "distances" in results

        # Verify both retrievers were called
        mock_retriever_1.retrieve.assert_called_once()
        mock_retriever_2.retrieve.assert_called_once()
