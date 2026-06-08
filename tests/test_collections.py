"""Tests for src.config.collections module."""
from pathlib import Path
from unittest.mock import patch

import pytest

from src.config.collections import CollectionSpec, DocumentType, get_available_collections


def test_context_weight_default_value():
    """CollectionSpec.context_weight defaults to 5."""
    with patch("src.config.collections.MODEL_SPECS", {
        "test-model": {
            "max_context_tokens": 512,
            "overlap_tokens": 128,
            "embed_dim": 768,
            "supports_late_chunking": False,
        }
    }):
        spec = CollectionSpec(
            name="test_col",
            db_path=Path("/tmp/test"),
            embed_model="test-model",
            doc_type=DocumentType.GAZETE,
        )
        assert spec.context_weight == 5


def test_context_weight_custom_value():
    """CollectionSpec.context_weight can be set explicitly."""
    with patch("src.config.collections.MODEL_SPECS", {
        "test-model": {
            "max_context_tokens": 512,
            "overlap_tokens": 128,
            "embed_dim": 768,
            "supports_late_chunking": False,
        }
    }):
        spec = CollectionSpec(
            name="test_col",
            db_path=Path("/tmp/test"),
            embed_model="test-model",
            doc_type=DocumentType.TUTANAK,
            context_weight=3,
        )
        assert spec.context_weight == 3


def test_get_available_collections_returns_list():
    """get_available_collections should return a list."""
    result = get_available_collections()
    assert isinstance(result, list)


def test_get_available_collections_has_required_fields():
    """Each collection dict should have required keys and correct types."""
    result = get_available_collections()
    required_fields = {"name", "type", "embedding_model", "count", "spec"}
    for collection in result:
        assert isinstance(collection, dict), f"Expected dict, got {type(collection)}"
        assert required_fields.issubset(collection.keys()), (
            f"Collection missing required fields. Got: {collection.keys()}"
        )
        # Type assertions for each field
        assert isinstance(collection["name"], str), f"name should be str, got {type(collection['name'])}"
        assert isinstance(collection["type"], str), f"type should be str, got {type(collection['type'])}"
        assert isinstance(collection["embedding_model"], str), f"embedding_model should be str, got {type(collection['embedding_model'])}"
        assert isinstance(collection["count"], int), f"count should be int, got {type(collection['count'])}"
        assert isinstance(collection["spec"], CollectionSpec), f"spec should be CollectionSpec, got {type(collection['spec'])}"


def test_get_available_collections_sorted_by_name():
    """Collections should be sorted alphabetically by name."""
    result = get_available_collections()
    names = [c["name"] for c in result]
    assert names == sorted(names), f"Collections not sorted. Got: {names}"


def test_get_available_collections_handles_missing_collection():
    """get_available_collections should return count=0 for missing collections."""
    # Mock open_collection to raise an error that simulates a missing collection
    from chromadb.errors import InvalidCollectionException

    with patch("src.common.chroma.open_collection") as mock_open:
        mock_open.side_effect = InvalidCollectionException("Collection not found")
        result = get_available_collections()

        # Verify that we got a result (didn't crash)
        assert isinstance(result, list)
        # All collections should have count=0 due to mocked exception
        for collection in result:
            assert collection["count"] == 0, f"Expected count=0 for missing collection, got {collection['count']}"
