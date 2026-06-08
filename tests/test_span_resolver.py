"""Tests for chunk_id to span resolution."""

import pytest
import json
import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.common.span_resolver import chunk_id_to_span, resolve_spans_from_cache
from src.config.collections import CollectionSpec


@pytest.fixture
def mock_spec():
    """Minimal collection spec for testing."""
    spec = MagicMock(spec=CollectionSpec)
    spec.name = "test_collection"
    spec.min_chunk_chars = 500
    spec.max_chunk_chars = 1000
    return spec


class TestChunkIdToSpan:
    def test_malformed_chunk_id(self, mock_spec):
        # No underscore
        result = chunk_id_to_span("no_separator", mock_spec)
        assert result is None

    def test_chunk_id_missing_manifest(self, mock_spec, tmp_path):
        """Mock: chunk_id exists but document_id not in manifest."""
        chunk_id = "missing_doc_42"
        with patch("src.common.span_resolver.sqlite3.connect") as mock_conn:
            mock_db = MagicMock()
            mock_conn.return_value = mock_db
            mock_db.execute.return_value.fetchone.return_value = None
            result = chunk_id_to_span(chunk_id, mock_spec)
            assert result is None

    def test_chunk_id_missing_cache_file(self, mock_spec, tmp_path, monkeypatch):
        """Mock: document exists but cache file doesn't."""
        from src.config import settings

        monkeypatch.setattr(settings, "MANIFEST_DB", str(tmp_path / "manifest.db"))
        monkeypatch.setattr(settings, "PARSE_CACHE_DIR", tmp_path / "cache")
        monkeypatch.setattr(settings, "OCR_ENGINE", "tesseract")

        chunk_id = "doc_id_0"
        with patch("src.common.span_resolver.sqlite3.connect") as mock_conn:
            mock_db = MagicMock()
            mock_conn.return_value = mock_db
            mock_row = MagicMock()
            mock_row.__getitem__ = lambda self, key: "/tmp/source.pdf"
            mock_db.execute.return_value.fetchone.return_value = mock_row

            with patch("src.common.span_resolver.Path.exists", return_value=True):
                with patch("src.common.span_resolver._file_hash", return_value="abc123"):
                    result = chunk_id_to_span(chunk_id, mock_spec)
                    assert result is None

    def test_chunk_index_out_of_bounds(self, mock_spec, tmp_path, monkeypatch):
        """Mock: cache file exists but chunk_idx too large."""
        from src.config import settings

        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        monkeypatch.setattr(settings, "MANIFEST_DB", str(tmp_path / "manifest.db"))
        monkeypatch.setattr(settings, "PARSE_CACHE_DIR", cache_dir)
        monkeypatch.setattr(settings, "OCR_ENGINE", "tesseract")

        chunk_id = "doc_id_5"  # idx=5 but cache has only 2 chunks
        cache_key = "dummy"
        cache_file = cache_dir / f"{cache_key}.json"
        cache_file.write_text(json.dumps({"chunks": [{"span": [0, 100], "text": "chunk0"}]}))

        with patch("src.common.span_resolver.sqlite3.connect") as mock_conn:
            mock_db = MagicMock()
            mock_conn.return_value = mock_db
            mock_row = MagicMock()
            mock_row.__getitem__ = lambda self, key: "/tmp/source.pdf"
            mock_db.execute.return_value.fetchone.return_value = mock_row

            with patch("src.common.span_resolver.Path.exists", return_value=True):
                with patch("src.common.span_resolver._file_hash", return_value=cache_key):
                    result = chunk_id_to_span(chunk_id, mock_spec)
                    assert result is None


class TestResolveSpansFromCache:
    def test_empty_chunk_ids(self, mock_spec):
        spans, errors = resolve_spans_from_cache([], mock_spec)
        assert spans == []
        assert errors == []

    def test_malformed_chunk_ids(self, mock_spec):
        with patch("src.common.span_resolver.sqlite3.connect") as mock_conn:
            mock_db = MagicMock()
            mock_conn.return_value = mock_db
            spans, errors = resolve_spans_from_cache(["bad_format", "no_underscore"], mock_spec)
            assert len(spans) == 0
            assert len(errors) == 2
            assert "format tanınmadı" in errors[0]
