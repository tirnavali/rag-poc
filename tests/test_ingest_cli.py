"""Tests for src.trainer.ingestion.ingest CLI commands.

Direct cmd_* invocation via argparse.Namespace — avoids subprocess overhead.
Manifest isolated via monkeypatch (CLI cmd_* instantiates DocumentManifest()
with default db_path captured at import time, so we patch the imported reference).
"""
from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import pytest

from src.trainer.ingestion import ingest as ingest_module
from src.trainer.ingestion.ingest import (
    _load_request,
    _validate_request,
    cmd_diff,
    cmd_list_collections,
    cmd_list_types,
    cmd_status,
    cmd_validate,
)
from src.trainer.ingestion.manifest import DocumentManifest


# ─── Fixtures ────────────────────────────────────────────────


@pytest.fixture
def isolated_manifest(monkeypatch, tmp_path):
    """Replace DocumentManifest in ingest module with one bound to tmp_path."""
    test_db = tmp_path / "test_manifest.db"
    monkeypatch.setattr(
        ingest_module,
        "DocumentManifest",
        lambda: DocumentManifest(db_path=test_db),
    )
    return test_db


@pytest.fixture
def valid_request_file(tmp_path):
    """A minimal valid ingest_request.json (uses inline press_clip — no PDF needed)."""
    data = {
        "version": "1.0",
        "collection": "gazete_arsivi",
        "batch_id": "test-batch",
        "documents": [
            {
                "document_id": "test-press-001",
                "document_type": "press_clip",
                "document_date": "1999-01-01",
                "author": "Test Yazar",
                "source_name": "Test Gazete",
                "title": "Test Başlık",
                "metadata": {"dokuman_metni": "Test metin içeriği"},
            }
        ],
    }
    p = tmp_path / "request.json"
    p.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return p


def _write_request(tmp_path, data, name="request.json"):
    p = tmp_path / name
    p.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return p


# ─── cmd_list_collections ────────────────────────────────────


def test_list_collections_runs_without_crash(capsys):
    cmd_list_collections(Namespace())
    out = capsys.readouterr().out
    # Rich table truncates long collection names; check prefix substrings
    assert "gazete_arsivi" in out
    assert "minutes_jina" in out  # truncated form acceptable
    # doc_type values must render (regression check: source_type AttributeError)
    assert "tutanak" in out
    assert "gazete" in out
    assert "onerge" in out
    assert "Doküman Tipi" in out


# ─── cmd_list_types ──────────────────────────────────────────


def test_list_types_runs(capsys):
    cmd_list_types(Namespace())
    out = capsys.readouterr().out
    assert "tutanak" in out
    assert "press_clip" in out
    assert "pdf_report" in out
    assert "kanun_teklifi" in out


# ─── cmd_status ──────────────────────────────────────────────


def test_status_empty_manifest(isolated_manifest, capsys):
    ns = Namespace(collection=None, document_type=None)
    cmd_status(ns)
    out = capsys.readouterr().out
    assert "Manifest boş" in out


def test_status_with_collection_filter(isolated_manifest, capsys):
    ns = Namespace(collection="tbmm_minutes_docling_jina_v3", document_type=None)
    cmd_status(ns)
    out = capsys.readouterr().out
    assert "tbmm_minutes_docling_jina_v3" in out


# ─── cmd_validate ────────────────────────────────────────────


def test_validate_valid_request(valid_request_file, capsys):
    ns = Namespace(validate=str(valid_request_file))
    cmd_validate(ns)
    out = capsys.readouterr().out
    assert "Doğrulama Başarılı" in out or "doğrulandı" in out


def test_validate_missing_collection_field(tmp_path):
    bad = _write_request(tmp_path, {"version": "1.0", "documents": []})
    with pytest.raises((ValueError, SystemExit)):
        cmd_validate(Namespace(validate=str(bad)))


def test_validate_missing_document_id(tmp_path):
    bad = _write_request(
        tmp_path,
        {
            "version": "1.0",
            "collection": "gazete_arsivi",
            "documents": [{"document_type": "press_clip"}],
        },
    )
    with pytest.raises(SystemExit):
        cmd_validate(Namespace(validate=str(bad)))


def test_validate_duplicate_document_ids(tmp_path):
    bad = _write_request(
        tmp_path,
        {
            "version": "1.0",
            "collection": "gazete_arsivi",
            "documents": [
                {"document_id": "x", "document_type": "press_clip"},
                {"document_id": "x", "document_type": "press_clip"},
            ],
        },
    )
    with pytest.raises(SystemExit):
        cmd_validate(Namespace(validate=str(bad)))


def test_validate_bad_document_type(tmp_path):
    bad = _write_request(
        tmp_path,
        {
            "version": "1.0",
            "collection": "gazete_arsivi",
            "documents": [{"document_id": "x", "document_type": "nonexistent_type"}],
        },
    )
    with pytest.raises(SystemExit):
        cmd_validate(Namespace(validate=str(bad)))


def test_validate_missing_source_file(tmp_path):
    bad = _write_request(
        tmp_path,
        {
            "version": "1.0",
            "collection": "tbmm_minutes_docling_jina_v3",
            "documents": [
                {
                    "document_id": "x",
                    "document_type": "tutanak",
                    "document_source": "/nonexistent/path/file.pdf",
                }
            ],
        },
    )
    with pytest.raises(SystemExit):
        cmd_validate(Namespace(validate=str(bad)))


# ─── cmd_diff ────────────────────────────────────────────────


def test_diff_empty_manifest_all_new(isolated_manifest, valid_request_file, capsys):
    ns = Namespace(diff=str(valid_request_file))
    cmd_diff(ns)
    out = capsys.readouterr().out
    assert "Yeni" in out


# ─── _load_request ───────────────────────────────────────────


def test_load_request_bad_version(tmp_path):
    bad = _write_request(
        tmp_path, {"version": "2.0", "collection": "x", "documents": []}
    )
    with pytest.raises(ValueError, match="version"):
        _load_request(bad)


def test_load_request_missing_collection_key(tmp_path):
    bad = _write_request(tmp_path, {"version": "1.0", "documents": []})
    with pytest.raises(ValueError, match="collection"):
        _load_request(bad)


def test_load_request_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        _load_request(tmp_path / "does_not_exist.json")
