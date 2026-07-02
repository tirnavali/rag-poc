"""Unit tests for the /api/term-candidates endpoints ("Öğrenilen Terimler"
review queue). Calls the FastAPI route functions directly (not via TestClient)
to avoid spinning up the real RAGService (heavy model loading happens in
server.py's lifespan, which only fires for an actual ASGI app start — a plain
import/direct call never triggers it). Each test gets an isolated sqlite file.
"""
from __future__ import annotations

import pytest
from fastapi import HTTPException

from src.api import db
from src.api.server import approve_term_candidate, get_term_candidates, reject_term_candidate


def _isolated_db(monkeypatch, tmp_path):
    db_path = tmp_path / "test_sessions.db"
    monkeypatch.setattr(db, "DB_PATH", db_path)
    db.init_db()


def test_get_term_candidates_defaults_to_pending(monkeypatch, tmp_path):
    _isolated_db(monkeypatch, tmp_path)
    db.upsert_term_candidate("kadük", "hükümsüz sayılan kanun teklifleri")

    result = get_term_candidates()

    assert len(result) == 1
    assert result[0]["term"] == "kadük"


def test_get_term_candidates_filters_by_status(monkeypatch, tmp_path):
    _isolated_db(monkeypatch, tmp_path)
    db.upsert_term_candidate("kadük", "hükümsüz sayılan kanun teklifleri")
    cid = db.list_term_candidates(status="pending")[0]["id"]
    db.set_term_candidate_status(cid, "approved")

    assert get_term_candidates(status="pending") == []
    assert len(get_term_candidates(status="approved")) == 1


def test_approve_term_candidate_updates_status_and_live_synonyms(monkeypatch, tmp_path):
    _isolated_db(monkeypatch, tmp_path)
    db.upsert_term_candidate("kadük", "hükümsüz sayılan kanun teklifleri")
    cid = db.list_term_candidates(status="pending")[0]["id"]

    result = approve_term_candidate(cid)

    assert result == {"id": cid, "status": "approved"}
    assert db.list_approved_term_synonyms() == {"kadük": ["hükümsüz sayılan kanun teklifleri"]}


def test_reject_term_candidate_updates_status_and_negative_constraints(monkeypatch, tmp_path):
    _isolated_db(monkeypatch, tmp_path)
    db.upsert_term_candidate("kadük", "yanlış")
    cid = db.list_term_candidates(status="pending")[0]["id"]

    result = reject_term_candidate(cid)

    assert result == {"id": cid, "status": "rejected"}
    assert db.list_rejected_term_hypotheses() == [{"term": "kadük", "hypothesis": "yanlış"}]


def test_approve_unknown_id_raises_404(monkeypatch, tmp_path):
    _isolated_db(monkeypatch, tmp_path)
    with pytest.raises(HTTPException) as exc_info:
        approve_term_candidate(999999)
    assert exc_info.value.status_code == 404


def test_reject_unknown_id_raises_404(monkeypatch, tmp_path):
    _isolated_db(monkeypatch, tmp_path)
    with pytest.raises(HTTPException) as exc_info:
        reject_term_candidate(999999)
    assert exc_info.value.status_code == 404
