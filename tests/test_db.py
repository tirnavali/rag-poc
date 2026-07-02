"""Unit tests for src/api/db.py's term_candidates table (the "Öğrenilen Terimler"
human-review queue). Each test gets an isolated sqlite file (monkeypatched
DB_PATH) so these never touch the real data_lake/rag_sessions.db.
"""
from __future__ import annotations

from src.api import db


def _isolated_db(monkeypatch, tmp_path):
    db_path = tmp_path / "test_sessions.db"
    monkeypatch.setattr(db, "DB_PATH", db_path)
    db.init_db()
    return db_path


def test_upsert_term_candidate_creates_pending_row(monkeypatch, tmp_path):
    _isolated_db(monkeypatch, tmp_path)
    db.upsert_term_candidate("kadük", "hükümsüz sayılan kanun teklifleri", source_query="q1")

    rows = db.list_term_candidates(status="pending")
    assert len(rows) == 1
    assert rows[0]["term"] == "kadük"
    assert rows[0]["hypothesis"] == "hükümsüz sayılan kanun teklifleri"
    assert rows[0]["times_seen"] == 1
    assert rows[0]["last_source_query"] == "q1"


def test_upsert_term_candidate_increments_times_seen(monkeypatch, tmp_path):
    """The same (term, hypothesis) discovered again (different query/session)
    increments the counter instead of creating a duplicate row — a confidence
    signal for the human reviewer, not an auto-approval."""
    _isolated_db(monkeypatch, tmp_path)
    db.upsert_term_candidate("kadük", "hükümsüz sayılan kanun teklifleri", source_query="q1")
    db.upsert_term_candidate("kadük", "hükümsüz sayılan kanun teklifleri", source_query="q2")
    db.upsert_term_candidate("kadük", "hükümsüz sayılan kanun teklifleri", source_query="q3")

    rows = db.list_term_candidates(status="pending")
    assert len(rows) == 1
    assert rows[0]["times_seen"] == 3
    assert rows[0]["last_source_query"] == "q3"  # most recent


def test_upsert_different_hypotheses_for_same_term_are_distinct_rows(monkeypatch, tmp_path):
    _isolated_db(monkeypatch, tmp_path)
    db.upsert_term_candidate("kadük", "hükümsüz sayılan kanun teklifleri")
    db.upsert_term_candidate("kadük", "yanlış bir tahmin")

    rows = db.list_term_candidates(status="pending")
    assert len(rows) == 2


def test_approve_moves_out_of_pending_and_into_live_synonyms(monkeypatch, tmp_path):
    _isolated_db(monkeypatch, tmp_path)
    db.upsert_term_candidate("kadük", "hükümsüz sayılan kanun teklifleri")
    candidate_id = db.list_term_candidates(status="pending")[0]["id"]

    ok = db.set_term_candidate_status(candidate_id, "approved")

    assert ok is True
    assert db.list_term_candidates(status="pending") == []
    approved = db.list_term_candidates(status="approved")
    assert len(approved) == 1
    assert approved[0]["reviewed_at"] is not None
    assert db.list_approved_term_synonyms() == {"kadük": ["hükümsüz sayılan kanun teklifleri"]}


def test_reject_surfaces_as_negative_constraint(monkeypatch, tmp_path):
    _isolated_db(monkeypatch, tmp_path)
    db.upsert_term_candidate("kadük", "yanlış bir tahmin")
    candidate_id = db.list_term_candidates(status="pending")[0]["id"]

    db.set_term_candidate_status(candidate_id, "rejected")

    rejected = db.list_rejected_term_hypotheses()
    assert rejected == [{"term": "kadük", "hypothesis": "yanlış bir tahmin"}]
    assert db.list_approved_term_synonyms() == {}


def test_set_status_on_unknown_id_returns_false(monkeypatch, tmp_path):
    _isolated_db(monkeypatch, tmp_path)
    assert db.set_term_candidate_status(999999, "approved") is False


def test_list_approved_term_synonyms_groups_multiple_hypotheses_per_term(monkeypatch, tmp_path):
    """A single term can have more than one approved official phrasing."""
    _isolated_db(monkeypatch, tmp_path)
    db.upsert_term_candidate("kadük", "hükümsüz sayılan kanun teklifleri")
    db.upsert_term_candidate("kadük", "işlemden kaldırılan teklifler")
    for row in db.list_term_candidates(status="pending"):
        db.set_term_candidate_status(row["id"], "approved")

    synonyms = db.list_approved_term_synonyms()
    assert set(synonyms["kadük"]) == {"hükümsüz sayılan kanun teklifleri", "işlemden kaldırılan teklifler"}
