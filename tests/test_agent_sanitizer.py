"""Unit tests for SanitizerAgent (offline — chat is monkeypatched)."""
import pytest

from src.agent.sanitizer import SanitizerAgent
from src.common.llm_client_pool import LLMClientPool
from src.config.pipeline_loader import load_pipeline_config


def _sanitizer():
    cfg = load_pipeline_config()
    pool = LLMClientPool.from_config(cfg)
    return SanitizerAgent(pool, cfg), pool, cfg


def _stub_chat_response(payload_json: str):
    """Build an object matching ollama's chat() return shape: .message.content."""
    import types
    return types.SimpleNamespace(message=types.SimpleNamespace(content=payload_json))


def test_validate_fail_open_marks_validation_skipped(monkeypatch):
    sanitizer, pool, cfg = _sanitizer()
    client = pool.get_client(cfg.sanitizer.block)

    def boom(**kwargs):
        raise RuntimeError("sanitizer down")

    monkeypatch.setattr(client, "chat", boom)
    res = sanitizer.validate("soru", "yanıt", [])
    assert res.passes is True  # fail-open: do not block the answer
    assert res.checks == {"validation_ran": False}
    assert res.issues and "skipped" in res.issues[0].lower()


def test_validate_captures_corrected_answer_on_failure(monkeypatch):
    sanitizer, pool, cfg = _sanitizer()
    client = pool.get_client(cfg.sanitizer.block)
    payload = (
        '{"passes": false, "checks": {"is_turkish": true}, '
        '"issues": ["eksik"], "corrected_answer": "düzeltilmiş metin"}'
    )
    monkeypatch.setattr(client, "chat", lambda **k: _stub_chat_response(payload))
    res = sanitizer.validate("soru", "ham yanıt", [])
    assert res.passes is False
    assert res.corrected_answer == "düzeltilmiş metin"


def test_validate_drops_identical_corrected_answer(monkeypatch):
    sanitizer, pool, cfg = _sanitizer()
    client = pool.get_client(cfg.sanitizer.block)
    payload = (
        '{"passes": true, "checks": {}, "issues": [], '
        '"corrected_answer": "aynı yanıt"}'
    )
    monkeypatch.setattr(client, "chat", lambda **k: _stub_chat_response(payload))
    res = sanitizer.validate("soru", "aynı yanıt", [])
    # passes=True → no correction surfaced
    assert res.corrected_answer is None


def test_format_source_summary_omits_missing_fields():
    """Sparse metadata must not render as '? | ? | ?' (false-negative driver)."""
    summary = SanitizerAgent._format_source_summary([
        {"source_name": "Sabah", "date": "1997-01-04", "author": "X"},
        {"date": "1998-02-02"},                 # only date present
        {},                                     # nothing
    ])
    assert "?" not in summary
    assert "Sabah | 1997-01-04 | X" in summary
    assert "1998-02-02" in summary
    # A fully-empty source reads as a real (metadata-less) source, not "? | ? | ?".
    assert "metaveri yok" in summary


def test_format_source_summary_falls_back_through_title_keys():
    """source_name yoksa source_title/title üzerinden atıf üretilir."""
    summary = SanitizerAgent._format_source_summary([
        {"source_title": "Bütçe Görüşmeleri", "date": "2020-11-01"},
        {"title": "Genel Kurul"},
    ])
    assert "Bütçe Görüşmeleri | 2020-11-01" in summary
    assert "Genel Kurul" in summary


def test_validate_passes_with_sparse_source_metadata(monkeypatch):
    """Eksik metaveriyle bile sanitizer'a giden özet '?' içermez ve PASS dönebilir."""
    sanitizer, pool, cfg = _sanitizer()
    client = pool.get_client(cfg.sanitizer.block)
    captured = {}

    def _chat(**kwargs):
        captured["user"] = kwargs["messages"][1]["content"]
        return _stub_chat_response('{"passes": true, "checks": {}, "issues": []}')

    monkeypatch.setattr(client, "chat", _chat)
    res = sanitizer.validate(
        "soru", "yanıt",
        sources=[{"document_id": "d1"}, {"date": "1999"}],  # source_name/author yok
        context="BAĞLAM metni burada.",
    )
    assert res.passes is True
    # The prompt the LLM saw must not contain the misleading "? | ? | ?" pattern.
    assert "? | ?" not in captured["user"]
