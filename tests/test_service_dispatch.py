"""Tests for RAGService dispatch between legacy PlanningAgent and OrchestratorAgent."""
from __future__ import annotations

import pytest

from src.config.pipeline_loader import load_pipeline_config
from src.generator.service import RAGService


def test_service_uses_orchestrator_when_flag_enabled(monkeypatch):
    svc = RAGService()

    # Force config load + flip flag
    cfg = load_pipeline_config()
    cfg.orchestrator.enabled = True

    captured = {}

    def fake_load(*a, **kw):
        return cfg
    monkeypatch.setattr("src.generator.service.load_pipeline_config", fake_load, raising=False)
    # Also patch the import location used inside _get_orchestrator / _get_agent
    monkeypatch.setattr("src.config.pipeline_loader.load_pipeline_config", fake_load)

    def fake_run(query, session_collections):
        captured["called"] = "orchestrator"
        captured["query"] = query
        captured["session"] = session_collections
        from src.agent.schemas import AgentOutput
        return AgentOutput(answer="ok")

    # Force orchestrator path: pre-set the attribute so the lazy-init returns it.
    class _FakeOrch:
        def __init__(self): pass
        def run(self, query, session_collections, stream_callback=None):
            return fake_run(query, session_collections)

    svc._orchestrator = _FakeOrch()
    out = svc.run_agent("hello", session_collections=["c1"])
    assert captured["called"] == "orchestrator"
    assert out.answer == "ok"


def test_service_uses_legacy_planner_when_flag_disabled(monkeypatch):
    svc = RAGService()

    cfg = load_pipeline_config()
    cfg.orchestrator.enabled = False

    def fake_load(*a, **kw):
        return cfg
    monkeypatch.setattr("src.config.pipeline_loader.load_pipeline_config", fake_load)

    captured = {}

    class _FakeLegacy:
        def run(self, query, *, trace=None, session_collections=None):
            captured["called"] = "legacy"
            captured["session"] = session_collections
            from src.agent.schemas import AgentOutput
            return AgentOutput(answer="legacy")

    svc._agent = _FakeLegacy()
    out = svc.run_agent("hello", session_collections=["c1"])
    assert captured["called"] == "legacy"
    assert out.answer == "legacy"
    # session_collections artık legacy yola da iletiliyor.
    assert captured["session"] == ["c1"]
