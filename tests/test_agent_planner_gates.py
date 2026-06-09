"""Pre-planner gate flow tests with mocked components."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.agent.planner import PlanningAgent
from src.agent.schemas import (
    BadWordsResult,
    ScopeResult,
    SearchPlan,
    ValidationResult,
)


@pytest.fixture
def agent(monkeypatch):
    """Build a PlanningAgent with all collaborators mocked."""
    config = MagicMock()
    config.classifier.confidence_threshold = 0.6
    config.classifier.enabled = True
    config.bad_words_filter.enabled = True
    config.bad_words_filter.response_message = "Lütfen saygılı dil kullanın."
    config.off_domain_response_template = (
        "Alan dışı.\n1. {suggestion_0}\n2. {suggestion_1}\n3. {suggestion_2}"
    )
    config.planner.re_retrieval_max_retries = 0
    config.sanitizer.max_retries = 0
    pool = MagicMock()

    factory = lambda *a, **kw: MagicMock()
    monkeypatch.setattr("src.agent.planner.BadWordsFilter", factory)
    monkeypatch.setattr("src.agent.planner.ScopeClassifier", factory)
    monkeypatch.setattr("src.agent.planner.Suggester", factory)
    monkeypatch.setattr("src.agent.planner.SearchTool", factory)
    monkeypatch.setattr("src.agent.planner.ContextBuilderTool", factory)
    monkeypatch.setattr("src.agent.planner.AnswerTool", factory)
    monkeypatch.setattr("src.agent.planner.SanitizerAgent", factory)

    return PlanningAgent(config, pool)


def test_bad_words_short_circuit_returns_bad_word_scope(agent):
    agent._bad_words.check.return_value = BadWordsResult(matched=True, matched_terms=["aptal"])

    output = agent.run("aptal bir soru")

    assert output.scope == "bad_word"
    assert "saygılı dil" in output.answer
    assert output.plan is None
    assert output.sources == []
    agent._classifier.classify.assert_not_called()


def test_off_domain_short_circuit_returns_off_domain_scope(agent):
    agent._bad_words.check.return_value = BadWordsResult(matched=False)
    agent._classifier.classify.return_value = ScopeResult(
        scope="off_domain", confidence=0.9, reason="hava"
    )
    agent._suggester.suggest.return_value = ["q1", "q2", "q3"]

    output = agent.run("hava bugün nasıl")

    assert output.scope == "off_domain"
    assert output.suggestions == ["q1", "q2", "q3"]
    assert "1. q1" in output.answer
    assert "2. q2" in output.answer
    assert "3. q3" in output.answer
    assert output.plan is None


def test_low_confidence_off_domain_falls_through_to_planner(agent, monkeypatch):
    agent._bad_words.check.return_value = BadWordsResult(matched=False)
    agent._classifier.classify.return_value = ScopeResult(
        scope="off_domain", confidence=0.3, reason="emin değilim"
    )
    monkeypatch.setattr(agent, "_generate_plan", lambda *a, **kw: None)
    monkeypatch.setattr(agent, "_fallback_plan", lambda q, allowed_keys=None: SearchPlan(intent="unknown", resources=[], reasoning=""))
    monkeypatch.setattr(agent, "_execute_plan", lambda *a, **kw: [])
    monkeypatch.setattr(agent, "_call_answering", lambda *a, **kw: ("", "fallback answer"))
    monkeypatch.setattr(agent, "_validate_output", lambda *a, **kw: ValidationResult(passes=True, issues=[], corrected_answer=None, retry_hint=None))
    monkeypatch.setattr(agent, "_needs_reretrieval", lambda r: False)
    monkeypatch.setattr(agent, "_needs_quality_reretrieval", lambda a, v: False)
    agent._context_tool.build.return_value = ("ctx", [])

    output = agent.run("borderline")

    assert output.scope == "in_scope"
    agent._suggester.suggest.assert_not_called()


def test_in_scope_runs_planner(agent, monkeypatch):
    agent._bad_words.check.return_value = BadWordsResult(matched=False)
    agent._classifier.classify.return_value = ScopeResult(
        scope="in_scope", confidence=0.95, reason="siyasi"
    )
    monkeypatch.setattr(agent, "_generate_plan", lambda *a, **kw: None)
    monkeypatch.setattr(agent, "_fallback_plan", lambda q, allowed_keys=None: SearchPlan(intent="unknown", resources=[], reasoning=""))
    monkeypatch.setattr(agent, "_execute_plan", lambda *a, **kw: [])
    monkeypatch.setattr(agent, "_call_answering", lambda *a, **kw: ("", "ok"))
    monkeypatch.setattr(agent, "_validate_output", lambda *a, **kw: ValidationResult(passes=True, issues=[], corrected_answer=None, retry_hint=None))
    monkeypatch.setattr(agent, "_needs_reretrieval", lambda r: False)
    monkeypatch.setattr(agent, "_needs_quality_reretrieval", lambda a, v: False)
    agent._context_tool.build.return_value = ("ctx", [])

    output = agent.run("Özal döneminde gazete")

    assert output.scope == "in_scope"
    agent._suggester.suggest.assert_not_called()


def test_disabled_bad_words_filter_skips_check(monkeypatch):
    config = MagicMock()
    config.classifier.enabled = False
    config.bad_words_filter.enabled = False
    config.off_domain_response_template = "x"
    pool = MagicMock()

    factory = lambda *a, **kw: MagicMock()
    monkeypatch.setattr("src.agent.planner.BadWordsFilter", factory)
    monkeypatch.setattr("src.agent.planner.ScopeClassifier", factory)
    monkeypatch.setattr("src.agent.planner.Suggester", factory)
    monkeypatch.setattr("src.agent.planner.SearchTool", factory)
    monkeypatch.setattr("src.agent.planner.ContextBuilderTool", factory)
    monkeypatch.setattr("src.agent.planner.AnswerTool", factory)
    monkeypatch.setattr("src.agent.planner.SanitizerAgent", factory)

    agent = PlanningAgent(config, pool)
    assert agent._bad_words is None
    assert agent._classifier is None
