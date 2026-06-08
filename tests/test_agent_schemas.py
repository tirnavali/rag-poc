"""Unit tests for the Planning Agent Pydantic schemas."""
import pytest
from pydantic import ValidationError

from src.agent.schemas import (
    AgentOutput,
    AgentTraceEvent,
    BadWordsResult,
    CollectionSearchPlan,
    ScopeResult,
    SearchPlan,
    SearchQueryDraft,
    SuggestionList,
    ValidationResult,
)
from src.common.schemas import FilterCriteria


def test_search_query_draft_defaults():
    d = SearchQueryDraft(text="Kardak")
    assert d.top_k == 5
    assert d.filters is None


def test_search_query_draft_coerces_dict_filters():
    d = SearchQueryDraft(text="x", filters={"year": 1996})
    assert isinstance(d.filters, FilterCriteria)
    assert d.filters.year == 1996


def test_collection_search_plan_defaults():
    p = CollectionSearchPlan(
        collection="minutes_nomic",
        query_drafts=[SearchQueryDraft(text="x")],
    )
    assert p.mode == "parallel"
    assert p.priority == 1


def test_collection_search_plan_bad_mode():
    with pytest.raises(ValidationError):
        CollectionSearchPlan(collection="c", mode="bogus", query_drafts=[])


def test_search_plan_intent_literal():
    with pytest.raises(ValidationError):
        SearchPlan(intent="not_an_intent", resources=[], reasoning="r")


def test_search_plan_valid():
    p = SearchPlan(intent="factual", resources=[], reasoning="çünkü")
    assert p.intent == "factual"


def test_validation_result_defaults():
    v = ValidationResult(passes=True)
    assert v.checks == {}
    assert v.issues == []
    assert v.retry_hint is None
    assert v.corrected_answer is None


def test_validation_result_corrected_answer():
    v = ValidationResult(passes=False, corrected_answer="düzeltilmiş yanıt")
    assert v.corrected_answer == "düzeltilmiş yanıt"


def test_agent_trace_event_required_fields():
    with pytest.raises(ValidationError):
        AgentTraceEvent(trace_id="abc", phase="planning")  # missing latency_ms


def test_agent_output_defaults():
    o = AgentOutput(answer="cevap")
    assert o.thinking == ""
    assert o.trace == []
    assert o.sources == []
    assert o.re_retrieved is False
    assert o.plan is None


def test_bad_words_result_defaults():
    r = BadWordsResult(matched=False)
    assert r.matched is False
    assert r.matched_terms == []


def test_scope_result_literal_values():
    r = ScopeResult(scope="off_domain", confidence=0.92, reason="alan dışı")
    assert r.scope == "off_domain"
    assert 0.0 <= r.confidence <= 1.0
    assert r.reason == "alan dışı"


def test_suggestion_list_holds_three_strings():
    s = SuggestionList(suggestions=["a", "b", "c"])
    assert len(s.suggestions) == 3


def test_agent_output_default_scope_is_in_scope():
    out = AgentOutput(answer="x")
    assert out.scope == "in_scope"
    assert out.suggestions == []


def test_agent_output_off_domain_carries_suggestions():
    out = AgentOutput(
        answer="redirect",
        scope="off_domain",
        suggestions=["q1", "q2", "q3"],
    )
    assert out.scope == "off_domain"
    assert out.suggestions == ["q1", "q2", "q3"]


def test_agent_output_bad_word_scope_allowed():
    out = AgentOutput(answer="reject", scope="bad_word")
    assert out.scope == "bad_word"
