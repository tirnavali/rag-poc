"""Unit tests for PlanningAgent orchestration logic (offline — no LLM/Chroma).

Covers the pure-Python decision logic and locks in two fixes:
  - _execute_single must call SearchTool.search(collection_key=...) (param name).
  - _execute_single must translate FilterCriteria into a Chroma where dict
    ($and/$eq), not pass a raw model_dump() dict.
"""
import pytest

from src.agent.planner import PlanningAgent
from src.agent.schemas import CollectionSearchPlan, SearchPlan, SearchQueryDraft
from src.agent.tracer import PipelineTracer
from src.common.llm_client_pool import LLMClientPool
from src.common.schemas import FilterCriteria
from src.config.pipeline_loader import load_pipeline_config


def _agent() -> PlanningAgent:
    cfg = load_pipeline_config()
    pool = LLMClientPool.from_config(cfg)
    return PlanningAgent(cfg, pool)


def test_needs_reretrieval_true_when_below_threshold():
    agent = _agent()
    assert agent._needs_reretrieval([{"documents": ["only one"]}]) is True


def test_needs_reretrieval_false_when_enough():
    agent = _agent()
    assert agent._needs_reretrieval([{"documents": ["a", "b", "c", "d"]}]) is False


def test_merge_results_deduplicates_by_chunk_id():
    agent = _agent()
    original = [{
        "documents": ["a"], "metadatas": [{"chunk_id": "1"}], "distances": [0.1],
    }]
    new = [{
        "documents": ["a-dup", "b"],
        "metadatas": [{"chunk_id": "1"}, {"chunk_id": "2"}],
        "distances": [0.2, 0.3],
    }]
    merged = agent._merge_results(original, new)
    all_ids = [m["chunk_id"] for r in merged for m in r.get("metadatas", [])]
    assert all_ids.count("1") == 1
    assert "2" in all_ids


def test_fallback_plan_fills_query_template():
    agent = _agent()
    plan = agent._fallback_plan("Kardak krizi")
    assert plan.intent == "unknown"
    assert plan.resources
    drafts = plan.resources[0].query_drafts
    assert any("Kardak" in d.text for d in drafts)


def test_new_planner_class_returns_search_plan(monkeypatch):
    """The Planner class wraps PlanningAgent._generate_plan into a public method."""
    from src.agent.planner import Planner

    cfg = load_pipeline_config()
    pool = LLMClientPool.from_config(cfg)
    planner = Planner(cfg, pool)

    def _no_plan(*a, **kw):
        return None
    monkeypatch.setattr(planner._inner, "_generate_plan", _no_plan)

    plan = planner.plan("test query")
    assert plan is not None
    assert plan.intent == "unknown"
    assert plan.resources


def test_execute_single_passes_collection_key_and_translates_filter(monkeypatch):
    agent = _agent()
    captured = {}

    def fake_search(*, collection_key, query_text, filters, top_k):
        captured["collection_key"] = collection_key
        captured["filters"] = filters
        captured["top_k"] = top_k
        return {"documents": [], "metadatas": [], "distances": []}

    monkeypatch.setattr(agent._search_tool, "search", fake_search)

    draft = SearchQueryDraft(
        text="ege adalari",
        filters=FilterCriteria(year=1996, author="Deniz Baykal"),
        top_k=7,
    )
    agent._execute_single("tbmm_minutes", draft, PipelineTracer())

    assert captured["collection_key"] == "tbmm_minutes"
    assert captured["top_k"] == 7
    # Translated to Chroma where dict (multi-field → $and), not a raw dict.
    assert "$and" in captured["filters"]


def test_execute_single_no_filter_passes_none(monkeypatch):
    agent = _agent()
    captured = {}

    def fake_search(*, collection_key, query_text, filters, top_k):
        captured["filters"] = filters
        return {"documents": [], "metadatas": [], "distances": []}

    monkeypatch.setattr(agent._search_tool, "search", fake_search)
    draft = SearchQueryDraft(text="serbest sorgu")
    agent._execute_single("tbmm_minutes", draft, PipelineTracer())
    assert captured["filters"] is None


def test_execute_plan_parallel_runs_all_drafts(monkeypatch):
    agent = _agent()
    seen = []

    def fake_search(*, collection_key, query_text, filters, top_k):
        seen.append(query_text)
        return {"documents": ["d"], "metadatas": [{"chunk_id": query_text}], "distances": [0.1]}

    monkeypatch.setattr(agent._search_tool, "search", fake_search)
    plan = SearchPlan(
        intent="factual",
        resources=[CollectionSearchPlan(
            collection="tbmm_minutes",
            mode="parallel",
            query_drafts=[SearchQueryDraft(text="q1"), SearchQueryDraft(text="q2")],
        )],
        reasoning="r",
    )
    results = agent._execute_plan(plan, PipelineTracer())
    assert len(results) == 2
    assert set(seen) == {"q1", "q2"}


def test_execute_plan_tags_re_retrieval_phase(monkeypatch):
    agent = _agent()
    monkeypatch.setattr(
        agent._search_tool, "search",
        lambda **k: {"documents": [], "metadatas": [], "distances": []},
    )
    tracer = PipelineTracer()
    plan = SearchPlan(
        intent="factual",
        resources=[CollectionSearchPlan(
            collection="tbmm_minutes",
            query_drafts=[SearchQueryDraft(text="q")],
        )],
        reasoning="r",
    )
    agent._execute_plan(plan, tracer, phase="re_retrieval")
    assert tracer.events
    assert all(e.phase == "re_retrieval" for e in tracer.events)
