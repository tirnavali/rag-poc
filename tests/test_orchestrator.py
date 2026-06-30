"""End-to-end orchestrator tests with mocked SearchTool and answering.

The orchestrator is the single agent pipeline. These tests exercise the core
flow (planning → policy → allocation → retrieve → assemble → judge → answer →
sanitize → cite) with the stage-2 gates (bad_words/policy/allocation) and the
LLM-dependent stages (intent, clarification, judge-LLM) disabled for
determinism. Intent and clarification have their own focused tests below.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.agent.orchestrator import OrchestratorAgent
from src.agent.schemas import (
    CollectionSearchPlan,
    SearchPlan,
    SearchQueryDraft,
)
from src.common.llm_client_pool import LLMClientPool
from src.common.schemas import ExtractedFilterResponse, FilterCriteria
from src.config.pipeline_loader import load_pipeline_config


def _make_plan(*collections: str) -> SearchPlan:
    return SearchPlan(
        intent="factual",
        query_type="fact",
        resources=[
            CollectionSearchPlan(
                collection=c,
                query_drafts=[SearchQueryDraft(text="q", top_k=5)],
            )
            for c in collections
        ],
        reasoning="r",
    )


def _make_search_result(chunk_ids, doc_ids, collection: str) -> dict:
    return {
        "documents": [f"body-{i}" for i in chunk_ids],
        "metadatas": [
            {
                "chunk_id": cid,
                "document_id": did,
                "doc_type": "gazete",
                "source_title": f"t-{cid}",
                "_source_collection": collection,
            }
            for cid, did in zip(chunk_ids, doc_ids)
        ],
        "distances": [0.1 for _ in chunk_ids],
    }


def _disable_gates(agent) -> None:
    """Disable LLM-dependent stages so the core flow runs offline/deterministic."""
    agent._classifier = None
    agent._config.clarification.enabled = False
    agent._config.judge.llm.enabled = False


def _agent(
    monkeypatch,
    plan_collections=("gazete_arsivi",),
    result_chunks_by_collection=None,
    policy_enabled=False,
):
    cfg = load_pipeline_config()
    cfg.policy.enabled = policy_enabled
    pool = LLMClientPool.from_config(cfg)
    agent = OrchestratorAgent(cfg, pool)
    _disable_gates(agent)

    monkeypatch.setattr(
        agent._planner, "plan",
        lambda q, tracer=None, **kw: _make_plan(*plan_collections),
    )

    def _search(collection_key, query_text, filters=None, top_k=5):
        chunks = (result_chunks_by_collection or {}).get(collection_key, [])
        return _make_search_result(
            chunk_ids=[c["chunk_id"] for c in chunks],
            doc_ids=[c["document_id"] for c in chunks],
            collection=collection_key,
        )
    monkeypatch.setattr(agent._search_tool, "search", _search)

    monkeypatch.setattr(
        agent._answer_tool, "generate",
        lambda query, context, mufettis_mode=False: ("thinking", "Cevap metni."),
    )
    monkeypatch.setattr(agent._sanitizer, "validate", lambda *a, **kw: None)
    return agent


def test_orchestrator_no_allowed_collections_returns_refuse(monkeypatch):
    # Policy gate ON for this test: session selection excludes the planned collection.
    agent = _agent(monkeypatch, plan_collections=("disallowed_collection",), policy_enabled=True)
    out = agent.run("q", session_collections=["gazete_arsivi"])
    assert out.policy_result.allowed_collections == []
    assert out.answer
    assert out.answer != "Cevap metni."


def test_orchestrator_policy_disabled_allows_planner_suggestions(monkeypatch):
    chunks = [{"chunk_id": f"a{i}", "document_id": f"da{i}"} for i in range(3)]
    agent = _agent(
        monkeypatch,
        plan_collections=("col_a",),
        result_chunks_by_collection={"col_a": chunks},
        policy_enabled=False,
    )
    # Session does not include col_a, but policy is off → planner suggestion allowed.
    out = agent.run("q", session_collections=["something_else"])
    assert out.policy_result.allowed_collections == ["col_a"]
    assert out.answer == "Cevap metni."


def test_orchestrator_happy_path_returns_answer(monkeypatch):
    chunks_a = [{"chunk_id": f"a{i}", "document_id": f"da{i}"} for i in range(3)]
    chunks_b = [{"chunk_id": f"b{i}", "document_id": f"db{i}"} for i in range(3)]
    agent = _agent(
        monkeypatch,
        plan_collections=("col_a", "col_b"),
        result_chunks_by_collection={"col_a": chunks_a, "col_b": chunks_b},
    )
    out = agent.run("q", session_collections=["col_a", "col_b"])
    assert out.answer == "Cevap metni."
    assert out.evidence_decision.action == "answer"
    assert out.evidence_decision.judge_type == "heuristic"
    assert len(out.sources) >= 4
    assert out.assembly
    assert out.policy_result.allowed_collections == ["col_a", "col_b"]


def test_orchestrator_zero_chunks_returns_clarify(monkeypatch):
    agent = _agent(
        monkeypatch,
        plan_collections=("col_a",),
        result_chunks_by_collection={"col_a": []},
    )
    out = agent.run("q", session_collections=["col_a"])
    assert out.evidence_decision.action == "clarify"


def test_orchestrator_requery_expand_adds_new_chunks(monkeypatch):
    """Judge 'expand' triggers a bounded re-query that brings NEW chunks (not reserves)."""
    agent = _agent(monkeypatch, plan_collections=("col_a",))
    # Broaden returns a fresh plan; the search returns a new chunk set on re-query.
    monkeypatch.setattr(agent._planner, "broaden", lambda *a, **kw: _make_plan("col_a"))

    calls = {"n": 0}

    def _search(collection_key, query_text, filters=None, top_k=5):
        calls["n"] += 1
        ids = ["a0"] if calls["n"] == 1 else ["a1", "a2", "a3"]
        return _make_search_result(ids, [f"d-{i}" for i in ids], collection_key)
    monkeypatch.setattr(agent._search_tool, "search", _search)

    out = agent.run("q", session_collections=["col_a"])
    assert out.expanded is True
    assert calls["n"] >= 2  # a second (re-query) search was issued
    assert out.evidence_decision.action == "answer"  # post-expand judge is satisfied


def test_orchestrator_single_collection_failure_continues(monkeypatch):
    chunks_a = [{"chunk_id": f"a{i}", "document_id": f"da{i}"} for i in range(3)]
    cfg = load_pipeline_config()
    pool = LLMClientPool.from_config(cfg)
    agent = OrchestratorAgent(cfg, pool)
    _disable_gates(agent)

    monkeypatch.setattr(
        agent._planner, "plan",
        lambda q, tracer=None, **kw: _make_plan("col_a", "col_b"),
    )

    def _search(collection_key, query_text, filters=None, top_k=5):
        if collection_key == "col_b":
            raise RuntimeError("boom")
        return _make_search_result(
            chunk_ids=[c["chunk_id"] for c in chunks_a],
            doc_ids=[c["document_id"] for c in chunks_a],
            collection=collection_key,
        )
    monkeypatch.setattr(agent._search_tool, "search", _search)
    monkeypatch.setattr(
        agent._answer_tool, "generate",
        lambda query, context, mufettis_mode=False: ("t", "ok"),
    )
    monkeypatch.setattr(agent._sanitizer, "validate", lambda *a, **kw: None)

    out = agent.run("q", session_collections=["col_a", "col_b"])
    used_collections = {item.collection_name for item in (out.assembly or [])}
    assert "col_a" in used_collections


def test_orchestrator_emits_phase_trace_events(monkeypatch):
    chunks = [{"chunk_id": f"k{i}", "document_id": f"d{i}"} for i in range(4)]
    agent = _agent(
        monkeypatch,
        plan_collections=("col_a", "col_b"),
        result_chunks_by_collection={"col_a": chunks[:2], "col_b": chunks[2:]},
    )
    out = agent.run("q", session_collections=["col_a", "col_b"])
    phases = {e.phase for e in out.trace}
    for expected in ("planning", "policy", "allocation", "retrieval", "assembly",
                     "judge", "answering", "citation"):
        assert expected in phases


def test_orchestrator_disabled_stages_absent_from_trace(monkeypatch):
    """bad_words/clarification stages are absent when disabled; policy/allocation still tracked."""
    chunks = [{"chunk_id": f"k{i}", "document_id": f"d{i}"} for i in range(3)]
    agent = _agent(monkeypatch, plan_collections=("col_a",),
                   result_chunks_by_collection={"col_a": chunks})
    out = agent.run("q", session_collections=["col_a"])
    phases = {e.phase for e in out.trace}
    assert "bad_words_filter" not in phases  # stage-2, off
    assert "probe" not in phases             # clarification disabled in these tests
    assert "clarification" not in phases


def test_orchestrator_propagates_extracted_filters_to_retrieval(monkeypatch):
    """FE → Planner.plan → flat allocation filters → retrieval → search."""
    mock_fe = MagicMock()
    mock_fe.model = "test-model"
    mock_fe.extract.return_value = ExtractedFilterResponse(
        refined_query="iklim", filters=FilterCriteria(year=2023)
    )

    cfg = load_pipeline_config()
    pool = LLMClientPool.from_config(cfg)
    agent = OrchestratorAgent(cfg, pool, filter_extractor=mock_fe)
    _disable_gates(agent)

    monkeypatch.setattr(
        agent._planner, "_generate_plan",
        lambda q, tracer, allowed_keys=None: _make_plan("col_a"),
    )

    chunks = [{"chunk_id": f"a{i}", "document_id": f"da{i}"} for i in range(3)]
    captured = {}

    def _search(collection_key, query_text, filters=None, top_k=5):
        captured["filters"] = filters
        return _make_search_result(
            chunk_ids=[c["chunk_id"] for c in chunks],
            doc_ids=[c["document_id"] for c in chunks],
            collection=collection_key,
        )
    monkeypatch.setattr(agent._search_tool, "search", _search)
    monkeypatch.setattr(
        agent._answer_tool, "generate",
        lambda query, context, mufettis_mode=False: ("t", "ok"),
    )
    monkeypatch.setattr(agent._sanitizer, "validate", lambda *a, **kw: None)

    agent.run("2023 iklim", session_collections=["col_a"])

    mock_fe.extract.assert_called_once_with("2023 iklim")
    assert captured["filters"] == {"year": {"$eq": 2023}}


def test_orchestrator_falls_back_to_refined_query_when_no_drafts(monkeypatch):
    """A collection with no drafts is searched with refined_query, not the raw query."""
    mock_fe = MagicMock()
    mock_fe.model = "test-model"
    mock_fe.extract.return_value = ExtractedFilterResponse(
        refined_query="iklim", filters=FilterCriteria()
    )

    cfg = load_pipeline_config()
    pool = LLMClientPool.from_config(cfg)
    agent = OrchestratorAgent(cfg, pool, filter_extractor=mock_fe)
    _disable_gates(agent)

    plan_no_drafts = SearchPlan(
        intent="factual",
        query_type="fact",
        resources=[CollectionSearchPlan(collection="col_a", query_drafts=[])],
        reasoning="r",
    )
    monkeypatch.setattr(
        agent._planner, "_generate_plan",
        lambda q, tracer, allowed_keys=None: plan_no_drafts,
    )

    captured = {}

    def _search(collection_key, query_text, filters=None, top_k=5):
        captured["query_text"] = query_text
        return _make_search_result(chunk_ids=["a0"], doc_ids=["da0"], collection=collection_key)
    monkeypatch.setattr(agent._search_tool, "search", _search)
    monkeypatch.setattr(
        agent._answer_tool, "generate",
        lambda query, context, mufettis_mode=False: ("t", "ok"),
    )
    monkeypatch.setattr(agent._sanitizer, "validate", lambda *a, **kw: None)

    agent.run("2023 iklim degisikligi", session_collections=["col_a"])
    assert captured["query_text"] == "iklim"


def test_orchestrator_runs_each_planner_draft_as_parallel_query(monkeypatch):
    """Every planner draft runs as a separate search; RRF dedupes shared chunks."""
    cfg = load_pipeline_config()
    pool = LLMClientPool.from_config(cfg)
    agent = OrchestratorAgent(cfg, pool)
    _disable_gates(agent)

    plan = SearchPlan(
        intent="factual",
        query_type="fact",
        resources=[
            CollectionSearchPlan(
                collection="col_a",
                query_drafts=[
                    SearchQueryDraft(text="draft-1", top_k=5),
                    SearchQueryDraft(text="draft-2", top_k=5),
                    SearchQueryDraft(text="draft-3", top_k=5),
                ],
            )
        ],
        reasoning="r",
    )
    monkeypatch.setattr(agent._planner, "plan", lambda q, tracer=None, **kw: plan)

    seen_queries = []

    def _search(collection_key, query_text, filters=None, top_k=5):
        seen_queries.append(query_text)
        return _make_search_result(
            chunk_ids=["shared", query_text],
            doc_ids=["d-shared", f"d-{query_text}"],
            collection=collection_key,
        )
    monkeypatch.setattr(agent._search_tool, "search", _search)
    monkeypatch.setattr(
        agent._answer_tool, "generate",
        lambda query, context, mufettis_mode=False: ("t", "ok"),
    )
    monkeypatch.setattr(agent._sanitizer, "validate", lambda *a, **kw: None)

    out = agent.run("q", session_collections=["col_a"])

    assert sorted(seen_queries) == ["draft-1", "draft-2", "draft-3"]
    chunk_ids = [s["chunk_id"] for s in out.sources]
    assert chunk_ids.count("shared") == 1


def test_orchestrator_caps_query_variants(monkeypatch):
    """Planner.plan caps total drafts to normal_max_query_variants (breadth)."""
    cfg = load_pipeline_config()
    cfg.planner.normal_max_query_variants = 2
    pool = LLMClientPool.from_config(cfg)
    agent = OrchestratorAgent(cfg, pool)
    _disable_gates(agent)
    agent._production = ["col_a"]  # treat the mock collection as the production universe

    big_plan = SearchPlan(
        intent="factual", query_type="fact",
        resources=[CollectionSearchPlan(
            collection="col_a",
            query_drafts=[SearchQueryDraft(text=f"d{i}", top_k=5) for i in range(6)],
        )],
        reasoning="r",
    )
    monkeypatch.setattr(agent._planner, "_generate_plan", lambda q, tracer, allowed_keys=None: big_plan)

    seen = []

    def _search(collection_key, query_text, filters=None, top_k=5):
        seen.append(query_text)
        # Two distinct-doc chunks so the judge is satisfied (no expand → no re-query).
        return _make_search_result([f"x-{query_text}-0", f"x-{query_text}-1"],
                                    [f"d-{query_text}-0", f"d-{query_text}-1"], collection_key)
    monkeypatch.setattr(agent._search_tool, "search", _search)
    monkeypatch.setattr(agent._answer_tool, "generate", lambda query, context, mufettis_mode=False: ("t", "ok"))
    monkeypatch.setattr(agent._sanitizer, "validate", lambda *a, **kw: None)

    agent.run("q", session_collections=["col_a"])
    assert len(seen) == 2  # capped to normal_max_query_variants


def _ambiguous_result(collection: str) -> dict:
    """A facet-rich, chunk-bearing search result (3 distinct years → ambiguous)."""
    rows = [
        ("c0", "d0", 1997, "bütçe"),
        ("c1", "d1", 2001, "ekonomi"),
        ("c2", "d2", 2010, "dış politika"),
    ]
    return {
        "documents": [f"body-{cid}" for cid, *_ in rows],
        "metadatas": [
            {"chunk_id": cid, "document_id": did, "doc_type": "tutanak",
             "source_title": f"t-{cid}", "year": year, "topics": [topic],
             "_source_collection": collection}
            for cid, did, year, topic in rows
        ],
        "distances": [0.1, 0.1, 0.1],
    }


def _clarify_agent(monkeypatch):
    cfg = load_pipeline_config()
    cfg.judge.llm.enabled = False
    pool = LLMClientPool.from_config(cfg)
    agent = OrchestratorAgent(cfg, pool)
    agent._classifier = None  # skip intent LLM; clarification stays ENABLED
    captured = {}

    def _plan(q, tracer=None, **kw):
        captured["constraints"] = kw.get("constraints")
        return _make_plan("col_a")
    monkeypatch.setattr(agent._planner, "plan", _plan)
    monkeypatch.setattr(agent._search_tool, "search",
                        lambda collection_key, query_text, filters=None, top_k=5: _ambiguous_result(collection_key))
    monkeypatch.setattr(agent._answer_tool, "generate",
                        lambda query, context, mufettis_mode=False: ("t", "ok"))
    monkeypatch.setattr(agent._sanitizer, "validate", lambda *a, **kw: None)
    return agent, captured


def test_orchestrator_clarification_interactive_applies_constraints(monkeypatch):
    agent, captured = _clarify_agent(monkeypatch)
    seen = {}

    def callback(questions):
        seen["axes"] = [q.axis for q in questions]
        return {"year": "1997"}

    out = agent.run("meclis ne konuştu", session_collections=["col_a"], clarification_callback=callback)
    assert "year" in seen["axes"]
    assert captured["constraints"] == {"year": 1997}
    assert out.clarification is not None
    assert out.clarification.asked is True
    assert out.clarification.year == 1997


def test_orchestrator_clarification_auto_when_no_callback(monkeypatch):
    agent, captured = _clarify_agent(monkeypatch)
    # No clarification_callback → non-interactive auto-narrowing + assumption note.
    out = agent.run("meclis ne konuştu", session_collections=["col_a"])
    assert out.clarification is not None
    assert out.clarification.auto_applied is True
    assert out.clarification.note
    assert captured["constraints"] and "year" in captured["constraints"]


def test_orchestrator_clarification_skipped_when_unambiguous(monkeypatch):
    """A narrow probe (single dominant year) skips clarification entirely."""
    agent, captured = _clarify_agent(monkeypatch)
    narrow = {
        "documents": ["b0", "b1", "b2"],
        "metadatas": [
            {"chunk_id": f"c{i}", "document_id": f"d{i}", "doc_type": "tutanak",
             "source_title": "t", "year": 1997, "topics": ["bütçe"], "_source_collection": "col_a"}
            for i in range(3)
        ],
        "distances": [0.1, 0.1, 0.1],
    }
    monkeypatch.setattr(agent._search_tool, "search",
                        lambda collection_key, query_text, filters=None, top_k=5: narrow)
    out = agent.run("1997 bütçe", session_collections=["col_a"])
    assert captured["constraints"] in (None, {})
    # No clarification constraints were applied.
    assert out.clarification is None or not (out.clarification.asked or out.clarification.auto_applied)


def test_orchestrator_exposes_stage_reasoning_in_trace(monkeypatch):
    """expose_thinking surfaces judge reasoning (and answer preview) in trace details."""
    chunks = [{"chunk_id": f"k{i}", "document_id": f"d{i}"} for i in range(3)]
    agent = _agent(monkeypatch, plan_collections=("col_a",),
                   result_chunks_by_collection={"col_a": chunks})
    assert agent._config.expose_thinking  # default on
    out = agent.run("q", session_collections=["col_a"])
    judge_ev = next(e for e in out.trace if e.phase == "judge")
    assert judge_ev.details.get("reasoning")  # heuristic reason populated
    ans_ev = next(e for e in out.trace if e.phase == "answering")
    assert ans_ev.details.get("answer_preview") == "Cevap metni."


def test_orchestrator_on_phase_end_streams_each_stage(monkeypatch):
    """on_phase_end fires once per completed stage (live per-stage UI streaming)."""
    chunks = [{"chunk_id": f"k{i}", "document_id": f"d{i}"} for i in range(3)]
    agent = _agent(monkeypatch, plan_collections=("col_a",),
                   result_chunks_by_collection={"col_a": chunks})
    streamed = []
    agent.run("q", session_collections=["col_a"],
              on_phase_end=lambda ev: streamed.append(ev.phase))
    # Core stages each streamed exactly once, in order.
    for expected in ("planning", "retrieval", "assembly", "judge", "answering", "citation"):
        assert expected in streamed


def test_fuse_draft_chunks_ranks_shared_chunks_first():
    from src.agent.schemas import Chunk

    def _c(cid: str) -> Chunk:
        return Chunk(
            chunk_id=cid, document_id=cid, collection_name="col",
            doc_type="gazete", source_title="t", text="b", score=1.0,
        )

    list_a = [_c("shared"), _c("only_a")]
    list_b = [_c("only_b"), _c("shared")]
    fused = OrchestratorAgent._fuse_draft_chunks([list_a, list_b])

    assert [c.chunk_id for c in fused][0] == "shared"
    assert sorted(c.chunk_id for c in fused) == ["only_a", "only_b", "shared"]


def test_fuse_draft_chunks_single_list_passthrough():
    from src.agent.schemas import Chunk

    chunks = [
        Chunk(chunk_id=f"c{i}", document_id=f"d{i}", collection_name="col",
              doc_type="gazete", source_title="t", text="b", score=1.0)
        for i in range(3)
    ]
    fused = OrchestratorAgent._fuse_draft_chunks([chunks])
    assert [c.chunk_id for c in fused] == ["c0", "c1", "c2"]
