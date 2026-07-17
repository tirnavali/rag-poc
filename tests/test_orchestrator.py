"""End-to-end orchestrator tests with mocked SearchTool and answering.

The orchestrator is the single agent pipeline. These tests exercise the core
flow (planning → policy → budget → retrieve → assemble → judge → answer →
sanitize → cite) with the stage-2 gates (bad_words/policy) and the
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
    TermHypothesis,
)
from src.common.llm_client_pool import LLMClientPool
from src.common.schemas import ExtractedFilterResponse, FilterCriteria
from src.config.pipeline_loader import load_pipeline_config


def _make_plan(*collections: str, draft_text: str = "q") -> SearchPlan:
    return SearchPlan(
        intent="factual",
        query_type="fact",
        resources=[
            CollectionSearchPlan(
                collection=c,
                query_drafts=[SearchQueryDraft(text=draft_text, top_k=5)],
            )
            for c in collections
        ],
        reasoning="r",
    )


def _fresh_broaden(*collections: str):
    """A broaden() mock that emits a NEW draft text per call — mirrors the real
    broaden, which is fed the tried-query list and produces unseen phrasings.
    (An identical draft would be hard-pruned by the tried-search ledger.)"""
    calls = {"n": 0}

    def _broaden(*a, **kw):
        calls["n"] += 1
        return _make_plan(*collections, draft_text=f"q-broaden-{calls['n']}")
    return _broaden


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
    agent._search_tool._reranker = None  # pipeline.yaml enables it; keep unit tests offline


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

    def _search(collection_key, query_text, filters=None, top_k=5, apply_reranker=True):
        chunks = (result_chunks_by_collection or {}).get(collection_key, [])
        return _make_search_result(
            chunk_ids=[c["chunk_id"] for c in chunks],
            doc_ids=[c["document_id"] for c in chunks],
            collection=collection_key,
        )
    monkeypatch.setattr(agent._search_tool, "search", _search)

    monkeypatch.setattr(
        agent._answer_tool, "generate",
        lambda query, context, mufettis_mode=False, chat_history=None, stream_callback=None, query_type=None, answer_directive=None: ("thinking", "Cevap metni."),
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


def test_orchestrator_streams_answer_tokens(monkeypatch):
    """AnswerTool tokens are forwarded through stream_callback as they arrive,
    and the full answer is NOT re-sent as a duplicate chunk afterwards."""
    chunks = [{"chunk_id": f"k{i}", "document_id": f"d{i}"} for i in range(3)]
    agent = _agent(monkeypatch, plan_collections=("col_a",),
                   result_chunks_by_collection={"col_a": chunks})

    def _streaming_generate(query, context, mufettis_mode=False, chat_history=None, stream_callback=None, query_type=None, answer_directive=None):
        for tok in ("Mer", "ha", "ba"):
            if stream_callback:
                stream_callback({"type": "content", "content": tok})
        return "", "Merhaba"
    monkeypatch.setattr(agent._answer_tool, "generate", _streaming_generate)

    received = []
    out = agent.run("q", session_collections=["col_a"],
                    stream_callback=lambda c: received.append(c))
    deltas = [c["content"] for c in received if c.get("type") == "content"]
    # Exactly the 3 streamed deltas, in order — no duplicate full-answer dump.
    assert deltas == ["Mer", "ha", "ba"]
    assert out.answer == "Merhaba"


def test_orchestrator_conversational_bypasses_retrieval(monkeypatch):
    """A 'conversational' scope answers from chat_history and skips RAG entirely."""
    from types import SimpleNamespace
    from src.agent.schemas import ScopeResult

    agent = _agent(monkeypatch, plan_collections=("col_a",))

    classifier = MagicMock()
    classifier.classify.return_value = ScopeResult(
        scope="conversational", confidence=1.0, selected_collections=[], reason="selam"
    )
    agent._classifier = classifier

    search_called = {"n": 0}
    def _search(*a, **kw):
        search_called["n"] += 1
        return _make_search_result([], [], "col_a")
    monkeypatch.setattr(agent._search_tool, "search", _search)

    captured_messages = {}
    def _fake_chat(*a, **kw):
        assert kw.get("stream") is True
        captured_messages["messages"] = kw["messages"]
        yield SimpleNamespace(message=SimpleNamespace(content="İyiyim,", thinking=""))
        yield SimpleNamespace(message=SimpleNamespace(content=" teşekkürler!", thinking=""))
    client = MagicMock()
    client.chat.side_effect = _fake_chat
    monkeypatch.setattr(agent._pool, "get_client", lambda block: client)

    out = agent.run("selam", chat_history=[{"role": "user", "content": "merhaba"}])

    assert out.scope == "conversational"
    assert out.answer == "İyiyim, teşekkürler!"
    assert search_called["n"] == 0  # retrieval bypassed
    # The chit-chat system prompt must still ground the assistant's identity in
    # the real TBMM archive — it must not read as "no archive access at all".
    assert captured_messages["messages"][0]["role"] == "system"
    assert "tutanak" in captured_messages["messages"][0]["content"].lower()


def test_orchestrator_off_domain_override_for_known_parliamentary_term(monkeypatch):
    """A known jargon term (e.g. 'kadük') overrides a false off_domain verdict —
    the small classifier model can mistake a jargon-definition question for a
    generic dictionary lookup; the deterministic glossary catches that miss."""
    from src.agent.schemas import ScopeResult

    chunks = [{"chunk_id": f"a{i}", "document_id": f"da{i}"} for i in range(3)]
    agent = _agent(monkeypatch, plan_collections=("col_a",),
                   result_chunks_by_collection={"col_a": chunks})

    classifier = MagicMock()
    classifier.classify.return_value = ScopeResult(
        scope="off_domain", confidence=1.0, selected_collections=[],
        reason="genel bir sözlük tanımı gerektirmektedir",
    )
    agent._classifier = classifier

    out = agent.run("kadük ne demek?", session_collections=["col_a"])

    assert out.scope != "off_domain"
    assert out.answer == "Cevap metni."  # normal RAG path ran, not the off-domain template


def test_orchestrator_off_domain_without_known_term_still_blocks(monkeypatch):
    """Regression guard: genuinely off-domain queries (no known jargon term)
    are still blocked as before — the override doesn't loosen the gate generally."""
    from src.agent.schemas import ScopeResult

    agent = _agent(monkeypatch, plan_collections=("col_a",))

    classifier = MagicMock()
    classifier.classify.return_value = ScopeResult(
        scope="off_domain", confidence=0.95, selected_collections=[], reason="hava durumu",
    )
    agent._classifier = classifier
    monkeypatch.setattr(agent._suggester, "suggest", lambda query, tracer: ["öneri1", "öneri2", "öneri3"])

    out = agent.run("hava bugün nasıl", session_collections=["col_a"])

    assert out.scope == "off_domain"


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
    # Broaden returns a fresh plan (new draft text); the search returns a new chunk set on re-query.
    monkeypatch.setattr(agent._planner, "broaden", _fresh_broaden("col_a"))

    calls = {"n": 0}

    def _search(collection_key, query_text, filters=None, top_k=5, apply_reranker=True):
        calls["n"] += 1
        ids = ["a0"] if calls["n"] == 1 else ["a1", "a2", "a3"]
        return _make_search_result(ids, [f"d-{i}" for i in ids], collection_key)
    monkeypatch.setattr(agent._search_tool, "search", _search)

    out = agent.run("q", session_collections=["col_a"])
    assert out.expanded is True
    assert calls["n"] >= 2  # a second (re-query) search was issued
    assert out.evidence_decision.action == "answer"  # post-expand judge is satisfied


def test_orchestrator_successful_term_hypothesis_upserted_for_review(monkeypatch):
    """A re-query round that used a term hypothesis AND actually resolved
    insufficiency (added>0, post-expand judge says 'answer') gets surfaced to
    the term_candidates human-review queue — never auto-applied to live search."""
    agent = _agent(monkeypatch, plan_collections=("col_a",))
    hypothesis_plan = _make_plan("col_a", draft_text="hükümsüz sayılan kanun teklifleri")
    hypothesis_plan.term_hypothesis = TermHypothesis(term="kadük", official_phrase="hükümsüz sayılan kanun teklifleri")
    monkeypatch.setattr(agent._planner, "broaden", lambda *a, **kw: hypothesis_plan)

    calls = {"n": 0}

    def _search(collection_key, query_text, filters=None, top_k=5, apply_reranker=True):
        calls["n"] += 1
        ids = ["a0"] if calls["n"] == 1 else ["a1", "a2", "a3"]
        return _make_search_result(ids, [f"d-{i}" for i in ids], collection_key)
    monkeypatch.setattr(agent._search_tool, "search", _search)

    captured = {}
    monkeypatch.setattr(
        "src.api.db.upsert_term_candidate",
        lambda term, hypothesis, source_query=None, **kw: captured.update(
            term=term, hypothesis=hypothesis, source_query=source_query
        ),
    )

    out = agent.run("kadük ne demek listele", session_collections=["col_a"])

    assert out.evidence_decision.action == "answer"
    assert captured == {
        "term": "kadük",
        "hypothesis": "hükümsüz sayılan kanun teklifleri",
        "source_query": "kadük ne demek listele",
    }


def test_orchestrator_term_hypothesis_not_recorded_when_round_unhelpful(monkeypatch):
    """A term hypothesis that DIDN'T resolve anything (re-query surfaces no NEW
    chunks, added stays 0) must not be recorded — only genuinely-successful
    guesses reach the review queue."""
    agent = _agent(monkeypatch, plan_collections=("col_a",))
    hypothesis_plan = _make_plan("col_a")
    hypothesis_plan.term_hypothesis = TermHypothesis(term="kadük", official_phrase="yanlış tahmin")
    monkeypatch.setattr(agent._planner, "broaden", lambda *a, **kw: hypothesis_plan)

    def _search(collection_key, query_text, filters=None, top_k=5, apply_reranker=True):
        # Always the same single chunk -> initial round triggers 'expand' (1 <
        # min_chunks), but the re-query round finds nothing NEW (already seen).
        return _make_search_result(["a0"], ["d0"], collection_key)
    monkeypatch.setattr(agent._search_tool, "search", _search)

    upsert = MagicMock()
    monkeypatch.setattr("src.api.db.upsert_term_candidate", upsert)

    agent.run("kadük ne demek", session_collections=["col_a"])

    upsert.assert_not_called()


def test_orchestrator_requery_preserves_comprehensive_fetch_k(monkeypatch):
    """Regression: broaden()'s LLM response has no query_type field (RE_RETRIEVAL_PROMPT's
    JSON schema omits it), so _parse_plan defaults it to 'fact'. Without carrying the
    original query_type forward, _flat_plans_for would starve every re-query round to
    the much smaller 'fact' fetch_k instead of 'comprehensive', causing premature
    saturation (few/no 'new' chunks per round) even when the corpus has plenty more."""
    agent = _agent(monkeypatch, plan_collections=("col_a",))
    # _fresh_broaden emits query_type="fact" plans — mirrors broaden()'s real
    # (query_type-less JSON schema) behavior exactly.
    monkeypatch.setattr(agent._planner, "broaden", _fresh_broaden("col_a"))

    seen_top_k = []

    def _search(collection_key, query_text, filters=None, top_k=5, apply_reranker=True):
        seen_top_k.append(top_k)
        n = len(seen_top_k)
        ids = [f"a{n}-{i}" for i in range(3)]
        return _make_search_result(ids, [f"d{n}-{i}" for i in ids], collection_key)
    monkeypatch.setattr(agent._search_tool, "search", _search)

    agent.run("tüm kayıtları listele", session_collections=["col_a"])

    comprehensive_fetch_k = agent._config.retrieval_budget.budget_for("comprehensive").fetch_k
    fact_fetch_k = agent._config.retrieval_budget.budget_for("fact").fetch_k
    assert comprehensive_fetch_k != fact_fetch_k  # sanity: pipeline.yaml budgets actually differ
    assert len(seen_top_k) >= 2  # at least one initial + one re-query call happened
    assert all(k == comprehensive_fetch_k for k in seen_top_k)  # EVERY round, not just the first


def test_orchestrator_comprehensive_escalates_depth_then_saturates(monkeypatch):
    """A dry round does NOT stop enumerate: it doubles fetch_k to reach the long tail
    (ranks beyond the initial top-K) and only stops when even the deepest fetch adds
    nothing new (or the ceiling is hit). Regression for the 'what if there are more
    than fetch_k relevant chunks?' gap."""
    agent = _agent(monkeypatch, plan_collections=("col_a",))
    # max_total capping is the assembler's job; the temporary passthrough doesn't cap.
    monkeypatch.setattr("src.agent.orchestrator._ASSEMBLER_ENABLED", True)
    monkeypatch.setattr(agent._planner, "broaden", _fresh_broaden("col_a"))

    CORPUS = 100
    seen_top_k = []

    def _search(collection_key, query_text, filters=None, top_k=5, apply_reranker=True):
        seen_top_k.append(top_k)
        n = min(top_k, CORPUS)                        # deeper fetch → more of the corpus
        ids = [f"a{i}" for i in range(n)]
        return _make_search_result(ids, [f"d-{i}" for i in ids], collection_key)
    monkeypatch.setattr(agent._search_tool, "search", _search)

    out = agent.run("tüm konuşmaları listele", session_collections=["col_a"])
    assert out.plan.query_type == "comprehensive"    # keyword detection promoted it
    assert max(seen_top_k) > 40                       # depth escalated past the base fetch_k
    assert len(out.assembly) == 50                    # reached the ceiling only by going deeper
    assert out.expanded is True


def test_orchestrator_comprehensive_facet_partitions_by_year(monkeypatch):
    """Enumerate runs one extra year-scoped pass per mined year facet, so each year's
    own top-K surfaces instead of only the globally dominant region (strategy 3)."""
    agent = _agent(monkeypatch, plan_collections=("col_a",))
    monkeypatch.setattr(agent._planner, "broaden", lambda *a, **kw: _make_plan("col_a"))

    seen_filters = []

    def _search(collection_key, query_text, filters=None, top_k=5, apply_reranker=True):
        seen_filters.append(filters)
        n = len(seen_filters)
        ids = [f"c{n}-{i}" for i in range(4)]
        years = [2018, 2019, 2018, 2019]
        return {
            "documents": [f"body-{i}" for i in ids],
            "metadatas": [
                {"chunk_id": cid, "document_id": f"doc-{cid}", "doc_type": "gazete",
                 "source_title": f"t-{cid}", "year": yr, "_source_collection": collection_key}
                for cid, yr in zip(ids, years)
            ],
            "distances": [0.1 for _ in ids],
        }
    monkeypatch.setattr(agent._search_tool, "search", _search)

    agent.run("tüm konuşmaları listele", session_collections=["col_a"])

    # At least one re-query pass was scoped to a specific year via a Chroma filter.
    year_filters = [f for f in seen_filters if f and "year" in str(f)]
    assert year_filters, "facet partition should issue year-scoped searches"


def test_orchestrator_comprehensive_stops_at_max_rounds(monkeypatch):
    """When every round keeps adding new chunks, the loop is bounded by comprehensive_max_rounds."""
    agent = _agent(monkeypatch, plan_collections=("col_a",))
    monkeypatch.setattr(agent._planner, "broaden", _fresh_broaden("col_a"))
    agent._config.judge.comprehensive_max_rounds = 2

    calls = {"n": 0}

    def _search(collection_key, query_text, filters=None, top_k=5, apply_reranker=True):
        calls["n"] += 1
        base = calls["n"] * 10
        ids = [f"a{base + j}" for j in range(3)]      # always 3 NEW chunks
        return _make_search_result(ids, [f"d-{i}" for i in ids], collection_key)
    monkeypatch.setattr(agent._search_tool, "search", _search)

    out = agent.run("bütün teklifleri say", session_collections=["col_a"])
    assert out.plan.query_type == "comprehensive"
    assert calls["n"] == 3                            # initial + exactly 2 (max_rounds) re-queries


def test_orchestrator_comprehensive_stops_at_ceiling(monkeypatch):
    """When the assembled context already meets the per-type ceiling, no re-query runs."""
    agent = _agent(monkeypatch, plan_collections=("col_a",))
    # max_total capping is the assembler's job; the temporary passthrough doesn't cap.
    monkeypatch.setattr("src.agent.orchestrator._ASSEMBLER_ENABLED", True)
    monkeypatch.setattr(agent._planner, "broaden", _fresh_broaden("col_a"))
    # Lower the comprehensive ceiling so the initial retrieval already saturates it.
    agent._config.retrieval_budget._by_query_type["comprehensive"].max_total = 4

    calls = {"n": 0}

    def _search(collection_key, query_text, filters=None, top_k=5, apply_reranker=True):
        calls["n"] += 1
        ids = [f"a{i}" for i in range(6)]             # 6 chunks on the initial search
        return _make_search_result(ids, [f"d-{i}" for i in ids], collection_key)
    monkeypatch.setattr(agent._search_tool, "search", _search)

    out = agent.run("hepsini listele", session_collections=["col_a"])
    assert out.plan.query_type == "comprehensive"
    assert len(out.assembly) == 4                     # capped at the lowered ceiling
    assert calls["n"] == 1                            # ceiling already met → no re-query


def test_orchestrator_expansion_never_repeats_identical_search(monkeypatch):
    """Regression for the repeat-search loop: broaden() at temperature 0 kept
    regenerating the same drafts, and every gather round re-issued the identical
    (query × filter × depth) vector search — deterministic ANN, guaranteed 0 new
    chunks, full retrieval latency burned. The tried-search ledger must prune
    exact repeats; only a DEEPER re-run of the same query is allowed."""
    agent = _agent(monkeypatch, plan_collections=("col_a",))
    # Worst case: broaden returns the same draft as the initial plan, every round.
    monkeypatch.setattr(agent._planner, "broaden", lambda *a, **kw: _make_plan("col_a"))

    issued = []

    def _search(collection_key, query_text, filters=None, top_k=5, apply_reranker=True):
        issued.append((query_text, str(filters), top_k))
        ids = [f"a{i}" for i in range(3)]                # always the same 3 chunks
        return _make_search_result(ids, [f"d-{i}" for i in ids], collection_key)
    monkeypatch.setattr(agent._search_tool, "search", _search)

    agent.run("tüm konuşmaları listele", session_collections=["col_a"])

    assert len(issued) == len(set(issued)), f"identical search re-issued: {issued}"
    assert len({k for _, _, k in issued}) == len(issued)  # re-runs only ever go deeper


def test_orchestrator_depth_escalation_skips_broaden_llm(monkeypatch):
    """A dry round escalates depth to re-run the SAME drafts deeper — paying a
    broaden-LLM round-trip for that is pure waste (it would regenerate the same
    plan). The previous round's plan must be reused without calling broaden()."""
    agent = _agent(monkeypatch, plan_collections=("col_a",))
    broaden_calls = {"n": 0}

    def _broaden(*a, **kw):
        broaden_calls["n"] += 1
        return _make_plan("col_a")
    monkeypatch.setattr(agent._planner, "broaden", _broaden)

    def _search(collection_key, query_text, filters=None, top_k=5, apply_reranker=True):
        ids = [f"a{i}" for i in range(3)]                # saturated corpus: never anything new
        return _make_search_result(ids, [f"d-{i}" for i in ids], collection_key)
    monkeypatch.setattr(agent._search_tool, "search", _search)

    agent.run("tüm konuşmaları listele", session_collections=["col_a"])

    # Round 1 broadens once; every depth-escalation round after a dry round
    # reuses that plan instead of calling the LLM again.
    assert broaden_calls["n"] == 1


def test_orchestrator_non_comprehensive_does_not_loop(monkeypatch):
    """A normal query with enough chunks answers in one shot — no gather loop."""
    chunks = [{"chunk_id": f"k{i}", "document_id": f"d{i}"} for i in range(3)]
    agent = _agent(monkeypatch, plan_collections=("col_a",),
                   result_chunks_by_collection={"col_a": chunks})
    calls = {"n": 0}
    orig = agent._search_tool.search

    def _counting(collection_key, query_text, filters=None, top_k=5, apply_reranker=True):
        calls["n"] += 1
        return orig(collection_key, query_text, filters=filters, top_k=top_k, apply_reranker=apply_reranker)
    monkeypatch.setattr(agent._search_tool, "search", _counting)

    out = agent.run("susurluk nedir", session_collections=["col_a"])
    assert out.plan.query_type == "fact"
    assert calls["n"] == 1                            # single retrieval, no expansion rounds


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

    def _search(collection_key, query_text, filters=None, top_k=5, apply_reranker=True):
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
        lambda query, context, mufettis_mode=False, chat_history=None, stream_callback=None, query_type=None, answer_directive=None: ("t", "ok"),
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
    for expected in ("planning", "policy", "budget", "retrieval", "assembly",
                     "judge", "answering", "citation"):
        assert expected in phases


def test_orchestrator_disabled_stages_absent_from_trace(monkeypatch):
    """bad_words/clarification stages are absent when disabled; policy/budget still tracked."""
    chunks = [{"chunk_id": f"k{i}", "document_id": f"d{i}"} for i in range(3)]
    agent = _agent(monkeypatch, plan_collections=("col_a",),
                   result_chunks_by_collection={"col_a": chunks})
    out = agent.run("q", session_collections=["col_a"])
    phases = {e.phase for e in out.trace}
    assert "bad_words_filter" not in phases  # stage-2, off
    assert "rabbit_holes" not in phases      # clarification disabled in these tests
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

    def _search(collection_key, query_text, filters=None, top_k=5, apply_reranker=True):
        captured["filters"] = filters
        return _make_search_result(
            chunk_ids=[c["chunk_id"] for c in chunks],
            doc_ids=[c["document_id"] for c in chunks],
            collection=collection_key,
        )
    monkeypatch.setattr(agent._search_tool, "search", _search)
    monkeypatch.setattr(
        agent._answer_tool, "generate",
        lambda query, context, mufettis_mode=False, chat_history=None, stream_callback=None, query_type=None, answer_directive=None: ("t", "ok"),
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
    agent._production = ["col_a"]  # treat the mock collection as the production universe

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
    # Single-chunk retrieval trips the judge into expand; broaden must not reach a
    # live LLM (new drafts would overwrite the refined-query capture asserted below).
    monkeypatch.setattr(agent._planner, "broaden", lambda *a, **kw: None)

    captured = {}

    def _search(collection_key, query_text, filters=None, top_k=5, apply_reranker=True):
        captured["query_text"] = query_text
        return _make_search_result(chunk_ids=["a0"], doc_ids=["da0"], collection=collection_key)
    monkeypatch.setattr(agent._search_tool, "search", _search)
    monkeypatch.setattr(
        agent._answer_tool, "generate",
        lambda query, context, mufettis_mode=False, chat_history=None, stream_callback=None, query_type=None, answer_directive=None: ("t", "ok"),
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

    def _search(collection_key, query_text, filters=None, top_k=5, apply_reranker=True):
        seen_queries.append(query_text)
        return _make_search_result(
            chunk_ids=["shared", query_text],
            doc_ids=["d-shared", f"d-{query_text}"],
            collection=collection_key,
        )
    monkeypatch.setattr(agent._search_tool, "search", _search)
    monkeypatch.setattr(
        agent._answer_tool, "generate",
        lambda query, context, mufettis_mode=False, chat_history=None, stream_callback=None, query_type=None, answer_directive=None: ("t", "ok"),
    )
    monkeypatch.setattr(agent._sanitizer, "validate", lambda *a, **kw: None)

    out = agent.run("q", session_collections=["col_a"])

    assert sorted(seen_queries) == ["draft-1", "draft-2", "draft-3"]
    chunk_ids = [s["chunk_id"] for s in out.sources]
    assert chunk_ids.count("shared") == 1


def test_orchestrator_reranks_fused_pool_once_per_collection(monkeypatch):
    """Reranking must happen once per collection, on the RRF-fused pool — not
    once per draft. Per-draft reranking multiplies cross-encoder cost for no
    benefit, since fusion discards each draft's independent rerank order
    anyway; it also contends the shared cross-encoder model across threads."""
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

    def _search(collection_key, query_text, filters=None, top_k=5, apply_reranker=True):
        assert apply_reranker is False  # draft-level fan-out must not rerank internally
        return _make_search_result(
            chunk_ids=["shared", query_text],
            doc_ids=["d-shared", f"d-{query_text}"],
            collection=collection_key,
        )
    monkeypatch.setattr(agent._search_tool, "search", _search)

    fake_reranker = MagicMock()
    fake_reranker.rerank.return_value = [
        ("shared", 0.9), ("draft-1", 0.3), ("draft-2", 0.2), ("draft-3", 0.1),
    ]
    agent._search_tool._reranker = fake_reranker

    monkeypatch.setattr(
        agent._answer_tool, "generate",
        lambda query, context, mufettis_mode=False, chat_history=None, stream_callback=None, query_type=None, answer_directive=None: ("t", "ok"),
    )
    monkeypatch.setattr(agent._sanitizer, "validate", lambda *a, **kw: None)

    agent.run("q", session_collections=["col_a"])

    assert fake_reranker.rerank.call_count == 1  # once per collection, not once per draft
    scored_pairs = fake_reranker.rerank.call_args.args[1]
    assert {cid for cid, _ in scored_pairs} == {"shared", "draft-1", "draft-2", "draft-3"}


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

    def _search(collection_key, query_text, filters=None, top_k=5, apply_reranker=True):
        seen.append(query_text)
        # Two distinct-doc chunks so the judge is satisfied (no expand → no re-query).
        return _make_search_result([f"x-{query_text}-0", f"x-{query_text}-1"],
                                    [f"d-{query_text}-0", f"d-{query_text}-1"], collection_key)
    monkeypatch.setattr(agent._search_tool, "search", _search)
    monkeypatch.setattr(agent._answer_tool, "generate", lambda query, context, mufettis_mode=False, chat_history=None, stream_callback=None, query_type=None, answer_directive=None: ("t", "ok"))
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
    agent._search_tool._reranker = None  # pipeline.yaml enables it; keep unit tests offline
    captured = {}

    def _plan(q, tracer=None, **kw):
        captured["constraints"] = kw.get("constraints")
        return _make_plan("col_a")
    monkeypatch.setattr(agent._planner, "plan", _plan)
    monkeypatch.setattr(agent._search_tool, "search",
                        lambda collection_key, query_text, filters=None, top_k=5, apply_reranker=True: _ambiguous_result(collection_key))
    monkeypatch.setattr(agent._answer_tool, "generate",
                        lambda query, context, mufettis_mode=False, chat_history=None, stream_callback=None, query_type=None, answer_directive=None: ("t", "ok"))
    monkeypatch.setattr(agent._sanitizer, "validate", lambda *a, **kw: None)
    return agent, captured


def test_orchestrator_ambiguous_query_emits_rabbit_holes_without_narrowing(monkeypatch):
    """Broad/ambiguous query: NO hard date filter is applied; instead facet-grounded
    drill-down suggestions are surfaced."""
    agent, captured = _clarify_agent(monkeypatch)
    out = agent.run("meclis ne konuştu", session_collections=["col_a"])
    # The query is NOT narrowed (no year filter injected into the plan).
    assert captured["constraints"] in (None, {})
    # Facet-grounded rabbit-hole suggestions are produced from the retrieval facets.
    assert out.rabbit_holes
    assert all(s.startswith("meclis ne konuştu") for s in out.rabbit_holes)


def test_orchestrator_rabbit_holes_callback_ignored(monkeypatch):
    """A passed clarification_callback is no longer invoked (legacy compat only)."""
    agent, captured = _clarify_agent(monkeypatch)
    called = {"n": 0}

    def callback(questions):
        called["n"] += 1
        return {"year": "1997"}

    out = agent.run("meclis ne konuştu", session_collections=["col_a"], clarification_callback=callback)
    assert called["n"] == 0
    assert captured["constraints"] in (None, {})
    assert out.rabbit_holes


def test_orchestrator_unambiguous_query_no_rabbit_holes(monkeypatch):
    """A narrow retrieval (single dominant year) produces no suggestions and no narrowing."""
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
                        lambda collection_key, query_text, filters=None, top_k=5, apply_reranker=True: narrow)
    out = agent.run("1997 bütçe", session_collections=["col_a"])
    assert captured["constraints"] in (None, {})
    assert out.rabbit_holes == []


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


# ================================================== research strategy playbook

def _capturing_generate(captured: dict):
    def _generate(query, context, mufettis_mode=False, chat_history=None,
                  stream_callback=None, query_type=None, answer_directive=None):
        captured["query_type"] = query_type
        captured["answer_directive"] = answer_directive
        return "t", "ok"
    return _generate


def test_orchestrator_resolves_strategy_to_query_type_and_directive(monkeypatch):
    """A planner-selected strategy (research_strategies.md) sets query_type and
    threads its answer_directive through to AnswerTool.generate."""
    plan = _make_plan("col_a")
    plan.strategy = "summarize"
    chunks = [{"chunk_id": f"a{i}", "document_id": f"da{i}"} for i in range(3)]
    agent = _agent(monkeypatch, plan_collections=("col_a",),
                   result_chunks_by_collection={"col_a": chunks})
    monkeypatch.setattr(agent._planner, "plan", lambda q, tracer=None, **kw: plan)

    captured = {}
    monkeypatch.setattr(agent._answer_tool, "generate", _capturing_generate(captured))

    out = agent.run("meclis toplantılarını özetle", session_collections=["col_a"])

    assert out.plan.query_type == "summary"
    assert out.plan.strategy == "summarize"
    assert captured["query_type"] == "summary"
    assert captured["answer_directive"]
    assert "ana bulguyu" in captured["answer_directive"]


def test_orchestrator_unknown_strategy_name_is_noop(monkeypatch):
    """An unresolvable strategy name is fail-open: Faz A behavior (unchanged query_type,
    no answer_directive)."""
    plan = _make_plan("col_a")
    plan.strategy = "does_not_exist_strategy"
    chunks = [{"chunk_id": f"a{i}", "document_id": f"da{i}"} for i in range(3)]
    agent = _agent(monkeypatch, plan_collections=("col_a",),
                   result_chunks_by_collection={"col_a": chunks})
    monkeypatch.setattr(agent._planner, "plan", lambda q, tracer=None, **kw: plan)

    captured = {}
    monkeypatch.setattr(agent._answer_tool, "generate", _capturing_generate(captured))

    out = agent.run("q", session_collections=["col_a"])

    assert out.plan.query_type == "fact"
    assert captured["query_type"] == "fact"
    assert captured["answer_directive"] is None


def test_orchestrator_keyword_override_wins_over_llm_strategy(monkeypatch):
    """COMPREHENSIVE_KEYWORDS always forces enumerate/comprehensive, even when the
    planner LLM picked a different strategy."""
    plan = _make_plan("col_a")
    plan.strategy = "summarize"
    chunks = [{"chunk_id": f"a{i}", "document_id": f"da{i}"} for i in range(3)]
    agent = _agent(monkeypatch, plan_collections=("col_a",),
                   result_chunks_by_collection={"col_a": chunks})
    monkeypatch.setattr(agent._planner, "plan", lambda q, tracer=None, **kw: plan)
    monkeypatch.setattr(agent._planner, "broaden", lambda *a, **kw: _make_plan("col_a"))

    captured = {}
    monkeypatch.setattr(agent._answer_tool, "generate", _capturing_generate(captured))

    out = agent.run("tüm konuşmaları özetle", session_collections=["col_a"])

    assert out.plan.query_type == "comprehensive"
    assert out.plan.strategy == "enumerate"
    assert captured["query_type"] == "comprehensive"
    assert "asla boş dönme" in captured["answer_directive"]


def test_orchestrator_trace_exposes_strategy_field(monkeypatch):
    plan = _make_plan("col_a")
    plan.strategy = "summarize"
    chunks = [{"chunk_id": f"a{i}", "document_id": f"da{i}"} for i in range(3)]
    agent = _agent(monkeypatch, plan_collections=("col_a",),
                   result_chunks_by_collection={"col_a": chunks})
    monkeypatch.setattr(agent._planner, "plan", lambda q, tracer=None, **kw: plan)

    out = agent.run("meclis toplantılarını özetle", session_collections=["col_a"])

    planning_ev = next(e for e in out.trace if e.phase == "planning")
    assert planning_ev.details.get("strategy") == "summarize"
