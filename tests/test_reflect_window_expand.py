"""Reflect pencere-genişletme (window-expand) + chunk_index testleri (offline).

Kendi kendine yeterli (test_agent_reflect helper'larına bağımlı değil): reflect self-loop'u
gerçek graf üzerinden, scripted ReflectionOutput'lar ve suffix'li chunk-id döndüren bir
SearchTool mock'u ile sürer. Kapsam: fetch_neighbors id-inşası, _apply_reading_order
gruplama/sıralama, window-expand enjeksiyonu (out.sources'ta komşular + okuma sırası),
varsayılan-kapalı, tekrar-guard, exclude_seen dedup, tavan; + config/strateji flag'i.
"""
from __future__ import annotations

from unittest.mock import MagicMock

from src.agent.orchestrator import OrchestratorAgent
from src.agent.schemas import (
    Chunk,
    CollectionSearchPlan,
    OrchestratorState,
    ReflectionOutput,
    SearchPlan,
    SearchQueryDraft,
)
from src.common.llm_client_pool import LLMClientPool
from src.config.pipeline_loader import StrategyPlaybook, load_pipeline_config

COL = "tutanaklar_ctx1024"  # tutanak koleksiyonu → window_expand uygulanır


# --------------------------------------------------------------- builders

def _plan(strategy, *, draft="q") -> SearchPlan:
    return SearchPlan(
        intent="factual", query_type="fact", strategy=strategy,
        resources=[CollectionSearchPlan(
            collection=COL, query_drafts=[SearchQueryDraft(text=draft, top_k=5)])],
        reasoning="r",
    )


def _reflection(*, done=False, anchors=None, hop=1, next_draft="hop-q", reason=None):
    next_plan = None
    if next_draft is not None:
        next_plan = SearchPlan(
            intent="factual", query_type="fact",
            resources=[CollectionSearchPlan(
                collection=COL, query_drafts=[SearchQueryDraft(text=next_draft, top_k=5)])],
            reasoning="hop")
    return ReflectionOutput(done=done, done_reason=reason, hop_cursor=hop,
                            extracted_anchors=anchors or {}, next_plan=next_plan)


def _scripted(outputs):
    state = {"n": 0, "args": []}

    def _reflect(query, previous_plan, tracer=None, **kw):
        state["args"].append(kw)
        i = state["n"]
        state["n"] += 1
        return outputs[min(i, len(outputs) - 1)]
    _reflect.state = state
    return _reflect


def _suffix_search(document_id="docA", base=100, per_call=2):
    """Search mock returning FRESH chunks with PARSEABLE ids ``{document_id}_{i}`` so the
    window-expand anchor's chunk_id yields a (doc, index)."""
    counter = {"n": 0}

    def _search(collection_key, query_text, filters=None, top_k=5, apply_reranker=True):
        counter["n"] += 1
        start = base + (counter["n"] - 1) * 10
        idxs = [start + i for i in range(per_call)]
        return {
            "documents": [f"body-{i}" for i in idxs],
            "metadatas": [
                {"chunk_id": f"{document_id}_{i}", "document_id": document_id,
                 "doc_type": "tutanak", "source_title": f"t-{i}",
                 "_source_collection": collection_key}
                for i in idxs
            ],
            "distances": [0.1] * per_call,
        }
    _search.counter = counter
    return _search


def _agent(monkeypatch, *, plan, reflect, search, window_enabled=True,
           fetch_neighbors=None):
    cfg = load_pipeline_config()
    cfg.policy.enabled = False
    cfg.reflect.enabled = True
    cfg.reflect.window_expand.enabled = window_enabled
    pool = LLMClientPool.from_config(cfg)
    agent = OrchestratorAgent(cfg, pool)
    agent._classifier = None
    agent._config.clarification.enabled = False
    agent._config.judge.llm.enabled = False
    agent._search_tool._reranker = None
    monkeypatch.setattr(agent._planner, "plan", lambda q, tracer=None, **kw: plan)
    monkeypatch.setattr(agent._planner, "reflect", reflect)
    monkeypatch.setattr(agent._search_tool, "search", search)
    if fetch_neighbors is not None:
        monkeypatch.setattr(agent._search_tool, "fetch_neighbors", fetch_neighbors)
    monkeypatch.setattr(agent._answer_tool, "generate",
                        lambda **kw: ("thinking", "Cevap."))
    monkeypatch.setattr(agent._sanitizer, "validate", lambda *a, **kw: None)
    return agent


def _chunk(cid, doc, score=0.5):
    return Chunk(chunk_id=cid, document_id=doc, collection_name=COL, doc_type="tutanak",
                 source_title="t", text=f"body-{cid}", score=score, rerank_score=score,
                 metadata={"chunk_id": cid, "document_id": doc})


def _neighbors_stub(returned):
    """A fetch_neighbors stub that records calls and returns the given neighbor ids as the
    {documents, metadatas, distances} shape."""
    calls = []

    def _fn(collection_key, document_id, anchor_index, radius, max_total=None):
        calls.append((document_id, anchor_index, radius, max_total))
        ids = returned.get((document_id, anchor_index), [])
        return {
            "documents": [f"body-{i}" for i in ids],
            "metadatas": [
                {"chunk_id": f"{document_id}_{i}", "document_id": document_id,
                 "doc_type": "tutanak", "_source_collection": collection_key}
                for i in ids
            ],
            "distances": [0.0] * len(ids),
        }
    _fn.calls = calls
    return _fn


# --------------------------------------------------------------- unit: fetch_neighbors

def test_fetch_neighbors_builds_id_window_and_skips_missing(monkeypatch):
    from src.agent.tools import SearchTool

    st = SearchTool.__new__(SearchTool)  # bypass __init__ (needs no LLM here)
    got_ids = {}

    class _FakeColl:
        def get(self, ids, include):
            got_ids["ids"] = list(ids)
            # Only docA_9 and docA_12 "exist"; others missing → dropped.
            present = [i for i in ids if i in ("docA_9", "docA_12")]
            return {
                "ids": present,
                "documents": [f"b-{i}" for i in present],
                "metadatas": [{"document_id": "docA", "chunk_id": i} for i in present],
            }

    class _Spec:
        doc_type = None
    monkeypatch.setattr(st, "_get_search", lambda k: (MagicMock(collection=_FakeColl()), _Spec()))
    monkeypatch.setattr("src.agent.tools.normalize_metadata", lambda m: dict(m))
    monkeypatch.setattr("src.agent.tools.format_prefix", lambda m, dt: "")

    out = st.fetch_neighbors(COL, "docA", 10, radius=2)
    # window [8,12] excluding anchor 10 → 8,9,11,12
    assert got_ids["ids"] == ["docA_8", "docA_9", "docA_11", "docA_12"]
    # only existing ids returned, with distances=0
    assert [m["chunk_id"] for m in out["metadatas"]] == ["docA_9", "docA_12"]
    assert out["distances"] == [0.0, 0.0]


def test_fetch_neighbors_clamps_negative_and_respects_max_total(monkeypatch):
    from src.agent.tools import SearchTool
    st = SearchTool.__new__(SearchTool)
    seen = {}

    class _FakeColl:
        def get(self, ids, include):
            seen["ids"] = list(ids)
            return {"ids": [], "documents": [], "metadatas": []}

    class _Spec:
        doc_type = None
    monkeypatch.setattr(st, "_get_search", lambda k: (MagicMock(collection=_FakeColl()), _Spec()))
    # anchor 1, radius 3 → raw window [max(0,-2)=0 .. 4] excl 1 = 0,2,3,4; max_total=2 keeps closest
    st.fetch_neighbors(COL, "docA", 1, radius=3, max_total=2)
    assert all(int(i.rsplit("_", 1)[1]) >= 0 for i in seen["ids"])
    assert len(seen["ids"]) == 2
    assert set(seen["ids"]) == {"docA_0", "docA_2"}  # closest to anchor 1


# --------------------------------------------------------------- unit: reading-order

def test_apply_reading_order_groups_and_sorts():
    state = OrchestratorState(request_id="x", user_query="q")
    # Scattered: anchor doc chunks at positions 0 and 3 (tail neighbor), other doc between.
    state.assembled_chunks = [
        _chunk("docA_10", "docA"),   # anchor
        _chunk("docB_5", "docB"),    # unrelated
        _chunk("docA_8", "docA"),    # neighbor (came earlier in order)
        _chunk("docA_12", "docA"),   # neighbor tail
    ]
    state.window_anchor_keys = [f"{COL}\x1fdocA"]
    OrchestratorAgent._apply_reading_order(state)
    ids = [c.chunk_id for c in state.assembled_chunks]
    # docA bucket emitted contiguously, index-sorted, at first-sighting (position 0); docB untouched
    assert ids == ["docA_8", "docA_10", "docA_12", "docB_5"]


def test_apply_reading_order_noop_without_anchors():
    state = OrchestratorState(request_id="x", user_query="q")
    state.assembled_chunks = [_chunk("docA_10", "docA"), _chunk("docA_8", "docA")]
    OrchestratorAgent._apply_reading_order(state)
    assert [c.chunk_id for c in state.assembled_chunks] == ["docA_10", "docA_8"]  # unchanged


# --------------------------------------------------------------- integration: full graph

def test_window_expand_injects_neighbors_in_reading_order(monkeypatch):
    # per_call=1 → initial retrieval = docA_100, reflect hop = docA_110 (the anchor).
    fetch = _neighbors_stub({("docA", 110): [108, 109, 111, 112]})
    reflect = _scripted([
        _reflection(done=False, anchors={"sira_sayisi": 5}, hop=1, next_draft="hop-1"),
        _reflection(done=True, hop=2, next_draft=None),
    ])
    agent = _agent(monkeypatch, plan=_plan("kanun_kabul_oylama"), reflect=reflect,
                   search=_suffix_search("docA", base=100, per_call=1), fetch_neighbors=fetch)
    out = agent.run("Kalkınma kaç oyla", session_collections=[COL])
    src_ids = [s.get("chunk_id") for s in out.sources]
    # neighbors of the reflect-hop anchor (docA_110) are spliced in
    for cid in ("docA_108", "docA_109", "docA_111", "docA_112"):
        assert cid in src_ids, f"{cid} missing from sources"
    # docA block is contiguous and index-sorted (reading order): 100,108,109,110,111,112
    doc_a = [c for c in src_ids if c and c.startswith("docA_")]
    assert doc_a == sorted(doc_a, key=lambda c: int(c.rsplit("_", 1)[1]))
    assert fetch.calls[0][:2] == ("docA", 110)  # fetch_neighbors invoked for the hop anchor


def test_window_expand_disabled_by_default(monkeypatch):
    fetch = _neighbors_stub({("docA", 100): [98, 99]})
    reflect = _scripted([
        _reflection(done=False, anchors={"sira_sayisi": 5}, hop=1, next_draft="hop-1"),
        _reflection(done=True, hop=2, next_draft=None),
    ])
    agent = _agent(monkeypatch, plan=_plan("kanun_kabul_oylama"), reflect=reflect,
                   search=_suffix_search("docA", base=100), fetch_neighbors=fetch,
                   window_enabled=False)   # global kill-switch OFF
    agent.run("Kalkınma kaç oyla", session_collections=[COL])
    assert fetch.calls == []  # never fetched


def test_window_expand_repeat_guard(monkeypatch):
    # Same anchor doc across two retrieving rounds → fetch_neighbors called ONCE for that doc.
    fetch = _neighbors_stub({("docA", 100): [98, 99], ("docA", 110): [108]})
    reflect = _scripted([
        _reflection(done=False, anchors={"sira_sayisi": 5}, hop=1, next_draft="hop-1"),
        _reflection(done=False, anchors={"sira_sayisi": 5}, hop=2, next_draft="hop-2"),
        _reflection(done=True, hop=3, next_draft=None),
    ])
    # search returns docA anchors both rounds (base 100 → 100, then 110…)
    agent = _agent(monkeypatch, plan=_plan("kanun_kabul_oylama"), reflect=reflect,
                   search=_suffix_search("docA", base=100), fetch_neighbors=fetch)
    agent.run("Kalkınma kaç oyla", session_collections=[COL])
    docs_expanded = [c[0] for c in fetch.calls]
    assert docs_expanded.count("docA") == 1  # repeat guard: docA expanded once


# --------------------------------------------------------------- config / strategy flag

def test_window_expand_config_defaults():
    we = load_pipeline_config().reflect.window_expand
    assert we.enabled is False
    assert we.neighbor_radius == 2
    assert we.max_neighbors_per_hop == 8
    assert we.anchor_count == 1


def test_strategy_window_expand_flag_parsed():
    got = StrategyPlaybook._finalize({"window_expand": "true"})
    assert got["window_expand"] is True
    assert StrategyPlaybook._finalize({})["window_expand"] is False
    # live playbook: kanun_kabul_oylama opts in
    strat = load_pipeline_config().get_strategy("kanun_kabul_oylama")
    assert strat and strat.get("window_expand") is True
