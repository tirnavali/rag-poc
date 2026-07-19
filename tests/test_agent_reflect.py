"""Adaptive reflect/re-plan node tests (mode: adaptive strategies).

Two layers, both offline:
  * Orchestrator-level — patch ``Planner.reflect`` with scripted ReflectionOutputs and
    a stateful SearchTool, exercise the reflect self-loop through the real graph
    (hop advancement, anchor accumulation, DONE/max_rounds termination, fail-open,
    kill-switch, comprehensive-override reconcile, answer-directive threading,
    assembler-refresh, verification-anchor context injection).
  * Planner-level — mock ``client.chat`` and verify REFLECT_PROMPT assembly +
    ReflectionOutput parsing + fail-open None.

Mirrors the fixture style of tests/test_orchestrator.py and
tests/test_agent_planner_strategy.py. Note: the multi-hop *reasoning* itself lives in
the reflect LLM and is NOT covered here — these tests verify the plumbing.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock

from src.agent.orchestrator import OrchestratorAgent
from src.agent.planner import Planner
from src.agent.schemas import (
    CollectionSearchPlan,
    ReflectionOutput,
    SearchPlan,
    SearchQueryDraft,
)
from src.common.llm_client_pool import LLMClientPool
from src.config.pipeline_loader import load_pipeline_config

COL = "tutanaklar_ctx1024"


# ------------------------------------------------------------------- builders

def _plan(strategy: str | None, *, collection: str = COL, draft: str = "q") -> SearchPlan:
    return SearchPlan(
        intent="factual",
        query_type="fact",
        strategy=strategy,
        resources=[CollectionSearchPlan(
            collection=collection, query_drafts=[SearchQueryDraft(text=draft, top_k=5)],
        )],
        reasoning="r",
    )


def _reflection(*, done=False, anchors=None, hop=1, next_draft="hop-q",
                collection=COL, reason=None) -> ReflectionOutput:
    next_plan = None
    if next_draft is not None:
        next_plan = SearchPlan(
            intent="factual", query_type="fact",
            resources=[CollectionSearchPlan(
                collection=collection, query_drafts=[SearchQueryDraft(text=next_draft, top_k=5)],
            )],
            reasoning="hop",
        )
    return ReflectionOutput(
        done=done, done_reason=reason, hop_cursor=hop,
        extracted_anchors=anchors or {}, next_plan=next_plan,
    )


def _counting_search(per_call: int = 2):
    """Search mock returning FRESH chunk ids per call → each round adds `per_call`."""
    counter = {"n": 0}

    def _search(collection_key, query_text, filters=None, top_k=5, apply_reranker=True):
        counter["n"] += 1
        base = counter["n"] * 100
        ids = [f"c{base + i}" for i in range(per_call)]
        return {
            "documents": [f"body-{i}" for i in ids],
            "metadatas": [
                {"chunk_id": cid, "document_id": f"d{cid}", "doc_type": "tutanak",
                 "source_title": f"t-{cid}", "_source_collection": collection_key}
                for cid in ids
            ],
            "distances": [0.1] * per_call,
        }
    _search.counter = counter
    return _search


def _fixed_search(collection_key, query_text, filters=None, top_k=5, apply_reranker=True):
    """Search mock always returning the SAME ids → reflect rounds add 0 new chunks."""
    ids = ["c1", "c2", "c3"]
    return {
        "documents": [f"body-{i}" for i in ids],
        "metadatas": [
            {"chunk_id": cid, "document_id": f"d{cid}", "doc_type": "tutanak",
             "source_title": f"t-{cid}", "_source_collection": collection_key}
            for cid in ids
        ],
        "distances": [0.1] * len(ids),
    }


def _ids_search(ids_by_round: list[list[str]]):
    """Search mock returning an explicit id list per call (clamped to the last entry
    once exhausted). Lets a test stage overlap between rounds — e.g. a reflect hop that
    re-surfaces already-seen ids alongside fresh ones — to exercise chunk-id exclusion."""
    state = {"n": 0}

    def _search(collection_key, query_text, filters=None, top_k=5, apply_reranker=True):
        idx = min(state["n"], len(ids_by_round) - 1)
        ids = ids_by_round[idx]
        state["n"] += 1
        return {
            "documents": [f"body-{cid}" for cid in ids],
            "metadatas": [
                {"chunk_id": cid, "document_id": f"d{cid}", "doc_type": "tutanak",
                 "source_title": f"t-{cid}", "_source_collection": collection_key}
                for cid in ids
            ],
            "distances": [0.1] * len(ids),
        }
    _search.state = state
    return _search


def _agent(monkeypatch, *, plan, reflect, search=None, capture=None, reflect_enabled=True):
    cfg = load_pipeline_config()
    cfg.policy.enabled = False
    cfg.reflect.enabled = reflect_enabled
    pool = LLMClientPool.from_config(cfg)
    agent = OrchestratorAgent(cfg, pool)
    # offline determinism (mirror test_orchestrator._disable_gates)
    agent._classifier = None
    agent._config.clarification.enabled = False
    agent._config.judge.llm.enabled = False
    agent._search_tool._reranker = None

    monkeypatch.setattr(agent._planner, "plan", lambda q, tracer=None, **kw: plan)
    monkeypatch.setattr(agent._planner, "reflect", reflect)
    monkeypatch.setattr(agent._search_tool, "search", search or _counting_search())

    def _generate(query, context, mufettis_mode=False, chat_history=None,
                  stream_callback=None, query_type=None, answer_directive=None):
        if capture is not None:
            capture["context"] = context
            capture["answer_directive"] = answer_directive
            capture["query_type"] = query_type
        return ("thinking", "Cevap metni.")
    monkeypatch.setattr(agent._answer_tool, "generate", _generate)
    monkeypatch.setattr(agent._sanitizer, "validate", lambda *a, **kw: None)
    return agent


def _scripted(outputs):
    """A reflect() mock returning scripted outputs, capturing per-call kwargs."""
    state = {"n": 0, "args": []}

    def _reflect(query, previous_plan, tracer=None, **kw):
        state["args"].append(kw)
        i = state["n"]
        state["n"] += 1
        return outputs[min(i, len(outputs) - 1)]
    _reflect.state = state
    return _reflect


# --------------------------------------------------------- orchestrator tests

def test_reflect_advances_hops_extracts_anchors_and_merges(monkeypatch):
    reflect = _scripted([
        _reflection(done=False, anchors={"esas_no": "2/773"}, hop=1, next_draft="hop-1"),
        _reflection(done=False, anchors={"sira_sayisi": 5}, hop=2, next_draft="hop-2"),
        _reflection(done=True, hop=3, next_draft=None, reason="found"),
    ])
    capture = {}
    agent = _agent(monkeypatch, plan=_plan("kanun_kabul_oylama"), reflect=reflect,
                   search=_counting_search(2), capture=capture)

    out = agent.run("Karayolları Trafik Kanunu kaç oyla kabul edildi", session_collections=[COL])

    assert reflect.state["n"] == 3                       # 2 retrieving rounds + 1 done round
    # anchors accumulate across rounds and are handed to the NEXT reflect call
    assert reflect.state["args"][1]["extracted_anchors"] == {"esas_no": "2/773"}
    assert reflect.state["args"][2]["extracted_anchors"] == {"esas_no": "2/773", "sira_sayisi": 5}
    # initial (2) + hop-1 (2) + hop-2 (2) = 6 chunks reached the answer (assembler refreshed)
    assert len(out.sources) == 6
    # verification identity injected into the answering context
    assert "DOĞRULANAN KİMLİK" in capture["context"]
    assert "esas_no=2/773" in capture["context"]
    assert "sira_sayisi=5" in capture["context"]
    assert out.answer == "Cevap metni."


def test_reflect_stops_on_done(monkeypatch):
    reflect = _scripted([_reflection(done=True, next_draft=None, reason="done")])
    agent = _agent(monkeypatch, plan=_plan("kanun_kabul_oylama"), reflect=reflect)
    out = agent.run("bir kanun kaç oyla kabul edildi", session_collections=[COL])
    assert reflect.state["n"] == 1
    assert out.answer == "Cevap metni."


def test_reflect_stops_at_max_rounds(monkeypatch):
    # Never signals done → bounded by strategy.max_rounds (kanun_kabul_oylama = 4).
    reflect = _scripted([_reflection(done=False, next_draft="hop", hop=1)])
    agent = _agent(monkeypatch, plan=_plan("kanun_kabul_oylama"), reflect=reflect,
                   search=_counting_search(2))
    out = agent.run("bir kanun kaç oyla", session_collections=[COL])
    strat_max = agent._config.get_strategy("kanun_kabul_oylama")["max_rounds"]
    assert reflect.state["n"] == strat_max              # terminated by ceiling, not recursion limit
    assert out.answer == "Cevap metni."
    # max_rounds exit must still run answering→validation→citation (regression: the
    # max_rounds path routes straight to "answering", not via _final_route).
    assert len(out.sources) > 0
    phases = {ev.phase for ev in out.trace}
    assert {"answering", "citation"} <= phases


def test_reflect_fail_open_when_llm_returns_none(monkeypatch):
    reflect = MagicMock(return_value=None)
    broaden = MagicMock(return_value=None)
    agent = _agent(monkeypatch, plan=_plan("kanun_kabul_oylama"), reflect=reflect)
    monkeypatch.setattr(agent._planner, "broaden", broaden)
    out = agent.run("bir kanun kaç oyla", session_collections=[COL])
    assert reflect.call_count == 1
    assert broaden.called                                # fell back to the generic broaden path
    assert out.answer == "Cevap metni."


def test_reflect_done_false_but_no_next_plan_stops(monkeypatch):
    reflect = _scripted([_reflection(done=False, next_draft=None, hop=1)])  # next_plan None
    agent = _agent(monkeypatch, plan=_plan("kanun_kabul_oylama"), reflect=reflect)
    out = agent.run("bir kanun kaç oyla", session_collections=[COL])
    assert reflect.state["n"] == 1                       # no crash, treated as done
    assert out.answer == "Cevap metni."


def test_reflect_not_entered_for_non_adaptive_strategy(monkeypatch):
    reflect = MagicMock()
    agent = _agent(monkeypatch, plan=_plan(None), reflect=reflect)  # no strategy → not adaptive
    out = agent.run("basit bir soru", session_collections=[COL])
    assert reflect.call_count == 0
    assert out.answer == "Cevap metni."


def test_reflect_kill_switch_suppresses_reflect(monkeypatch):
    reflect = MagicMock()
    agent = _agent(monkeypatch, plan=_plan("kanun_kabul_oylama"), reflect=reflect,
                   reflect_enabled=False)
    out = agent.run("bir kanun kaç oyla", session_collections=[COL])
    assert reflect.call_count == 0                       # adaptive but reflect disabled → generic path
    assert out.answer == "Cevap metni."


def test_reflect_added_zero_but_not_done_continues(monkeypatch):
    # Search always returns the same ids → every reflect hop merges 0 new chunks. The
    # generic loop would stop on added==0; the reflect self-loop must NOT — it runs to
    # max_rounds because the procedure (not saturation) decides termination.
    reflect = _scripted([_reflection(done=False, next_draft="hop", hop=1)])
    agent = _agent(monkeypatch, plan=_plan("kanun_kabul_oylama"), reflect=reflect,
                   search=_fixed_search)
    out = agent.run("bir kanun kaç oyla", session_collections=[COL])
    strat_max = agent._config.get_strategy("kanun_kabul_oylama")["max_rounds"]
    assert reflect.state["n"] == strat_max
    assert out.answer == "Cevap metni."


def test_reflect_not_blocked_when_initial_pool_fills_query_type_ceiling(monkeypatch):
    # Regression (live-run bug): reasoning/summary adaptive strategies' single-shot initial
    # retrieval already returns fetch_k=15 chunks == max_total_primary. If the reflect ceiling
    # were the query_type ceiling, reflect would stop at round 1 with stop_reason="ceiling"
    # WITHOUT running any hop. The reflect volume cap (reflect.max_total_chunks=50) must not
    # trip on that initial pool.
    reflect = _scripted([
        _reflection(done=False, next_draft="hop-1", hop=1),
        _reflection(done=True, next_draft=None, hop=2),
    ])
    agent = _agent(monkeypatch, plan=_plan("kanun_kabul_oylama"), reflect=reflect,
                   search=_counting_search(15))  # initial pool = 15 (== max_total_primary)
    out = agent.run("bir kanun kaç oyla kabul edildi", session_collections=[COL])
    assert reflect.state["n"] == 2               # ran a hop; NOT blocked at round-1 ceiling
    assert out.answer == "Cevap metni."


def test_reflect_volume_ceiling_still_guards(monkeypatch):
    # The volume cap remains a real safety valve: once assembled >= reflect.max_total_chunks
    # the loop stops before calling the reflect LLM.
    reflect = _scripted([_reflection(done=False, next_draft="hop", hop=1)])
    agent = _agent(monkeypatch, plan=_plan("kanun_kabul_oylama"), reflect=reflect,
                   search=_counting_search(10))
    agent._config.reflect.max_total_chunks = 5   # initial 10 >= 5 → ceiling on round 1
    out = agent.run("bir kanun kaç oyla", session_collections=[COL])
    assert reflect.state["n"] == 0               # reflect LLM never called (guarded)
    assert out.answer == "Cevap metni."


def test_reflect_assembler_refresh_surfaces_new_chunks(monkeypatch):
    # One retrieving round (adds 2) then done → assembled pool must include the reflect
    # chunks (proves _run_assembler ran after _merge_new_chunks).
    reflect = _scripted([
        _reflection(done=False, next_draft="hop-1", hop=1),
        _reflection(done=True, next_draft=None, hop=2),
    ])
    agent = _agent(monkeypatch, plan=_plan("kanun_kabul_oylama"), reflect=reflect,
                   search=_counting_search(2))
    out = agent.run("bir kanun kaç oyla", session_collections=[COL])
    assert len(out.sources) == 4                          # initial 2 + reflect 2


def test_reflect_preserves_adaptive_strategy_under_comprehensive_keyword(monkeypatch):
    # "tüm" is a COMPREHENSIVE_KEYWORD; kanun_gorusmeleri (adaptive, comprehensive) must
    # NOT be clobbered to "enumerate" by the deterministic override in _node_planning.
    reflect = _scripted([_reflection(done=True, next_draft=None)])
    agent = _agent(monkeypatch, plan=_plan("kanun_gorusmeleri"), reflect=reflect)
    out = agent.run("X kanununun tüm görüşmeleri neler", session_collections=[COL])
    assert out.plan.strategy == "kanun_gorusmeleri"
    assert reflect.state["n"] == 1


def test_reflect_answer_directive_reaches_generate(monkeypatch):
    reflect = _scripted([_reflection(done=True, next_draft=None)])
    capture = {}
    agent = _agent(monkeypatch, plan=_plan("kanun_kabul_oylama"), reflect=reflect, capture=capture)
    agent.run("bir kanun kaç oyla kabul edildi", session_collections=[COL])
    # kanun_kabul_oylama's answer_directive (the two-result vote instruction) is threaded through.
    assert capture["answer_directive"]
    assert "İKİ OLASILIĞI AYIR" in capture["answer_directive"]


# ------------------------------------------------ chunk-id exclusion filter tests

def _capture_run_retrieval(agent):
    """Wrap agent._run_retrieval, recording the exclude_ids kwarg of every call."""
    calls: list = []
    orig = agent._run_retrieval

    def _wrapped(plans, fallback_query, exclude_ids=None):
        calls.append(exclude_ids)
        return orig(plans, fallback_query, exclude_ids=exclude_ids)

    agent._run_retrieval = _wrapped  # type: ignore[method-assign]
    return calls


def test_reflect_excludes_seen_chunk_ids_when_flag_set(monkeypatch):
    """kanun_kabul_oylama declares exclude_seen_chunks: true → each reflect hop passes
    the run's already-seen chunk ids to _run_retrieval so fetch_k fills with novel ones.
    The initial retrieval carries no exclusion; the hop carries the initial pool's ids."""
    reflect = _scripted([
        _reflection(done=False, anchors={"esas_no": "2/773"}, hop=1, next_draft="hop-1"),
        _reflection(done=True, hop=2, next_draft=None, reason="found"),
    ])
    # initial round returns c100/c101; the reflect hop returns fresh c200/c201.
    search = _ids_search([["c100", "c101"], ["c200", "c201"]])
    agent = _agent(monkeypatch, plan=_plan("kanun_kabul_oylama"), reflect=reflect, search=search)
    calls = _capture_run_retrieval(agent)

    agent.run("X kanunu kaç oyla kabul edildi", session_collections=[COL])

    assert calls[0] is None                          # initial retrieval — no exclusion
    assert calls[1] == {"c100", "c101"}              # hop-1 holds out the initial pool's ids


def test_reflect_no_chunk_exclusion_when_strategy_lacks_the_flag(monkeypatch):
    """kanun_rapor_bolumu is adaptive/non-comprehensive too but does NOT set
    exclude_seen_chunks — every _run_retrieval call must be unfiltered (no-op check)."""
    reflect = _scripted([
        _reflection(done=False, hop=1, next_draft="hop-1"),
        _reflection(done=True, hop=2, next_draft=None),
    ])
    search = _ids_search([["c100", "c101"], ["c200", "c201"]])
    agent = _agent(monkeypatch, plan=_plan("kanun_rapor_bolumu"), reflect=reflect, search=search)
    calls = _capture_run_retrieval(agent)

    agent.run("bir kanunun genel gerekçesi nedir", session_collections=[COL])

    assert calls                                     # retrieval ran
    assert all(c is None for c in calls)             # never excludes


def test_run_retrieval_drops_seen_ids_and_fills_from_remainder(monkeypatch):
    """The core mechanic: excluded ids are dropped AFTER ranking, so the budget fills
    with the next-best novel chunks instead of re-surfacing seen ones."""
    from src.agent.orchestrator import CollectionExecutionPlan

    # 6 candidates; budget 3. Without exclusion the top-3 would be c1/c2/c3.
    search = _ids_search([["c1", "c2", "c3", "c4", "c5", "c6"]])
    agent = _agent(monkeypatch, plan=_plan("kanun_kabul_oylama"), reflect=MagicMock(), search=search)
    plan = CollectionExecutionPlan(
        collection_name=COL, priority=1, retrieval_budget=3, fetch_k=6,
        filters={}, query_drafts=["q"], route_reason="test",
    )

    results = agent._run_retrieval([plan], "q", exclude_ids={"c1", "c2", "c3"})

    got = {c.chunk_id for c in results[COL].chunks}
    assert got == {"c4", "c5", "c6"}                 # seen dropped, budget filled from remainder
    assert results[COL].fetched_count == 6           # pre-exclusion candidate count preserved


def test_seen_chunk_ids_collects_across_collections(monkeypatch):
    from src.agent.schemas import Chunk, OrchestratorState, RetrievalOutput

    state = OrchestratorState(request_id="t", user_query="q")
    state.retrieval_results = {
        "col_a": RetrievalOutput(collection_name="col_a", fetched_count=2, returned_count=2,
            latency_ms=0.0, chunks=[
                Chunk(chunk_id="a1", document_id="d1", collection_name="col_a", doc_type="tutanak",
                      source_title="", text="", score=1.0, metadata={}),
                Chunk(chunk_id="a2", document_id="d2", collection_name="col_a", doc_type="tutanak",
                      source_title="", text="", score=1.0, metadata={}),
            ]),
        "col_b": RetrievalOutput(collection_name="col_b", fetched_count=1, returned_count=1,
            latency_ms=0.0, chunks=[
                Chunk(chunk_id="b1", document_id="d3", collection_name="col_b", doc_type="onerge",
                      source_title="", text="", score=1.0, metadata={}),
            ]),
    }
    agent = _agent(monkeypatch, plan=_plan("kanun_kabul_oylama"), reflect=MagicMock())
    assert agent._seen_chunk_ids(state) == {"a1", "a2", "b1"}
    assert agent._seen_chunk_ids(OrchestratorState(request_id="t2", user_query="q")) == set()


# --------------------------------------------------------------- planner tests

def _mock_client(response_dict: dict) -> MagicMock:
    client = MagicMock()
    client.chat.return_value = MagicMock(message=MagicMock(content=json.dumps(response_dict)))
    return client


def _planner() -> Planner:
    cfg = load_pipeline_config()
    return Planner(cfg, LLMClientPool.from_config(cfg))


def test_planner_reflect_parses_output():
    planner = _planner()
    response = {
        "done": False,
        "done_reason": None,
        "hop_cursor": 2,
        "extracted_anchors": {"esas_no": "2/773", "sira_sayisi": 5},
        "resources": [{"collection": COL, "query_drafts": [{"text": "5 sıra sayılı", "top_k": 8}]}],
        "reasoning": "hop-2",
    }
    planner._pool.get_client = MagicMock(return_value=_mock_client(response))

    out = planner.reflect("q", _plan("kanun_kabul_oylama"), procedure="P",
                          answer_directive="AD", aliases=[], extracted_anchors={}, hop_cursor=1)
    assert out is not None
    assert out.done is False
    assert out.hop_cursor == 2
    assert out.extracted_anchors == {"esas_no": "2/773", "sira_sayisi": 5}
    assert out.next_plan is not None
    assert out.next_plan.resources[0].query_drafts[0].text == "5 sıra sayılı"
    assert out.next_plan.query_type == "fact"            # reflect JSON has no query_type → default


def test_planner_reflect_done_with_no_resources_yields_none_next_plan():
    planner = _planner()
    response = {"done": True, "hop_cursor": 3, "extracted_anchors": {}, "resources": [], "reasoning": "r"}
    planner._pool.get_client = MagicMock(return_value=_mock_client(response))
    out = planner.reflect("q", _plan("kanun_kabul_oylama"), procedure="P")
    assert out.done is True
    assert out.next_plan is None


def test_planner_reflect_prompt_contains_procedure_directive_and_aliases():
    planner = _planner()
    client = _mock_client({"done": True, "resources": [], "reasoning": "r"})
    planner._pool.get_client = MagicMock(return_value=client)

    planner.reflect(
        "q", _plan("kanun_kabul_oylama"),
        procedure="HOP 1 — ÇIPA: kanun adını ara",
        answer_directive="İKİ OLASILIĞI AYIR",
        aliases=[{"term": "genel gerekçe", "official_phrase": "sıra sayısı raporu"}],
        extracted_anchors={}, hop_cursor=1,
    )
    sys_prompt = client.chat.call_args.kwargs["messages"][0]["content"]
    assert "HOP 1 — ÇIPA" in sys_prompt
    assert "İKİ OLASILIĞI AYIR" in sys_prompt
    assert "genel gerekçe" in sys_prompt and "sıra sayısı raporu" in sys_prompt


def test_planner_reflect_returns_none_on_llm_error():
    planner = _planner()
    client = MagicMock()
    client.chat.side_effect = RuntimeError("boom")
    planner._pool.get_client = MagicMock(return_value=client)
    out = planner.reflect("q", _plan("kanun_kabul_oylama"), procedure="P")
    assert out is None


# ---------------------------------------- metadata omurgası (Faz 4) — sira_sayisi filtresi

def test_reflect_injects_sira_sayisi_filter_from_anchor(monkeypatch):
    """Reflect bir sıra sayısı çıpası çıkardığında, sonraki hop'un retrieval'ı kesin
    ``{'sira_sayisi': {'$eq': N}}`` where-filtresi taşır (metadata omurgası → Hop-2 kanun
    adı geçmese bile kimliksiz roll-call bölgesine ulaşır) ve trace bunu işaretler."""
    seen_filters = []

    def _capturing_search(collection_key, query_text, filters=None, top_k=5, apply_reranker=True):
        seen_filters.append(filters)
        n = len(seen_filters)
        ids = [f"c{n * 100 + i}" for i in range(2)]
        return {
            "documents": [f"body-{cid}" for cid in ids],
            "metadatas": [
                {"chunk_id": cid, "document_id": f"d{cid}", "doc_type": "tutanak",
                 "source_title": f"t-{cid}", "_source_collection": collection_key}
                for cid in ids
            ],
            "distances": [0.1, 0.1],
        }

    reflect = _scripted([
        # ilk reflection sıra sayısını çözer → SONRAKİ hop retrieval'ı filtreli olmalı
        _reflection(done=False, anchors={"sira_sayisi": 5, "esas_no": "2/773"},
                    hop=1, next_draft="hop-1"),
        _reflection(done=True, hop=2, next_draft=None, reason="found"),
    ])
    agent = _agent(monkeypatch, plan=_plan("kanun_kabul_oylama"), reflect=reflect,
                   search=_capturing_search)
    out = agent.run("Kalkınma Bankası kaç oyla kabul edildi", session_collections=[COL])

    # Çıpa çözüldükten sonraki retrieval kesin sıra sayısı filtresini taşır.
    assert {"sira_sayisi": {"$eq": 5}} in seen_filters
    # Trace bayrağı: filtreli reflect turunda True.
    def _phase(ev):
        return ev.get("phase") if isinstance(ev, dict) else getattr(ev, "phase", None)

    def _details(ev):
        d = ev.get("details") if isinstance(ev, dict) else getattr(ev, "details", None)
        return d or {}
    reflect_flags = [
        _details(ev).get("sira_sayisi_filter_available")
        for ev in out.trace if _phase(ev) == "reflect"
    ]
    assert True in reflect_flags


def test_reflect_no_sira_filter_without_anchor(monkeypatch):
    """Sıra sayısı çıpası yoksa filtre enjekte EDİLMEZ (retrieval filtresiz semantik)."""
    seen_filters = []

    def _capturing_search(collection_key, query_text, filters=None, top_k=5, apply_reranker=True):
        seen_filters.append(filters)
        n = len(seen_filters)
        ids = [f"c{n * 100 + i}" for i in range(2)]
        return {
            "documents": [f"body-{cid}" for cid in ids],
            "metadatas": [
                {"chunk_id": cid, "document_id": f"d{cid}", "doc_type": "tutanak",
                 "source_title": f"t-{cid}", "_source_collection": collection_key}
                for cid in ids
            ],
            "distances": [0.1, 0.1],
        }

    reflect = _scripted([
        _reflection(done=False, anchors={"esas_no": "2/773"}, hop=1, next_draft="hop-1"),
        _reflection(done=True, hop=2, next_draft=None, reason="found"),
    ])
    agent = _agent(monkeypatch, plan=_plan("kanun_kabul_oylama"), reflect=reflect,
                   search=_capturing_search)
    agent.run("Kalkınma Bankası kaç oyla", session_collections=[COL])

    assert all(f != {"sira_sayisi": {"$eq": 5}} for f in seen_filters)
