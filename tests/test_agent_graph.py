"""LangGraph graf katmanının testleri: topoloji, router doğruluk tablosu,
recursion payı ve callback yayılımı.

Genişletme döngüsünün DAVRANIŞ regresyonları tests/test_orchestrator.py'de
(38 test, graf üzerinden koşuyor); buradaki testler graf-özel sözleşmeleri
sabitler.
"""
from __future__ import annotations

from unittest.mock import MagicMock

from langchain_core.callbacks import BaseCallbackHandler

from src.agent.graph import (
    GRAPH_NODES,
    build_routers,
    recursion_limit_for,
)
from src.agent.orchestrator import OrchestratorAgent, _RequeryResult
from src.agent.schemas import (
    CollectionSearchPlan,
    EvidenceDecision,
    OrchestratorState,
    SearchPlan,
    SearchQueryDraft,
)
from src.common.llm_client_pool import LLMClientPool
from src.config.pipeline_loader import load_pipeline_config


# ------------------------------------------------------------------- harness

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


def _agent(monkeypatch) -> OrchestratorAgent:
    cfg = load_pipeline_config()
    cfg.policy.enabled = False
    pool = LLMClientPool.from_config(cfg)
    agent = OrchestratorAgent(cfg, pool)
    agent._classifier = None
    agent._config.clarification.enabled = False
    agent._config.judge.llm.enabled = False
    agent._search_tool._reranker = None

    monkeypatch.setattr(
        agent._planner, "plan",
        lambda q, tracer=None, **kw: _make_plan("gazete_arsivi"),
    )
    # Judge "expand" derse çevrim gerçek broaden-LLM'ine düşmesin (offline).
    monkeypatch.setattr(
        agent._planner, "broaden",
        lambda *a, **kw: _make_plan("gazete_arsivi", draft_text="q-broaden"),
    )

    def _search(collection_key, query_text, filters=None, top_k=5, apply_reranker=True):
        return {
            "documents": ["body-1"],
            "metadatas": [{
                "chunk_id": "c1", "document_id": "d1", "doc_type": "gazete",
                "source_title": "t-1", "_source_collection": collection_key,
            }],
            "distances": [0.1],
        }
    monkeypatch.setattr(agent._search_tool, "search", _search)
    monkeypatch.setattr(
        agent._answer_tool, "generate",
        lambda query, context, mufettis_mode=False, chat_history=None,
        stream_callback=None, query_type=None, answer_directive=None: ("thinking", "Cevap metni."),
    )
    monkeypatch.setattr(agent._sanitizer, "validate", lambda *a, **kw: None)
    return agent


def _gs(agent, *, action="answer", comprehensive=False, rounds=0, added=None,
        escalated=False, loop_stop=False, depth=None, fetch_k_max=None):
    """Router doğruluk tablosu için sentetik GraphState."""
    state = OrchestratorState(request_id="t", user_query="soru")
    state.evidence_decision = EvidenceDecision(
        sufficient=(action == "answer"), confidence=0.9,
        action=action, missing_aspects=[], judge_type="heuristic",
    )
    last_result = None
    if added is not None:
        last_result = _RequeryResult(
            added=added, draft_texts={}, term_hypothesis=None,
            plan=_make_plan("gazete_arsivi"), pruned=0,
        )
    return {
        "s": state,
        "comprehensive": comprehensive,
        "rounds": rounds,
        "last_result": last_result,
        "escalated": escalated,
        "loop_stop": loop_stop,
        "depth": depth,
        "fetch_k_max": fetch_k_max,
    }


# ------------------------------------------------------------------ topoloji

def test_graph_nodes_match_canonical_set(monkeypatch):
    agent = _agent(monkeypatch)
    drawable = agent._graph.get_graph()
    nodes = set(drawable.nodes) - {"__start__", "__end__"}
    assert nodes == set(GRAPH_NODES)


def test_graph_cycle_edges_exist(monkeypatch):
    """judge→expansion→judge_post_expand→expansion çevrimi ve final çıkışlar."""
    agent = _agent(monkeypatch)
    edges = {(e.source, e.target) for e in agent._graph.get_graph().edges}
    for pair in [
        ("judge", "expansion"), ("judge", "refuse"), ("judge", "answering"),
        ("expansion", "judge_post_expand"), ("expansion", "answering"),
        ("judge_post_expand", "expansion"), ("judge_post_expand", "answering"),
        ("answering", "validation"), ("validation", "citation"),
    ]:
        assert pair in edges, f"eksik kenar: {pair}"


# ------------------------------------------------- router doğruluk tablosu

def test_route_after_judge_truth_table(monkeypatch):
    agent = _agent(monkeypatch)
    # Pin the judge round knobs so this LOGIC test is independent of pipeline.yaml's
    # POC neutralization (which sets both to 0 to force single-shot retrieval).
    agent._config.judge.comprehensive_max_rounds = 4
    agent._config.judge.max_expand_iterations = 1
    route = build_routers(agent)["judge"]
    max_it = agent._config.judge.max_expand_iterations

    assert route(_gs(agent, action="answer")) == "answering"
    assert route(_gs(agent, action="clarify")) == "refuse"
    assert route(_gs(agent, action="refuse")) == "refuse"
    assert route(_gs(agent, action="expand", rounds=0)) == "expansion"
    # Tur tavanı: expand istese de rounds >= max → final rota
    assert route(_gs(agent, action="expand", rounds=max_it)) == "answering"
    # Comprehensive, judge answer dese bile toplamaya devam eder
    assert route(_gs(agent, action="answer", comprehensive=True, rounds=0)) == "expansion"


def test_route_after_expansion_ceiling_skips_judge(monkeypatch):
    agent = _agent(monkeypatch)
    route = build_routers(agent)["expansion"]
    assert route(_gs(agent, action="answer", loop_stop=True)) == "answering"
    assert route(_gs(agent, action="clarify", loop_stop=True)) == "refuse"
    assert route(_gs(agent, action="expand", loop_stop=False)) == "judge_post_expand"


def test_route_after_judge_post_expand_truth_table(monkeypatch):
    agent = _agent(monkeypatch)
    # Pin round knobs (independent of pipeline.yaml's POC neutralization, see above).
    agent._config.judge.comprehensive_max_rounds = 4
    agent._config.judge.max_expand_iterations = 1
    route = build_routers(agent)["judge_post_expand"]
    comp_max = agent._config.judge.comprehensive_max_rounds

    # Doygunluk: kuru tur + tırmanış yok → final
    assert route(_gs(agent, action="answer", added=0, rounds=1)) == "answering"
    assert route(_gs(agent, action="clarify", added=0, rounds=1)) == "refuse"
    # Kuru tur ama derinlik tırmanışı yapıldı → çevrime devam
    assert route(_gs(agent, action="answer", added=0, escalated=True,
                     comprehensive=True, rounds=1)) == "expansion"
    # Verimli tur + judge expand istiyor + tavan altında → devam
    # (pipeline.yaml default'u max_expand_iterations=1 → açıkça yükselt)
    agent._config.judge.max_expand_iterations = 3
    assert route(_gs(agent, action="expand", added=3, rounds=1)) == "expansion"
    # Verimli tur ama tavan doldu → final
    assert route(_gs(agent, action="expand", added=3, comprehensive=True,
                     rounds=comp_max)) == "answering"
    # Verimli tur, judge doydu, comprehensive değil → final
    assert route(_gs(agent, action="answer", added=3, rounds=1)) == "answering"


def _adaptive_gs(agent, *, strategy="kanun_kabul_oylama", action="answer",
                 rounds=0, reflect_done=False, loop_stop=False):
    """Sentetik GraphState — planner_output.strategy adaptive stratejiye kurulu."""
    gs = _gs(agent, action=action, rounds=rounds, loop_stop=loop_stop)
    plan = _make_plan("gazete_arsivi")
    plan.strategy = strategy
    gs["s"].planner_output = plan
    gs["reflect_done"] = reflect_done
    return gs


def test_reflect_self_loop_edges_exist(monkeypatch):
    agent = _agent(monkeypatch)
    edges = {(e.source, e.target) for e in agent._graph.get_graph().edges}
    for pair in [
        ("judge", "reflect"), ("reflect", "reflect"),
        ("reflect", "answering"), ("reflect", "refuse"),
    ]:
        assert pair in edges, f"eksik reflect kenarı: {pair}"


def test_route_after_judge_routes_adaptive_to_reflect(monkeypatch):
    agent = _agent(monkeypatch)
    route = build_routers(agent)["judge"]
    # Adaptive strateji: judge "answer" dese bile prosedür için reflect'e gider.
    assert route(_adaptive_gs(agent, action="answer", rounds=0)) == "reflect"
    # clarify/refuse adaptive'i de kısa-devre yapar.
    assert route(_adaptive_gs(agent, action="clarify")) == "refuse"
    assert route(_adaptive_gs(agent, action="refuse")) == "refuse"
    # strategy.max_rounds (kanun_kabul_oylama=4) tükenince reflect'e gitmez.
    strat = agent._config.get_strategy("kanun_kabul_oylama")
    assert route(_adaptive_gs(agent, action="answer", rounds=strat["max_rounds"])) == "answering"


def test_route_after_judge_kill_switch_suppresses_reflect(monkeypatch):
    agent = _agent(monkeypatch)
    agent._config.reflect.enabled = False
    route = build_routers(agent)["judge"]
    # Kill-switch kapalıyken adaptive sorgu reflect'e girmez (generic yola düşer).
    assert route(_adaptive_gs(agent, action="answer", rounds=0)) == "answering"


def test_route_after_reflect_truth_table(monkeypatch):
    agent = _agent(monkeypatch)
    route = build_routers(agent)["reflect"]
    strat_max = agent._config.get_strategy("kanun_kabul_oylama")["max_rounds"]
    # done → answering (clarify'a saygı → refuse).
    assert route(_adaptive_gs(agent, reflect_done=True, action="answer")) == "answering"
    assert route(_adaptive_gs(agent, reflect_done=True, action="clarify")) == "refuse"
    # ceiling (loop_stop) → answering.
    assert route(_adaptive_gs(agent, loop_stop=True, action="answer")) == "answering"
    # done değil + tur kaldı → reflect (self-loop).
    assert route(_adaptive_gs(agent, reflect_done=False, rounds=1)) == "reflect"
    # done değil ama max_rounds tükendi → answering.
    assert route(_adaptive_gs(agent, reflect_done=False, rounds=strat_max)) == "answering"


def test_routers_read_config_at_route_time(monkeypatch):
    """Construction sonrası config mutasyonu router kararına yansımalı."""
    agent = _agent(monkeypatch)
    route = build_routers(agent)["judge"]
    gs = _gs(agent, action="expand", rounds=2)
    agent._config.judge.max_expand_iterations = 1
    assert route(gs) == "answering"  # 2 >= 1 → tavan
    agent._config.judge.max_expand_iterations = 5
    assert route(gs) == "expansion"  # 2 < 5 → devam


# ------------------------------------------------------------ recursion payı

def test_recursion_limit_formula(monkeypatch):
    agent = _agent(monkeypatch)
    cfg = agent._config
    # Adaptive reflect turları strategy.max_rounds'tan (+ reflect.default_max_rounds)
    # gelir — judge knob'ları POC'ta 0 olsa bile paya dahil edilmeli.
    adaptive_max = max(
        (s.get("max_rounds") or 0) for s in cfg.strategy_playbook.by_name.values()
    )
    expected = 32 + 4 * max(
        cfg.judge.comprehensive_max_rounds, cfg.judge.max_expand_iterations,
        adaptive_max, cfg.reflect.default_max_rounds,
    )
    assert recursion_limit_for(cfg) == expected
    # Doğrusal yol (~14 node) + reflect self-loop turları için gerçekten yeterli pay:
    assert expected >= 14 + 2 * cfg.reflect.default_max_rounds


# --------------------------------------------------------- callback yayılımı

class _CollectingHandler(BaseCallbackHandler):
    def __init__(self):
        self.chain_names: list[str] = []

    def on_chain_start(self, serialized, inputs, **kwargs):
        name = kwargs.get("name") or (serialized or {}).get("name") or ""
        self.chain_names.append(name)


def test_callbacks_propagate_to_graph_nodes(monkeypatch):
    """run(callbacks=[handler]) → LangGraph node'ları handler'a chain olayı yayar
    ve cevap üretimi (stream dahil) bozulmaz."""
    agent = _agent(monkeypatch)
    collector = _CollectingHandler()
    tokens: list[dict] = []

    out = agent.run(
        "1990 yılında neler oldu?",
        [],
        stream_callback=tokens.append,
        callbacks=[collector],
    )

    assert out.answer == "Cevap metni."
    # Node adları (faz adları) chain olayları olarak göründü
    assert "planning" in collector.chain_names
    assert "retrieval" in collector.chain_names
    assert "answering" in collector.chain_names


def test_conversational_passes_explicit_think(monkeypatch):
    """Regresyon: langchain-ollama, reasoning truthy değilken sunucu thinking'ini
    istemci tarafında düşürür — conversational yol think'i açıkça geçmeli
    (None'a bırakılırsa thinking-default modellerde trace paneli boş kalır)."""
    agent = _agent(monkeypatch)
    client = MagicMock()
    client.chat.return_value = iter([])
    agent._pool.get_client = MagicMock(return_value=client)

    from src.agent.tracer import PipelineTracer

    out = agent._conversational_output("merhaba", None, PipelineTracer(), None)
    assert out.scope == "conversational"
    kwargs = client.chat.call_args.kwargs
    assert kwargs.get("think") is not None  # açık True/False, asla None değil


def test_run_without_callbacks_still_works(monkeypatch):
    agent = _agent(monkeypatch)
    out = agent.run("1990 yılında neler oldu?", [])
    assert out.answer == "Cevap metni."
    phases = [e.phase for e in out.trace]
    assert {"planning", "policy", "budget", "retrieval", "assembly",
            "judge", "answering", "validation", "citation"} <= set(phases)
