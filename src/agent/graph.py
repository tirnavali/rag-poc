"""Orkestratör pipeline'ının LangGraph StateGraph tanımı.

Her faz bir node'dur ve `OrchestratorAgent` üstündeki ince `_node_*`
metodlarına delege eder; faz isimleri (tracer sözleşmesi) değişmez.
Genişletme döngüsü `judge → expansion → judge_post_expand → expansion …`
çevrimidir; döngü sayaçları GraphState kanallarında, veri durumu ise tek
mutable `OrchestratorState` nesnesinde (`s` kanalı, referansla) yaşar.

Veri olmayan çalışma-anı nesneleri (tracer, stream_callback, sohbet
geçmişi) graf durumuna girmez: `RunContext` içinde
``config["configurable"]["run_ctx"]`` ile taşınır — derlenmiş graf böylece
durumsuz ve yeniden girilebilir kalır (tek derleme, eşzamanlı koşular).

Router'lar config'i route ANINDA okur (closure over agent): testlerin
construction sonrası ``judge.comprehensive_max_rounds`` benzeri mutasyonları
bu sayede çalışmaya devam eder.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Optional, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from src.agent.schemas import (
    AgentOutput,
    EvidenceDecision,
    OrchestratorState,
    ScopeResult,
    SearchPlan,
    ValidationResult,
)
from src.agent.tracer import PipelineTracer

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph

    from src.agent.orchestrator import OrchestratorAgent, _RequeryResult


# Kanonik node seti = tracer faz sözleşmesinin graf karşılığı.
# (off_domain "suggestion" fazını, conversational "answering" fazını içeriden
# yayar; "refuse" bugünkü gibi faz yaymaz.)
GRAPH_NODES = [
    "bad_words",
    "classification",
    "off_domain",
    "conversational",
    "planning",
    "policy",
    "budget",
    "retrieval",
    "rabbit_holes",
    "assembly",
    "judge",
    "expansion",
    "judge_post_expand",
    "refuse",
    "answering",
    "validation",
    "citation",
]


@dataclass
class RunContext:
    """Koşu başına, veri olmayan nesneler — graf durumu DEĞİL, config kanalı."""

    tracer: PipelineTracer
    query: str
    session_collections: list
    stream_callback: Optional[Callable] = None
    chat_history: Optional[list] = None


class GraphState(TypedDict, total=False):
    # Mevcut mutable pydantic durum nesnesi — node'lar yerinde mutasyona
    # uğratır, kanal değeri (referans) hiç değişmez.
    s: OrchestratorState
    scope_result: Optional[ScopeResult]
    # Terminal kanal: her END yolu bunu doldurur.
    output: Optional[AgentOutput]
    # policy/budget'ın refuse gerekçesi; judge yolunda evidence_decision'dan türetilir.
    refuse_reason: Optional[str]
    # --- genişletme döngüsü kontrol kanalları (bugünkü run() lokalleri) ---
    comprehensive: bool
    rounds: int
    depth: Optional[int]
    fetch_k_max: Optional[int]
    reuse_plan: Optional[SearchPlan]
    # Gerçek tipi orchestrator._RequeryResult — döngüsel import olmasın diye Any
    # (LangGraph, TypedDict ipuçlarını derleme anında get_type_hints ile çözer;
    # TYPE_CHECKING-altı forward referans burada NameError verir).
    last_result: Optional[Any]
    loop_stop: bool  # expansion node'unda ceiling'e takıldı → döngüden çık
    escalated: bool  # kuru turda derinlik ikiye katlandı → devam
    # --- cevap yolu taşıyıcıları (bugünkü run() lokalleri) ---
    thinking: str
    context: str
    validation: Optional[ValidationResult]


def initial_graph_state(state: OrchestratorState) -> GraphState:
    return {
        "s": state,
        "scope_result": None,
        "output": None,
        "refuse_reason": None,
        "comprehensive": False,
        "rounds": 0,
        "depth": None,
        "fetch_k_max": None,
        "reuse_plan": None,
        "last_result": None,
        "loop_stop": False,
        "escalated": False,
        "thinking": "",
        "context": "",
        "validation": None,
    }


def run_ctx(config: RunnableConfig) -> RunContext:
    return config["configurable"]["run_ctx"]


def _final_route(decision: Optional[EvidenceDecision]) -> str:
    """Döngü sonrası aksiyon kontrolü — orchestrator.py'nin eski :408-412'si."""
    action = decision.action if decision else "answer"
    if action in ("clarify", "refuse"):
        return "refuse"
    return "answering"


def build_routers(agent: "OrchestratorAgent") -> dict:
    """Koşullu kenar router'ları — agent üstüne closure (test edilebilir fabrika).

    Config route ANINDA okunur: testlerin construction sonrası
    ``judge.comprehensive_max_rounds`` benzeri mutasyonları etkili kalır.
    """

    def _max_rounds(gs: GraphState) -> int:
        if gs.get("comprehensive"):
            return agent._config.judge.comprehensive_max_rounds
        return agent._config.judge.max_expand_iterations

    def _need_more(gs: GraphState) -> bool:
        decision = gs["s"].evidence_decision
        return bool(gs.get("comprehensive") or (decision and decision.action == "expand"))

    # ------------------------------------------------------------- router'lar

    def route_after_bad_words(gs: GraphState) -> str:
        return END if gs.get("output") is not None else "classification"

    def route_after_classification(gs: GraphState) -> str:
        sr = gs.get("scope_result")
        if sr is None:  # classifier kapalı → doğrudan planlamaya
            return "planning"
        if (
            sr.scope == "off_domain"
            and sr.confidence >= agent._config.classifier.confidence_threshold
            and not agent._is_known_parliamentary_term(gs["s"].user_query)
        ):
            return "off_domain"
        if sr.scope == "conversational":
            return "conversational"
        return "planning"

    def route_after_policy(gs: GraphState) -> str:
        return "refuse" if gs.get("refuse_reason") else "budget"

    def route_after_budget(gs: GraphState) -> str:
        return "refuse" if gs.get("refuse_reason") else "retrieval"

    def route_after_judge(gs: GraphState) -> str:
        # Eski `while rounds < max_rounds: need_more ... break` girişi.
        if _need_more(gs) and gs.get("rounds", 0) < _max_rounds(gs):
            return "expansion"
        return _final_route(gs["s"].evidence_decision)

    def route_after_expansion(gs: GraphState) -> str:
        # Ceiling'e takılan tur judge_post_expand'i atlayıp döngüden çıkar
        # (eski :352-358'deki `break`).
        if gs.get("loop_stop"):
            return _final_route(gs["s"].evidence_decision)
        return "judge_post_expand"

    def route_after_judge_post_expand(gs: GraphState) -> str:
        result = gs.get("last_result")
        added = result.added if result else 0
        # Kuru tur + tırmanış yoksa: doygunluk (veya tek atımlık düz expand) → çık.
        if added == 0 and not gs.get("escalated"):
            return _final_route(gs["s"].evidence_decision)
        if _need_more(gs) and gs.get("rounds", 0) < _max_rounds(gs):
            return "expansion"
        return _final_route(gs["s"].evidence_decision)

    return {
        "bad_words": route_after_bad_words,
        "classification": route_after_classification,
        "policy": route_after_policy,
        "budget": route_after_budget,
        "judge": route_after_judge,
        "expansion": route_after_expansion,
        "judge_post_expand": route_after_judge_post_expand,
    }


def build_orchestrator_graph(agent: "OrchestratorAgent") -> "CompiledStateGraph":
    """Grafı derler; koşullu kenarlar build_routers(agent) fabrikasından gelir."""
    routers = build_routers(agent)

    g = StateGraph(GraphState)
    g.add_node("bad_words", agent._node_bad_words)
    g.add_node("classification", agent._node_classification)
    g.add_node("off_domain", agent._node_off_domain)
    g.add_node("conversational", agent._node_conversational)
    g.add_node("planning", agent._node_planning)
    g.add_node("policy", agent._node_policy)
    g.add_node("budget", agent._node_budget)
    g.add_node("retrieval", agent._node_retrieval)
    g.add_node("rabbit_holes", agent._node_rabbit_holes)
    g.add_node("assembly", agent._node_assembly)
    g.add_node("judge", agent._node_judge)
    g.add_node("expansion", agent._node_expansion)
    g.add_node("judge_post_expand", agent._node_judge_post_expand)
    g.add_node("refuse", agent._node_refuse)
    g.add_node("answering", agent._node_answering)
    g.add_node("validation", agent._node_validation)
    g.add_node("citation", agent._node_citation)

    g.add_edge(START, "bad_words")
    g.add_conditional_edges("bad_words", routers["bad_words"], ["classification", END])
    g.add_conditional_edges(
        "classification",
        routers["classification"],
        ["off_domain", "conversational", "planning"],
    )
    g.add_edge("off_domain", END)
    g.add_edge("conversational", END)
    g.add_edge("planning", "policy")
    g.add_conditional_edges("policy", routers["policy"], ["refuse", "budget"])
    g.add_conditional_edges("budget", routers["budget"], ["refuse", "retrieval"])
    g.add_edge("retrieval", "rabbit_holes")
    g.add_edge("rabbit_holes", "assembly")
    g.add_edge("assembly", "judge")
    g.add_conditional_edges(
        "judge", routers["judge"], ["expansion", "refuse", "answering"]
    )
    g.add_conditional_edges(
        "expansion", routers["expansion"], ["judge_post_expand", "refuse", "answering"]
    )
    g.add_conditional_edges(
        "judge_post_expand",
        routers["judge_post_expand"],
        ["expansion", "refuse", "answering"],
    )
    g.add_edge("answering", "validation")
    g.add_edge("validation", "citation")
    g.add_edge("citation", END)
    g.add_edge("refuse", END)

    return g.compile()


def recursion_limit_for(config: Any) -> int:
    """Doğrusal ~14 node + tur başına 2 node; >2× pay bırakır (default 25 dar)."""
    max_rounds = max(
        config.judge.comprehensive_max_rounds,
        config.judge.max_expand_iterations,
    )
    return 32 + 4 * max_rounds
