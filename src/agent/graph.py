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
    "reflect",
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
    reflect_done: bool  # adaptive reflect self-loop'unun DURMA sinyali → answering'e çık
    # Ard ardışık VERİMSİZ reflect hop sayacı (added==0 VE window_added==0). retrieval
    # node'unun hop-dalı günceller (verimli hop → 0 sıfırlar); route_after_assembly okur:
    # eşiğe (reflect.max_dry_hops) ulaşınca çözülemeyen döngü iptal edilir → answer.
    dry_hops: int
    # Reflect KARAR node'unun ürettiği bir sonraki hop planı — retrieval node'u hop-mode'a
    # geçer. None = round-0 / terminal. Her reflect dönüşü bu anahtarı set eder (bir önceki
    # hop'un stash'i üzerine yazılsın). {"plan": SearchPlan, "exclude_seen": bool, "window_expand": bool}.
    pending_hop: Optional[Any]
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
        "reflect_done": False,
        "pending_hop": None,
        "dry_hops": 0,
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

    def _adaptive_strategy(gs: GraphState):
        """Çözülen adaptive strateji dict'ini (SALT-OKUNUR) döndür, yoksa None.

        ``reflect.enabled`` ile kapılanır — kill-switch reflect yolunu gerçekten
        bastırsın (yoksa dead-flag olur). Paylaşılan dict ASLA mutasyona uğratılmaz.
        """
        if not agent._config.reflect.enabled:
            return None
        po = gs["s"].planner_output
        if not (po and po.strategy):
            return None
        s = agent._config.get_strategy(po.strategy)
        if s and s.get("mode") == "adaptive" and s.get("procedure"):
            return s
        return None

    def _is_adaptive(gs: GraphState) -> bool:
        return _adaptive_strategy(gs) is not None

    def _max_rounds(gs: GraphState) -> int:
        s = _adaptive_strategy(gs)
        if s is not None:
            # LOKAL fallback — paylaşılan dict'i mutasyona uğratma; n0'a çekilmiş
            # judge knob'larına DEĞİL, reflect.default_max_rounds'a düş.
            return s.get("max_rounds") or agent._config.reflect.default_max_rounds
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
        decision = gs["s"].evidence_decision
        # clarify/refuse her şeyden önce gelir (adaptive dahil).
        if decision and decision.action in ("clarify", "refuse"):
            return _final_route(decision)
        # Adaptive strateji → generic broaden yerine reflect self-loop'u; judge
        # aksiyonundan bağımsız (prosedür tek-tur "yeterli" yargısını tanımaz).
        if _is_adaptive(gs) and gs.get("rounds", 0) < _max_rounds(gs):
            return "reflect"
        # Eski `while rounds < max_rounds: need_more ... break` girişi.
        if _need_more(gs) and gs.get("rounds", 0) < _max_rounds(gs):
            return "expansion"
        return _final_route(decision)

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

    def route_after_retrieval(gs: GraphState) -> str:
        # Reflect hop retrieval'ı rabbit_holes'u ATLAR (hop'lar facet madenciliği yeniden
        # koşmaz) → doğrudan assembly. Round-0 (pending_hop yok) mevcut akış: rabbit_holes.
        return "assembly" if gs.get("pending_hop") else "rabbit_holes"

    def route_after_assembly(gs: GraphState) -> str:
        # Round-0 (pending_hop yok) → mevcut akış: judge.
        if not gs.get("pending_hop"):
            return "judge"
        # Reflect hop-sonrası durma noktası (retrieval çalıştıktan SONRA bilinen).
        result = gs.get("last_result")
        # drafts_exhausted_by_dedup: dedup bu turun TÜM taslaklarını eledi (result.draft_texts
        # boş) → bu turda hiçbir arama çalışmadı, devam etmenin anlamı yok. added==0 (gerçek ama
        # verimsiz arama) İLE karıştırma: orada draft_texts DOLU olur, döngü devam eder.
        if not (result and result.draft_texts):
            return _final_route(gs["s"].evidence_decision)
        # Çözülemeyen (dry) hop iptali: arama ÇALIŞTI ama net-yeni kanıt getirmedi
        # (added==0 VE window_added==0 → retrieval node'u dry_hops'u artırdı). Ard ardışık
        # verimsiz hop sayısı eşiğe ulaştıysa max_rounds tavanını beklemeden kes; drafts_exhausted
        # (hiç arama yok) ile tamamlayıcı ama ayrık durum. max_dry_hops=0 → kapalı (kill-switch).
        max_dry = agent._config.reflect.max_dry_hops
        if max_dry and gs.get("dry_hops", 0) >= max_dry:
            return _final_route(gs["s"].evidence_decision)
        # Tur kaldı → bir sonraki KARAR hop'u (reflect). Yoksa max_rounds tavanı → doğrudan
        # answering (clarify/refuse'a düşmez: çok-hop kanıtı eldeyken cevapla).
        if gs.get("rounds", 0) < _max_rounds(gs):
            return "reflect"
        return "answering"

    def route_after_reflect(gs: GraphState) -> str:
        # KARAR node'u: hop ürettiyse → gerçek retrieval node'u (hop-mode). Aksi hâlde
        # terminal: prosedür DURMA (done) / ceiling → answering (clarify/refuse'a saygı).
        if gs.get("pending_hop"):
            return "retrieval"
        if gs.get("reflect_done") or gs.get("loop_stop"):
            return _final_route(gs["s"].evidence_decision)
        return "answering"  # güvenlik (karar node'u daima pending_hop VEYA reflect_done set eder)

    return {
        "bad_words": route_after_bad_words,
        "classification": route_after_classification,
        "policy": route_after_policy,
        "budget": route_after_budget,
        "retrieval": route_after_retrieval,
        "assembly": route_after_assembly,
        "judge": route_after_judge,
        "expansion": route_after_expansion,
        "judge_post_expand": route_after_judge_post_expand,
        "reflect": route_after_reflect,
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
    g.add_node("reflect", agent._node_reflect)
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
    # retrieval iki mod: round-0 → rabbit_holes (mevcut akış); reflect hop → assembly (rabbit_holes atlanır).
    g.add_conditional_edges("retrieval", routers["retrieval"], ["rabbit_holes", "assembly"])
    g.add_edge("rabbit_holes", "assembly")
    # assembly: round-0 → judge; reflect hop-sonrası → reflect (bir sonraki karar) / answering / refuse.
    g.add_conditional_edges(
        "assembly", routers["assembly"], ["judge", "reflect", "answering", "refuse"]
    )
    g.add_conditional_edges(
        "judge", routers["judge"], ["reflect", "expansion", "refuse", "answering"]
    )
    g.add_conditional_edges(
        "expansion", routers["expansion"], ["judge_post_expand", "refuse", "answering"]
    )
    g.add_conditional_edges(
        "judge_post_expand",
        routers["judge_post_expand"],
        ["expansion", "refuse", "answering"],
    )
    # Adaptive multi-hop reflect KARAR node'u → gerçek retrieval node'u (hop-mode); çevrim
    # reflect → retrieval → assembly → reflect. DURMA'da (reflect_done/ceiling) answering'e/refuse'a çıkar.
    g.add_conditional_edges(
        "reflect", routers["reflect"], ["retrieval", "answering", "refuse"]
    )
    g.add_edge("answering", "validation")
    g.add_edge("validation", "citation")
    g.add_edge("citation", END)
    g.add_edge("refuse", END)

    return g.compile()


def recursion_limit_for(config: Any) -> int:
    """Doğrusal ~14 node + reflect hop başına 3 node (reflect→retrieval→assembly);
    >× pay bırakır (default 25 dar).

    Adaptive reflect çevrimi turları ``strategy.max_rounds``'tan gelir (judge
    knob'larından değil) — POC'ta judge knob'ları 0'a çekilse bile reflect döner,
    o yüzden en büyük strateji ``max_rounds``'unu + reflect fallback tavanını da
    hesaba kat. Katsayı 5: hop başına 3 node + pay (self-loop dönemi 4'tü).
    """
    adaptive_max = 0
    try:
        adaptive_max = max(
            (s.get("max_rounds") or 0)
            for s in config.strategy_playbook.by_name.values()
        )
    except (AttributeError, ValueError):
        adaptive_max = 0
    reflect_default = getattr(getattr(config, "reflect", None), "default_max_rounds", 0)
    max_rounds = max(
        config.judge.comprehensive_max_rounds,
        config.judge.max_expand_iterations,
        adaptive_max,
        reflect_default,
    )
    return 32 + 5 * max_rounds
