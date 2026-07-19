"""OrchestratorAgent — the single agent pipeline.

Unifies the former legacy/orchestrator split into one explicit state machine:

  bad_words? → intent (scope + tool/db) → probe + facets + clarification →
  planning → policy? → budget → retrieve → assemble → judge →
  (bounded re-query → re-assemble → re-judge) → answer → sanitize → cite

Stage-2 gates (`bad_words_filter`, `policy`) are toggled per-stage in
pipeline.yaml; when disabled the orchestrator supplies sensible fallbacks
(allowed = planner suggestions). The retrieval budget maps the planner's
query_type to a per-collection fetch_k — a plain config lookup, always on.
"""
from __future__ import annotations

import json
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import NamedTuple, Optional

from langchain_core.runnables import RunnableConfig

from src.agent.assembler import BalancedContextAssembler
from src.agent.bad_words_filter import BadWordsFilter
from src.agent.citations import CitationBuilder
from src.agent.clarifier import AmbiguityGate, FacetMiner, QueryRefiner
from src.agent.classifier import ScopeClassifier
from src.agent.judge import EvidenceJudge
from src.agent.planner import Planner
from src.agent.policy import PolicyEnforcer
from src.agent.sanitizer import SanitizerAgent
from src.agent.schemas import (
    AgentOutput,
    Chunk,
    CollectionExecutionPlan,
    ContextAssemblyItem,
    EvidenceDecision,
    OrchestratorState,
    PolicyResult,
    ReflectionOutput,
    RetrievalOutput,
    SearchPlan,
    TermHypothesis,
)
from src.agent.graph import (
    GraphState,
    RunContext,
    build_orchestrator_graph,
    initial_graph_state,
    recursion_limit_for,
    run_ctx,
)
from src.agent.suggester import Suggester
from src.agent.tools import AnswerTool, SearchTool
from src.agent.tracer import PipelineTracer
from src.common.chroma import where_year_filter
from src.common.filter_translators import build_chroma_where
from src.common.schemas import FilterCriteria
from src.common.llm_client_pool import LLMClientPool
from src.config.collections import COLLECTIONS, get_production_collection_keys
from src.config.document_types import FILTER_APPLICABILITY
from src.config.pipeline_loader import PipelineConfig
from src.config.settings import COMPREHENSIVE_KEYWORDS, PARLIAMENTARY_TERM_SYNONYMS
from src.generator.prompts import CONVERSATIONAL_SYS_PROMPT


# TEMPORARY — tek koleksiyonlu RAG testi için BalancedContextAssembler devre dışı.
# Geri açmak için: bu satırı True yap (ya da bu blok + _run_assembler/_assemble_passthrough'u
# silip her iki çağrı noktasını `self._assembler.run(state)`'e geri döndür).
_ASSEMBLER_ENABLED = False


_REFUSE_MESSAGES = {
    "no_allowed_collections": (
        "Seçili koleksiyonlarda bu konu için arama yapılamaz. "
        "Başlangıçta farklı koleksiyonlar seçin."
    ),
    "judge_refuse": "Yetkili kaynaklarla yanıt veremiyorum.",
    "clarify": "Sorunuzu netleştirir misiniz? Yeterli kanıt bulunamadı.",
}


class _RequeryResult(NamedTuple):
    """Outcome of one _requery_expand() / _enumerate_expand() round."""
    added: int
    draft_texts: dict[str, list[str]]
    term_hypothesis: Optional[TermHypothesis]
    plan: Optional[SearchPlan] = None
    pruned: int = 0
    excluded_seen_count: int = 0  # already-seen chunk ids held out of this hop's retrieval
    sira_filter: Optional[dict] = None  # positive {sira_sayisi: {$eq: N}} applied this hop, if any
    window_added: int = 0  # neighbor chunks spliced by window-expand this hop, if any


_CHUNK_ORDER_RE = re.compile(r"^(.+)_(\d+)$")  # span_resolver.py:46 pattern; {doc}_{index}


def _parse_chunk_order(chunk_id: str) -> Optional[tuple[str, int]]:
    """(document_prefix, reading-order index) from a chunk_id, or None if malformed.

    Chunk ids are ``{document_id}_{i}`` (pipeline.py) with ``i`` the 0-based reading
    order — greedy ``(.+)`` keeps underscore-bearing document ids intact. Field-free:
    the order lives in the id, so this works regardless of the chunk_index backfill.
    """
    m = _CHUNK_ORDER_RE.match(chunk_id or "")
    return (m.group(1), int(m.group(2))) if m else None


def _collect_first_filters(plan: SearchPlan) -> dict[str, dict]:
    """Per-collection Chroma where-filters from each resource's first draft.

    Filters are translated to ChromaDB `where` syntax here (the orchestrator
    path's single choke point), so SearchTool receives the same already-
    translated dict as the rest of the retrieval path expects.
    A raw model_dump (e.g. {"year_lte": 2000}) is NOT a valid Chroma filter:
    `year_lte`/`year_gte` are not real metadata fields and multi-field dicts
    need a `$and` wrapper — build_chroma_where handles both.
    """
    out: dict[str, dict] = {}
    for resource in plan.resources:
        if not resource.query_drafts:
            continue
        first = resource.query_drafts[0]
        if first.filters is None:
            continue
        # build_chroma_where resolves `author` to the collection's actual
        # labels ($in) per-collection, like the legacy _execute_single path.
        where = build_chroma_where(first.filters, resource.collection)
        if where:
            out[resource.collection] = where
    return out


def _collect_draft_texts(plan: SearchPlan) -> dict[str, list[str]]:
    """Per-collection planner query rewrites, in draft order, deduplicated.

    These are the alternative phrasings the planner generated for a collection.
    The orchestrator runs each as a parallel search and RRF-fuses the ranked
    lists, so the planner's query expansion actually contributes recall instead
    of being discarded. Blank drafts are dropped; a collection with no usable
    drafts is omitted (retrieval falls back to the raw query).
    """
    out: dict[str, list[str]] = {}
    for resource in plan.resources:
        seen: set[str] = set()
        texts: list[str] = []
        for draft in resource.query_drafts:
            text = (draft.text or "").strip()
            if text and text not in seen:
                seen.add(text)
                texts.append(text)
        if texts:
            out[resource.collection] = texts
    return out


class OrchestratorAgent:
    """The single agentic RAG pipeline (see module docstring for the stage order)."""

    def __init__(self, config: PipelineConfig, client_pool: LLMClientPool, filter_extractor=None) -> None:
        self._config = config
        self._pool = client_pool
        self._planner = Planner(config, client_pool, filter_extractor)
        self._policy = PolicyEnforcer(config.policy)
        self._search_tool = SearchTool(config, client_pool)
        self._assembler = BalancedContextAssembler(config.retrieval_budget)
        self._judge = EvidenceJudge(config.judge, client_pool)
        self._answer_tool = AnswerTool(client_pool, config)
        self._sanitizer = SanitizerAgent(client_pool, config)
        # Gates / clarification
        self._bad_words = (
            BadWordsFilter(config.bad_words_filter)
            if getattr(config, "bad_words_filter", None) and config.bad_words_filter.enabled
            else None
        )
        self._classifier = (
            ScopeClassifier(client_pool, config)
            if getattr(config, "classifier", None) and config.classifier.enabled
            else None
        )
        self._suggester = Suggester(client_pool, config)
        self._facet_miner = FacetMiner()
        self._gate = AmbiguityGate(config.clarification)
        self._refiner = QueryRefiner(client_pool, config)
        # The live retrieval universe — agent only ever sees production collections.
        self._production = get_production_collection_keys()
        # Graf bir kez derlenir (checkpointer yok); koşu başına her şey
        # GraphState kanalları + RunContext ile taşınır → yeniden girilebilir.
        self._graph = build_orchestrator_graph(self)

    # ====================================================================== run

    def run(
        self,
        query: str,
        session_collections: Optional[list[str]] = None,
        stream_callback: Optional[callable] = None,
        clarification_callback: Optional[callable] = None,
        deep_mode: bool = False,
        on_phase: Optional[callable] = None,
        on_phase_end: Optional[callable] = None,
        chat_history: Optional[list] = None,
        callbacks: Optional[list] = None,
    ) -> AgentOutput:
        """Pipeline'ı LangGraph grafı üzerinden çalıştırır.

        ``callbacks``: opsiyonel LangChain callback listesi (ör. Langfuse
        CallbackHandler) — graf config'iyle her node'a ve node içindeki her
        ChatOllama çağrısına yayılır. ``clarification_callback`` ve
        ``deep_mode`` köhnedir (kabul edilir, yok sayılır — flow_diagram.html).
        """
        state = OrchestratorState(request_id=str(uuid.uuid4()), user_query=query)
        tracer = PipelineTracer(on_phase=on_phase, on_phase_end=on_phase_end)
        ctx = RunContext(
            tracer=tracer,
            query=query,
            session_collections=session_collections or [],
            stream_callback=stream_callback,
            chat_history=chat_history,
        )
        result = self._graph.invoke(
            initial_graph_state(state),
            config={
                "configurable": {"run_ctx": ctx},
                "callbacks": callbacks or None,
                "recursion_limit": recursion_limit_for(self._config),
            },
        )
        return result["output"]

    # ================================================================= node'lar
    # Her node, eski run()'daki aşamanın birebir taşınmış gövdesidir; tracer faz
    # isimleri ve details alanları değişmedi (web trace sekmesi + smoke testi
    # sözleşmesi). Bileşen kontrolleri (classifier/bad_words None, clarification
    # enabled) çalışma ANINDA yapılır — testler construction sonrası mutasyon yapar.

    def _node_bad_words(self, gs: GraphState, config: RunnableConfig) -> dict:
        """Stage 0: bad-words gate (stage-2; off by default)."""
        ctx = run_ctx(config)
        if self._bad_words is None:
            return {}
        bw = self._bad_words.check(ctx.query)
        with ctx.tracer.phase("bad_words_filter", details={"matched": bw.matched, "matched_terms": bw.matched_terms}):
            pass
        if bw.matched:
            return {"output": AgentOutput(
                answer=self._config.bad_words_filter.response_message,
                scope="bad_word",
                trace=ctx.tracer.events,
            )}
        return {}

    def _node_classification(self, gs: GraphState, config: RunnableConfig) -> dict:
        """Stage 1: intent (scope + tool/db selection)."""
        ctx = run_ctx(config)
        state = gs["s"]
        if self._classifier is None:
            return {"scope_result": None}
        scope_result = self._classifier.classify(ctx.query, ctx.tracer)
        # Validate intent's collection picks against the production universe.
        # (Erken çıkış yolları — off_domain/conversational — bu alanı okumaz;
        # yönlendirme kararını route_after_classification verir.)
        prod = set(self._production)
        state.selected_collections = [
            c for c in scope_result.selected_collections if c in COLLECTIONS and c in prod
        ]
        return {"scope_result": scope_result}

    def _node_off_domain(self, gs: GraphState, config: RunnableConfig) -> dict:
        ctx = run_ctx(config)
        return {"output": self._off_domain_output(ctx.query, ctx.tracer)}

    def _node_conversational(self, gs: GraphState, config: RunnableConfig) -> dict:
        ctx = run_ctx(config)
        return {"output": self._conversational_output(
            ctx.query, ctx.chat_history, ctx.tracer, ctx.stream_callback
        )}

    def _node_planning(self, gs: GraphState, config: RunnableConfig) -> dict:
        """Stage 2: planning (intent + constraints → diversified plan)."""
        ctx = run_ctx(config)
        state = gs["s"]
        query = ctx.query
        with ctx.tracer.phase("planning") as tctx:
            state.planner_output = self._planner.plan(
                query,
                ctx.tracer,
                # Restrict the planner's catalog + routing to the production universe.
                selected_collections=state.selected_collections or self._production,
                constraints=state.applied_constraints or None,
                max_variants=self._config.planner.normal_max_query_variants,
            )
            # Resolve the planner-selected strategy (research_strategies.md) into a
            # query_type + answer_directive override. An unknown/undefined strategy
            # name is a no-op — fail-open, Faz A behavior.
            if state.planner_output and state.planner_output.strategy:
                strategy = self._config.get_strategy(state.planner_output.strategy)
                if strategy:
                    if strategy.get("query_type"):
                        state.planner_output.query_type = strategy["query_type"]
                    state.answer_directive = strategy.get("answer_directive") or None
            # Deterministic enumeration/exhaustive override: a keyword match promotes
            # query_type to 'comprehensive' (the planner LLM may also emit it). This
            # drives larger context caps + the iterative gather loop downstream, and
            # always wins over the LLM's strategy pick (guards small-model misses).
            # EXCEPTION: an adaptive strategy owns its own multi-hop procedure — many
            # adaptive triggers literally contain comprehensive keywords (e.g.
            # kanun_gorusmeleri's "tüm görüşmeler"), so clobbering strategy→"enumerate"
            # here would silently kill the reflect path. Preserve adaptive strategies
            # (name AND their deliberate query_type) intact.
            if state.planner_output and self._is_comprehensive(query):
                current = (
                    self._config.get_strategy(state.planner_output.strategy)
                    if state.planner_output.strategy else None
                )
                if not (current and current.get("mode") == "adaptive"):
                    state.planner_output.query_type = "comprehensive"
                    state.planner_output.strategy = "enumerate"
                    enumerate_strategy = self._config.get_strategy("enumerate")
                    if enumerate_strategy:
                        state.answer_directive = enumerate_strategy.get("answer_directive") or None
            if tctx and state.planner_output:
                tctx.update_details(
                    intent=state.planner_output.intent,
                    query_type=state.planner_output.query_type,
                    strategy=state.planner_output.strategy,
                    comprehensive=(state.planner_output.query_type == "comprehensive"),
                    collections=[r.collection for r in state.planner_output.resources],
                    drafts={r.collection: [d.text for d in r.query_drafts]
                            for r in state.planner_output.resources},
                )
                if self._config.expose_thinking and state.planner_output.reasoning:
                    tctx.update_details(reasoning=state.planner_output.reasoning)
        return {}

    def _node_policy(self, gs: GraphState, config: RunnableConfig) -> dict:
        """Stage 2a: policy (stage-2; off → allow planner suggestions)."""
        ctx = run_ctx(config)
        state = gs["s"]
        with ctx.tracer.phase("policy") as tctx:
            if self._config.policy.enabled:
                self._policy.run(state, ctx.session_collections)
            else:
                suggested = [r.collection for r in state.planner_output.resources] if state.planner_output else []
                state.policy_result = PolicyResult(allowed_collections=suggested, denied_collections=[])
            if tctx and state.policy_result:
                tctx.update_details(
                    enabled=self._config.policy.enabled,
                    allowed=state.policy_result.allowed_collections,
                    denied=state.policy_result.denied_collections,
                )
        if not state.policy_result.allowed_collections:
            return {"refuse_reason": "no_allowed_collections"}
        return {}

    def _node_budget(self, gs: GraphState, config: RunnableConfig) -> dict:
        """Stage 2b: budget (query_type → per-collection fetch_k)."""
        ctx = run_ctx(config)
        state = gs["s"]
        with ctx.tracer.phase("budget") as tctx:
            state.collection_plans = self._flat_plans_for(
                state.planner_output, state.policy_result.allowed_collections
            )
            if tctx:
                tctx.update_details(
                    plans=[
                        {"collection": p.collection_name,
                         "retrieval_budget": p.retrieval_budget, "fetch_k": p.fetch_k}
                        for p in state.collection_plans
                    ],
                )
        if not state.collection_plans:
            return {"refuse_reason": "no_allowed_collections"}
        return {}

    def _node_retrieval(self, gs: GraphState, config: RunnableConfig) -> dict:
        """Stage 3: retrieval (parallel fan-out + RRF + rerank)."""
        ctx = run_ctx(config)
        state = gs["s"]
        with ctx.tracer.phase("retrieval") as tctx:
            fallback = self._fallback_query(state)
            # Register the initial searches in the tried-ledger so expansion
            # rounds never re-run an identical (query × filter × depth) search.
            state.collection_plans, _ = self._dedupe_and_register(state, state.collection_plans, fallback)
            state.retrieval_results = self._run_retrieval(state.collection_plans, fallback)
            if tctx:
                tctx.update_details(per_collection={
                    name: {"fetched": ro.fetched_count, "returned": ro.returned_count, "latency_ms": ro.latency_ms}
                    for name, ro in state.retrieval_results.items()
                })
        return {}

    def _node_rabbit_holes(self, gs: GraphState, config: RunnableConfig) -> dict:
        """Stage 3.5: facet mining + rabbit-hole suggestions.

        Reuses the retrieval we just ran (no extra probe pass): mine facets from
        the hits and, for broad/ambiguous queries, offer drill-down suggestions.
        """
        ctx = run_ctx(config)
        state = gs["s"]
        if self._config.clarification.enabled:
            self._suggest_rabbit_holes(state, ctx.tracer)
        return {}

    def _node_assembly(self, gs: GraphState, config: RunnableConfig) -> dict:
        """Stage 4: assembly."""
        ctx = run_ctx(config)
        state = gs["s"]
        with ctx.tracer.phase("assembly") as tctx:
            self._run_assembler(state)
            if tctx:
                tctx.update_details(
                    primary_count=len(state.assembled_chunks),
                    collection_coverage=len({c.collection_name for c in state.assembled_chunks}),
                )
        return {}

    def _node_judge(self, gs: GraphState, config: RunnableConfig) -> dict:
        """Stage 5: judge + genişletme döngüsü parametrelerinin kurulumu."""
        ctx = run_ctx(config)
        state = gs["s"]
        with ctx.tracer.phase("judge") as tctx:
            self._judge.run(state)
            if tctx and state.evidence_decision:
                tctx.update_details(
                    judge_type=state.evidence_decision.judge_type,
                    action=state.evidence_decision.action,
                    confidence=state.evidence_decision.confidence,
                    missing_aspects=state.evidence_decision.missing_aspects,
                )
                if self._config.expose_thinking and state.evidence_decision.reasoning:
                    tctx.update_details(reasoning=state.evidence_decision.reasoning)

        # Genişletme döngüsü parametrelerinin kurulumu (eski while-öncesi lokaller).
        # Comprehensive queries gather until saturation (a round adds no new chunks)
        # or a hard ceiling (max rounds / max chunks). Other query types keep the
        # original behavior: expand only when the judge asks, bounded by
        # max_expand_iterations. Döngünün kendisi graf çevrimidir:
        # judge → expansion → judge_post_expand → expansion …
        comprehensive = bool(state.planner_output and state.planner_output.query_type == "comprehensive")
        # Comprehensive gather starts at the query_type depth and escalates it on a
        # dry round (strategy 1); non-comprehensive expansion keeps a single depth.
        depth: Optional[int] = None
        fetch_k_max: Optional[int] = None
        if comprehensive:
            depth = self._config.retrieval_budget.budget_for("comprehensive").fetch_k
            fetch_k_max = self._config.retrieval_budget.enumerate_fetch_k_max
        return {
            "comprehensive": comprehensive,
            "depth": depth,
            "fetch_k_max": fetch_k_max,
            "rounds": 0,
            "reuse_plan": None,
            "loop_stop": False,
            "escalated": False,
            "reflect_done": False,
        }

    def _ceiling_for(self, gs: GraphState) -> int:
        """Toplam chunk tavanı — route/node anında config'den okunur."""
        if gs.get("comprehensive"):
            return self._config.retrieval_budget.max_total_for("comprehensive")
        state = gs["s"]
        return self._config.retrieval_budget.max_total_for(
            state.planner_output.query_type if state.planner_output else "fact"
        )

    def _node_expansion(self, gs: GraphState, config: RunnableConfig) -> dict:
        """Stage 5.1 (çevrim gövdesi 1/2): bir genişletme turu."""
        ctx = run_ctx(config)
        state = gs["s"]
        rounds = gs["rounds"]
        depth = gs["depth"]
        # A dry round escalates depth and re-runs the SAME drafts deeper — the
        # previous round's plan is reused verbatim so no broaden-LLM call is paid
        # for a pure depth escalation.
        reuse_plan = gs["reuse_plan"]
        comprehensive = gs["comprehensive"]

        if len(state.assembled_chunks) >= self._ceiling_for(gs):
            stop_reason = "ceiling"
            with ctx.tracer.phase("expansion") as tctx:
                if tctx:
                    tctx.update_details(round=rounds + 1, skipped=True, stop_reason=stop_reason,
                                        assembled_total=len(state.assembled_chunks))
            return {"loop_stop": True}

        broaden_reused = reuse_plan is not None
        with ctx.tracer.phase("expansion") as tctx:
            if comprehensive:
                # Enumerate: facet-partitioned + depth-escalating gather so the
                # long tail (ranks beyond the top-K, under-covered years) actually
                # surfaces instead of stalling on the same neighborhood.
                result = self._enumerate_expand(state, ctx.tracer, fetch_k=depth, base_plan=reuse_plan)
            else:
                # Plain re-query: broaden the query, retrieve anew, merge unique.
                result = self._requery_expand(state, ctx.tracer)
            added, term_hypothesis = result.added, result.term_hypothesis
            self._run_assembler(state)
            if tctx:
                tctx.update_details(
                    round=rounds + 1, comprehensive=comprehensive, added=added,
                    depth=depth, expanded=state.expanded,
                    assembled_total=len(state.assembled_chunks),
                    drafts=result.draft_texts,
                    pruned_duplicates=result.pruned,
                    broaden_reused=broaden_reused,
                    term_hypothesis=term_hypothesis.model_dump() if term_hypothesis else None,
                )
        return {"last_result": result, "reuse_plan": None, "loop_stop": False}

    def _node_judge_post_expand(self, gs: GraphState, config: RunnableConfig) -> dict:
        """Stage 5.1 (çevrim gövdesi 2/2): tur sonu yeniden yargılama."""
        ctx = run_ctx(config)
        state = gs["s"]
        result = gs["last_result"]
        with ctx.tracer.phase("judge_post_expand") as tctx:
            self._judge.run(state)
            if tctx and state.evidence_decision:
                tctx.update_details(
                    judge_type=state.evidence_decision.judge_type,
                    action=state.evidence_decision.action,
                )
            # A term hypothesis that actually resolved insufficiency is a
            # candidate for the "Öğrenilen Terimler" human-review queue —
            # never auto-applied, just surfaced for approve/reject.
            term_hypothesis = result.term_hypothesis if result else None
            if term_hypothesis and result.added > 0 and state.evidence_decision.action == "answer":
                self._record_term_hypothesis(term_hypothesis, state)

        update: dict = {"rounds": gs["rounds"] + 1, "escalated": False}
        if result and result.added == 0:
            # A dry round exhausts the CURRENT depth's neighborhood, not the
            # corpus. Reach deeper (double fetch_k) before declaring saturation;
            # stop only once even the deepest fetch surfaces nothing new.
            depth = gs["depth"]
            fetch_k_max = gs["fetch_k_max"]
            if gs["comprehensive"] and depth is not None and depth < fetch_k_max:
                update.update({
                    "depth": min(depth * 2, fetch_k_max),
                    "reuse_plan": result.plan,
                    "escalated": True,
                })
        return update

    def _node_reflect(self, gs: GraphState, config: RunnableConfig) -> dict:
        """Stage 5.2: adaptive çok-hop reflect / re-plan (self-loop çevrim gövdesi).

        Yalnızca ``mode: adaptive`` stratejiler için girilir — yönlendirme
        (``_is_adaptive``) strateji modu + ``reflect.enabled`` üzerinden kapılar. Her tur
        bir prosedür hop'u yürütür: kanıttan varlık çıkar, DURMA'ya karar ver, değilse bir
        sonraki hop'u getir. Çok-katmanlı FAIL-OPEN: beklenmeyen hata / adaptive-olmayan
        strateji / reflect-LLM ``None`` → generic ``broaden()`` yoluna düşer ve çıkar.
        Durma: prosedür ``done`` sinyali, boş next_plan, tavan, veya ``strategy.max_rounds``
        (``route_after_reflect`` uygular). HER dal ``rounds``'u artırır; yalnızca "execute"
        dalı ``reflect_done=False`` döndürür (aksi hâlde recursion-limit'e kadar dönerdi).
        """
        ctx = run_ctx(config)
        state = gs["s"]
        rounds = gs["rounds"]
        try:
            strategy = (
                self._config.get_strategy(state.planner_output.strategy)
                if state.planner_output and state.planner_output.strategy else None
            )
            with ctx.tracer.phase("reflect") as tctx:
                # Fail-open A: adaptive değil / prosedür yok → tek broaden turu, çık.
                if not strategy or strategy.get("mode") != "adaptive" or not strategy.get("procedure"):
                    result = self._requery_expand(state, ctx.tracer)
                    self._run_assembler(state)
                    if tctx:
                        tctx.update_details(round=rounds + 1, adaptive=False,
                                            fell_back="not_adaptive", added=result.added)
                    return {"last_result": result, "rounds": rounds + 1, "reflect_done": True}

                # Hacim güvenlik tavanı — generic çevrimin query_type tavanı (_ceiling_for)
                # DEĞİL: reasoning/summary'de tek-atım ilk retrieval zaten o tavanı (15)
                # doldurur → reflect hiç hop atamadan durur. Reflect'in gerçek bağı
                # max_rounds + done; bu yalnızca bağlam taşmasını önleyen üst sınır.
                if len(state.assembled_chunks) >= self._config.reflect.max_total_chunks:
                    if tctx:
                        tctx.update_details(round=rounds + 1, skipped=True, stop_reason="ceiling",
                                            assembled_total=len(state.assembled_chunks))
                    return {"reflect_done": True, "loop_stop": True, "rounds": rounds + 1}

                evidence = self._compact_evidence(state)
                tried = self._tried_query_texts(state)
                reflection = self._planner.reflect(
                    ctx.query,
                    state.planner_output,
                    ctx.tracer,
                    procedure=strategy["procedure"],
                    answer_directive=strategy.get("answer_directive") or "",
                    anchor=strategy.get("anchor"),
                    target=strategy.get("target"),
                    aliases=strategy.get("aliases") or [],
                    extracted_anchors=state.extracted_anchors,
                    hop_cursor=state.hop_cursor,
                    evidence_summary=evidence,
                    tried_queries=tried,
                )

                # Fail-open B: reflect LLM başarısız → broaden turu, çık.
                if reflection is None:
                    result = self._requery_expand(state, ctx.tracer)
                    self._run_assembler(state)
                    if tctx:
                        tctx.update_details(round=rounds + 1, reflect_failed=True, added=result.added)
                    return {"last_result": result, "rounds": rounds + 1, "reflect_done": True}

                # Anchor'ları birleştir (yeni non-null kazanır) + hop imlecini ilerlet.
                merged = dict(state.extracted_anchors)
                merged.update({k: v for k, v in (reflection.extracted_anchors or {}).items() if v is not None})
                state.extracted_anchors = merged
                state.hop_cursor = reflection.hop_cursor or state.hop_cursor

                # DONE (ya da aranacak bir şey kalmadı) → eldekiyle cevapla.
                if reflection.done or reflection.next_plan is None or not reflection.next_plan.resources:
                    if tctx:
                        tctx.update_details(round=rounds + 1, hop=state.hop_cursor, done=True,
                                            done_reason=reflection.done_reason,
                                            anchors=state.extracted_anchors)
                    return {"reflect_done": True, "rounds": rounds + 1}

                # Sonraki hop'un retrieval'ını yürüt, sonra yeniden-birleştir (answering/tavan görsün).
                result = self._reflect_retrieve(
                    state, reflection.next_plan, ctx.tracer,
                    exclude_seen=bool(strategy.get("exclude_seen_chunks")),
                    window_expand=bool(strategy.get("window_expand")),
                )
                self._run_assembler(state)
                if tctx:
                    # Hop-2 metadata omurgası: sıra sayısı filtresi UYGULANDIYSA kesin
                    # where-araması (kanun adı geçmese bile roll-call'a ulaşır); yoksa
                    # (çıpa çözülmedi / koleksiyon taşımıyor) filtresiz semantiğe düşer.
                    sira_applied = result.sira_filter is not None
                    tctx.update_details(
                        round=rounds + 1, hop=state.hop_cursor, done=False, added=result.added,
                        anchors=state.extracted_anchors, drafts=result.draft_texts,
                        assembled_total=len(state.assembled_chunks),
                        hop2_semantic_fallback=not sira_applied,
                        sira_sayisi_filter_available=sira_applied,
                        sira_filter=result.sira_filter,
                        excluded_seen_chunks=result.excluded_seen_count,
                        window_added=result.window_added,
                    )
                return {"last_result": result, "rounds": rounds + 1, "reflect_done": False}
        except Exception as exc:  # fail-open: reflect asla cevap yolunu kırmasın
            state.errors.append(f"reflect_failed: {type(exc).__name__}")
            return {"reflect_done": True, "rounds": rounds + 1}

    def _node_refuse(self, gs: GraphState, config: RunnableConfig) -> dict:
        """Refuse/clarify çıkışı — gerekçe policy/budget kanalından ya da judge kararından."""
        ctx = run_ctx(config)
        state = gs["s"]
        reason = gs.get("refuse_reason")
        if not reason:
            action = state.evidence_decision.action if state.evidence_decision else "refuse"
            reason = "clarify" if action == "clarify" else "judge_refuse"
        return {"output": self._build_refuse_output(state, reason, ctx.tracer)}

    def _node_answering(self, gs: GraphState, config: RunnableConfig) -> dict:
        """Stage 6a: answer (token'lar stream_callback ile canlı akar)."""
        ctx = run_ctx(config)
        state = gs["s"]
        with ctx.tracer.phase("answering") as tctx:
            # Window-expanded belgeleri okuma sırasına diz (varsayılanda no-op) → çıpa +
            # komşuları bitişik; context ve citations aynı sıralı listeyi okur (hiza korunur).
            self._apply_reading_order(state)
            context = self._build_context(state)
            # Forward tokens to the UI as they arrive (real streaming) instead of
            # dumping the whole answer in one chunk after ~full generation latency.
            thinking, answer = self._answer_tool.generate(
                query=ctx.query, context=context, chat_history=ctx.chat_history,
                stream_callback=ctx.stream_callback,
                query_type=state.planner_output.query_type if state.planner_output else "fact",
                answer_directive=state.answer_directive,
            )
            state.final_answer = answer
            if tctx:
                tctx.update_details(answer_chars=len(answer), context_chars=len(context))
                if self._config.expose_thinking:
                    if thinking:
                        tctx.update_details(thinking=thinking)
                    tctx.update_details(answer_preview=answer[:600])
        return {"thinking": thinking, "context": context}

    def _node_validation(self, gs: GraphState, config: RunnableConfig) -> dict:
        """Stage 6b: sanitize (advisory — cevabı asla değiştirmez)."""
        ctx = run_ctx(config)
        state = gs["s"]
        with ctx.tracer.phase("validation") as tctx:
            validation = self._sanitizer.validate(
                query=ctx.query,
                answer=state.final_answer,
                sources=[c.metadata for c in state.assembled_chunks],
                context=gs["context"],
            )
            # Non-destructive: validation is an advisory quality signal only. The
            # validator returns just a decision JSON (passes/checks/issues) — it no
            # longer regenerates the answer, which used to dominate latency. We
            # surface passes/issues in the trace but never alter the answer.
            if tctx and validation:
                tctx.update_details(passes=getattr(validation, "passes", True))
                if self._config.expose_thinking:
                    issues = getattr(validation, "issues", None)
                    if issues:
                        tctx.update_details(issues=issues)
        return {"validation": validation}

    def _node_citation(self, gs: GraphState, config: RunnableConfig) -> dict:
        """Stage 6c: cite + nihai AgentOutput."""
        ctx = run_ctx(config)
        state = gs["s"]
        with ctx.tracer.phase("citation") as tctx:
            state.citations = CitationBuilder.build(state.assembled_chunks)
            if tctx:
                tctx.update_details(citation_count=len(state.citations))

        return {"output": AgentOutput(
            answer=state.final_answer,
            thinking=gs["thinking"],
            plan=state.planner_output,
            validation=gs["validation"],
            trace=ctx.tracer.events,
            sources=state.citations,
            policy_result=state.policy_result,
            evidence_decision=state.evidence_decision,
            assembly=state.balanced_context,
            expanded=state.expanded,
            clarification=state.clarification,
            rabbit_holes=state.rabbit_holes,
        )}

    # ============================================================== clarify

    @staticmethod
    def _is_comprehensive(query: str) -> bool:
        """Enumeration/exhaustive intent — substring match against COMPREHENSIVE_KEYWORDS."""
        q = (query or "").lower()
        return any(kw in q for kw in COMPREHENSIVE_KEYWORDS)

    @staticmethod
    def _is_known_parliamentary_term(query: str) -> bool:
        """Deterministic scope-classifier safety net: a known jargon term (e.g.
        'kadük') present in the query forces scope back to in_scope even when the
        small classifier model mistakes a jargon-definition question for an
        off-domain dictionary lookup. Same glossary that drives retrieval-time
        synonym expansion (src.common.text.expand_parliamentary_synonyms)."""
        q = (query or "").lower()
        return any(term in q for term in PARLIAMENTARY_TERM_SYNONYMS)

    def _suggest_rabbit_holes(self, state, tracer) -> None:
        """Mine facets from the main retrieval results and, for broad/ambiguous
        queries, surface facet-grounded drill-down ("rabbit hole") suggestions.

        There is no separate probe pass: facets come from the retrieval we already
        ran in Stage 3. We never narrow the query — broad queries are answered
        broadly and the user is offered suggestions to dig deeper. The phase is
        emitted unconditionally so the debug trace shows which query produced which
        suggestions (and why none, when not ambiguous).
        """
        cfg = self._config.clarification
        with tracer.phase("rabbit_holes") as ctx:
            # Reshape retrieval hits into the {"metadatas": [...]} form FacetMiner expects.
            results = [
                {"metadatas": [c.metadata for c in ro.chunks]}
                for ro in state.retrieval_results.values()
            ]
            state.facets = self._facet_miner.mine(results)
            ambiguous = self._gate.is_ambiguous(state.facets, state.user_query)
            if ambiguous:
                count = getattr(cfg, "suggestion_count", 3)
                state.rabbit_holes = self._refiner.rabbit_holes(
                    state.user_query, state.facets, count
                )
            if ctx:
                ctx.update_details(
                    query=state.user_query,
                    hits=state.facets.total,
                    ambiguous=ambiguous,
                    facets_used={
                        "years": [f.value for f in state.facets.years[:2]],
                        "topics": [f.value for f in state.facets.topics[:2]],
                        "authors": [f.value for f in state.facets.authors[:1]],
                    },
                    suggestions=state.rabbit_holes,
                )

    def _off_domain_output(self, query: str, tracer: PipelineTracer) -> AgentOutput:
        suggestions = self._suggester.suggest(query, tracer)
        template = (
            self._config.off_domain_response_template
            or "Bu sistem alan dışında.\n1. {suggestion_0}\n2. {suggestion_1}\n3. {suggestion_2}"
        )
        padded = (suggestions + ["", "", ""])[:3]
        answer = template.format(suggestion_0=padded[0], suggestion_1=padded[1], suggestion_2=padded[2])
        return AgentOutput(
            answer=answer, scope="off_domain", suggestions=suggestions,
            rabbit_holes=suggestions, trace=tracer.events,
        )

    # ====================================================== allocation fallback

    def _run_assembler(self, state: OrchestratorState) -> None:
        """TEMPORARY yönlendirme — bkz. modül üstündeki _ASSEMBLER_ENABLED."""
        if _ASSEMBLER_ENABLED:
            self._assembler.run(state)
        else:
            self._assemble_passthrough(state)

    @staticmethod
    def _assemble_passthrough(state: OrchestratorState) -> None:
        """TEMPORARY: BalancedContextAssembler'ın yerine geçer — max_per_document/max_total
        UYGULANMAZ, her koleksiyonun retrieval sonucu öncelik sırasıyla doğrudan context'e
        akar (koleksiyon zaten kendi fetch_k/retrieval_budget'ıyla sınırlı). assembler.py'a
        dokunulmadı; gerçek BalancedContextAssembler hâlâ orada, yalnızca çağrılmıyor.
        """
        assembled: list[Chunk] = []
        items: list[ContextAssemblyItem] = []
        for plan in sorted(state.collection_plans, key=lambda p: p.priority):
            rr = state.retrieval_results.get(plan.collection_name)
            if not rr:
                continue
            for chunk in rr.chunks:
                assembled.append(chunk)
                items.append(ContextAssemblyItem(
                    chunk_id=chunk.chunk_id,
                    collection_name=chunk.collection_name,
                    document_id=chunk.document_id,
                    slot_type="primary",
                    assembly_reason="assembler_temporarily_disabled",
                    order_index=len(items),
                ))
        state.assembled_chunks = assembled
        state.balanced_context = items

    def _flat_plans_for(
        self,
        search_plan: Optional[SearchPlan],
        allowed: list[str],
        fetch_k: Optional[int] = None,
        override_filter: Optional[dict] = None,
    ) -> list[CollectionExecutionPlan]:
        """Build per-collection execution plans from the query_type retrieval budget.

        All fused hits flow into a single pool, capped downstream by the assembler's
        `max_total_primary` / per-query-type `max_total`.

        ``fetch_k`` overrides the query_type depth (used by enumerate depth
        escalation). ``override_filter``, when given, replaces the planner-derived
        per-collection filter for every collection (used by facet-partitioned
        enumerate to scope a pass to a single year).
        """
        filters_by = _collect_first_filters(search_plan) if search_plan else {}
        drafts_by = _collect_draft_texts(search_plan) if search_plan else {}
        qt = search_plan.query_type if search_plan else "fact"
        if fetch_k is None:
            fetch_k = self._config.retrieval_budget.budget_for(qt).fetch_k
        plans = []
        for idx, name in enumerate(allowed):
            plans.append(CollectionExecutionPlan(
                collection_name=name,
                priority=idx + 1,
                retrieval_budget=fetch_k,
                fetch_k=fetch_k,
                filters=(override_filter if override_filter is not None else filters_by.get(name, {})),
                query_drafts=drafts_by.get(name, []),
                route_reason="enumerate_facet" if override_filter is not None else "query_type_budget",
            ))
        return plans

    def _fallback_query(self, state: OrchestratorState) -> str:
        refined = state.planner_output.refined_query if state.planner_output else None
        return refined or state.user_query

    # ==================================================== tried-search ledger

    @staticmethod
    def _search_key(collection: str, query_text: str, filters: Optional[dict]) -> str:
        """Canonical ledger key for one executed search (filters order-insensitive)."""
        return json.dumps([collection, query_text, filters or {}], sort_keys=True, ensure_ascii=False)

    @staticmethod
    def _tried_query_texts(state: OrchestratorState, limit: int = 20) -> list[str]:
        """Unique query texts already searched, in first-tried order (most recent
        ``limit`` kept) — fed to broaden() so the LLM stops regenerating them."""
        texts: list[str] = []
        seen: set[str] = set()
        for key in state.tried_searches:
            try:
                _, text, _ = json.loads(key)
            except (ValueError, TypeError):
                continue
            if text not in seen:
                seen.add(text)
                texts.append(text)
        return texts[-limit:]

    def _dedupe_and_register(
        self,
        state: OrchestratorState,
        plans: list[CollectionExecutionPlan],
        fallback_query: str,
    ) -> tuple[list[CollectionExecutionPlan], int]:
        """Prune drafts already searched at this-or-greater depth; register the rest.

        ANN search is deterministic, so re-running an identical (collection ×
        query × filter) at the same or shallower fetch_k cannot surface anything
        new — it only burns retrieval latency (the observed repeat-search loop).
        A plan whose every draft is pruned is dropped entirely; the same draft
        becomes eligible again at a deeper fetch_k (depth escalation).

        Returns (kept_plans, pruned_draft_count). Kept plans have their effective
        drafts materialized (the raw-query fallback becomes explicit), so what is
        registered here is exactly what _run_retrieval will execute.
        """
        kept_plans: list[CollectionExecutionPlan] = []
        pruned = 0
        for plan in plans:
            if not plan.enabled:
                continue
            kept: list[str] = []
            for text in plan.query_drafts or [fallback_query]:
                key = self._search_key(plan.collection_name, text, plan.filters)
                if state.tried_searches.get(key, 0) >= plan.fetch_k:
                    pruned += 1
                    continue
                kept.append(text)
                state.tried_searches[key] = max(state.tried_searches.get(key, 0), plan.fetch_k)
            if not kept:
                continue
            plan.query_drafts = kept
            kept_plans.append(plan)
        return kept_plans, pruned

    # ============================================================ retrieval

    def _run_retrieval(self, plans: list[CollectionExecutionPlan], fallback_query: str,
                       exclude_ids: Optional[set[str]] = None) -> dict[str, RetrievalOutput]:
        """Fan out one search per (collection × draft), RRF-fuse per collection.

        Returns a fresh RetrievalOutput map; does not mutate state, so the same
        helper serves both the initial retrieval and bounded re-query expansion.

        ``exclude_ids`` (reflect-hop novelty pressure): chunk ids dropped from the
        ranked pool BEFORE the top-k truncation, so a hop's fetch_k fills with
        genuinely-new chunks instead of re-surfacing ones already in the pool.
        Applied post-rank (never a ChromaDB where filter — chunk id is not indexed
        metadata), so it can never blackout a whole session the way a metadata $nin can.
        """
        active = [p for p in plans if p.enabled]
        results: dict[str, RetrievalOutput] = {}
        if not active:
            return results

        tasks: list[tuple[int, str]] = []
        for pi, plan in enumerate(active):
            drafts = plan.query_drafts or [fallback_query]
            for draft in drafts:
                tasks.append((pi, draft))

        def _one(task):
            pi, query_text = task
            plan = active[pi]
            t0 = time.perf_counter()
            try:
                result = self._search_tool.search(
                    collection_key=plan.collection_name,
                    query_text=query_text,
                    filters=plan.filters or None,
                    top_k=plan.fetch_k,
                    apply_reranker=False,
                )
            except Exception as exc:
                return pi, exc, (time.perf_counter() - t0) * 1000
            chunks = self._dict_to_chunks(result, plan.collection_name)
            return pi, chunks, (time.perf_counter() - t0) * 1000

        draft_lists: dict[int, list[list[Chunk]]] = {pi: [] for pi in range(len(active))}
        latency_by_plan: dict[int, float] = {pi: 0.0 for pi in range(len(active))}

        with ThreadPoolExecutor(max_workers=min(16, max(1, len(tasks)))) as ex:
            for pi, payload, latency in ex.map(_one, tasks):
                latency_by_plan[pi] = max(latency_by_plan[pi], latency)
                if not isinstance(payload, Exception):
                    draft_lists[pi].append(payload)

        for pi, plan in enumerate(active):
            fused = self._fuse_draft_chunks(draft_lists[pi])
            # Rerank once here, on the deduped pool, instead of inside each
            # draft's search() call above — otherwise N drafts means N
            # independent cross-encoder passes (contending for the same
            # cached model instance) whose individual top-k gets thrown away
            # by fusion anyway.
            rerank_ms = 0.0
            if self._search_tool.reranker_enabled and fused:
                t_rr0 = time.perf_counter()
                scores = self._search_tool.rerank(
                    fallback_query, [(c.chunk_id, c.text) for c in fused], top_n=len(fused)
                )
                rerank_ms = (time.perf_counter() - t_rr0) * 1000
                for c in fused:
                    if c.chunk_id in scores:
                        c.rerank_score = scores[c.chunk_id]
                fused.sort(key=lambda c: scores.get(c.chunk_id, float("-inf")), reverse=True)
            # Drop already-seen chunk ids after ranking so top-k fills with novel
            # chunks (reflect-hop novelty; no-op when exclude_ids is None/empty).
            ranked = fused if not exclude_ids else [c for c in fused if c.chunk_id not in exclude_ids]
            chunks = ranked[: plan.retrieval_budget]
            results[plan.collection_name] = RetrievalOutput(
                collection_name=plan.collection_name,
                chunks=chunks,
                fetched_count=len(fused),
                returned_count=len(chunks),
                latency_ms=latency_by_plan[pi] + rerank_ms,
                filter_applied=plan.filters or {},
            )
        return results

    @staticmethod
    def _record_term_hypothesis(term_hypothesis: TermHypothesis, state: OrchestratorState) -> None:
        """Surface a re-query term guess that actually resolved insufficiency to
        the "Öğrenilen Terimler" human-review queue. Never auto-applied to live
        retrieval — a human must approve it (src.api.db.list_approved_term_synonyms)
        before it affects search. Best-effort: review-queue bookkeeping must never
        break the answer path.
        """
        try:
            from src.api import db as term_db
            term_db.upsert_term_candidate(
                term=term_hypothesis.term,
                hypothesis=term_hypothesis.official_phrase,
                source_query=state.user_query,
            )
        except Exception:
            pass

    def _requery_expand(self, state: OrchestratorState, tracer: PipelineTracer) -> "_RequeryResult":
        """Bounded re-query: broaden the plan, retrieve anew, merge unique chunks.

        Unlike reserve-promotion this issues fresh vector searches. New chunks are
        appended to the existing primary buffers and the matching execution plan's
        budget is raised so the re-assembly step can surface them.

        Returns a ``_RequeryResult``: ``added`` (0 signals saturation so the caller's
        gather loop can stop), ``draft_texts`` (what was actually searched, for
        trace visibility), and ``term_hypothesis`` (the broaden LLM's guess at an
        official-term synonym, if any — surfaced for the term_candidates review flow).
        """
        # Lazy import: keeps the agent layer decoupled from the web API's storage
        # module at import time (matches the existing lazy-reranker-import pattern
        # in SearchTool.__init__); only paid when a re-query round actually runs.
        from src.api import db as term_db

        missing_aspects = state.evidence_decision.missing_aspects if state.evidence_decision else None
        broader = self._planner.broaden(
            state.user_query,
            state.planner_output,
            tracer,
            selected_collections=state.selected_collections or self._production,
            result_count=len(state.assembled_chunks),
            missing_aspects=missing_aspects,
            rejected_hypotheses=term_db.list_rejected_term_hypotheses(),
            tried_queries=self._tried_query_texts(state),
        )
        if broader is None or not broader.resources:
            return _RequeryResult(added=0, draft_texts={}, term_hypothesis=None)

        # RE_RETRIEVAL_PROMPT's JSON schema has no query_type field, so _parse_plan
        # silently defaults broader.query_type to "fact" — which would starve every
        # re-query round to the "fact" fetch_k budget instead of the in-flight query's
        # (e.g. comprehensive=40 vs fact=15), causing premature saturation. Re-query
        # rounds broaden the QUERY, not the query_type — carry the original forward
        # deterministically rather than trusting the LLM to echo it back.
        broader.query_type = state.planner_output.query_type if state.planner_output else broader.query_type

        # A hallucinated collection key fails safely inside SearchTool (KeyError →
        # caught and dropped in _run_retrieval), so no catalog pre-filter is needed.
        allowed = [r.collection for r in broader.resources]
        requery_query = broader.refined_query or state.user_query
        requery_plans = self._flat_plans_for(broader, allowed)
        requery_plans, pruned = self._dedupe_and_register(state, requery_plans, requery_query)
        draft_texts = {p.collection_name: list(p.query_drafts) for p in requery_plans}
        new_results = self._run_retrieval(requery_plans, requery_query) if requery_plans else {}
        if not new_results:
            state.expand_iterations += 1
            return _RequeryResult(added=0, draft_texts=draft_texts,
                                  term_hypothesis=broader.term_hypothesis, plan=broader, pruned=pruned)

        added_total = self._merge_new_chunks(state, new_results, requery_plans)
        if added_total > 0:
            state.expanded = True
        state.expand_iterations += 1
        return _RequeryResult(added=added_total, draft_texts=draft_texts,
                              term_hypothesis=broader.term_hypothesis, plan=broader, pruned=pruned)

    def _merge_new_chunks(
        self,
        state: OrchestratorState,
        new_results: dict[str, RetrievalOutput],
        plans: list[CollectionExecutionPlan],
    ) -> int:
        """Merge freshly-retrieved chunks into the live pool, deduped by chunk_id.

        A new collection is added along with its plan; an existing collection gets
        only its unseen chunks appended and its retrieval_budget raised so the next
        re-assembly can surface them. Returns the count of genuinely new chunks.
        """
        existing_by = {p.collection_name: p for p in state.collection_plans}
        plan_by_name = {p.collection_name: p for p in plans}
        added_total = 0
        for name, rr_new in new_results.items():
            rr = state.retrieval_results.get(name)
            if rr is None:
                state.retrieval_results[name] = rr_new
                if name in plan_by_name:
                    state.collection_plans.append(plan_by_name[name])
                added_total += len(rr_new.chunks)
                continue
            seen = {c.chunk_id for c in rr.chunks}
            fresh = [c for c in rr_new.chunks if c.chunk_id not in seen]
            if fresh:
                rr.chunks = rr.chunks + fresh
                if name in existing_by:
                    existing_by[name].retrieval_budget += len(fresh)
                added_total += len(fresh)
        return added_total

    def _expand_windows(
        self,
        state: OrchestratorState,
        results: dict[str, RetrievalOutput],
        window_expand: bool,
    ) -> int:
        """Window-expand (pencere-genişletme): splice each top anchor's document-internal
        chunk-order neighbors into ``results`` so _merge_new_chunks folds them into the pool.

        The anchor is the highest rerank_score chunk in THIS hop's fresh ``results`` — the
        region the hop just found (e.g. a self-identifying vote announcement). Its
        neighbors ride in via a separate id-get path (fetch_neighbors), so they bypass
        exclude_ids (we deliberately want the full region even if some were already seen)
        but are chunk_id-deduped here (vs this hop) and again at merge (vs the pool).
        Gated by the strategy flag AND the global reflect.window_expand.enabled knob;
        no-op otherwise. Returns the count of neighbor chunks spliced.
        """
        cfg = self._config.reflect.window_expand
        if not (window_expand and cfg.enabled) or not results:
            return 0
        candidates = sorted(
            ((c.rerank_score, col, c) for col, rr in results.items() for c in rr.chunks),
            key=lambda t: t[0], reverse=True,
        )
        budget = cfg.max_neighbors_per_hop
        spliced = 0
        anchors_done = 0
        for _score, col, anchor in candidates:
            if anchors_done >= cfg.anchor_count or budget <= 0:
                break
            parsed = _parse_chunk_order(anchor.chunk_id)
            if parsed is None:
                continue  # mock/malformed id (e.g. "c100") — skip, no crash
            doc_prefix, idx = parsed
            key = f"{col}\x1f{doc_prefix}"
            if key in state.window_anchor_keys:
                continue  # this document's window already expanded this run — repeat guard
            anchors_done += 1
            state.window_anchor_keys.append(key)  # record even at 0 neighbors → no re-fetch
            nbr = self._search_tool.fetch_neighbors(
                col, doc_prefix, idx, cfg.neighbor_radius, max_total=budget,
            )
            nbr_chunks = self._dict_to_chunks(nbr, col)
            existing_ids = {c.chunk_id for c in results[col].chunks}
            nbr_chunks = [c for c in nbr_chunks if c.chunk_id not in existing_ids]
            if not nbr_chunks:
                continue
            # Stamp neighbors just below the anchor's rerank_score so a future rerank-ordered
            # assembler keeps them beside the anchor (inert under the current passthrough).
            for j, ch in enumerate(nbr_chunks, start=1):
                ch.rerank_score = anchor.rerank_score - j * 1e-6
            results[col].chunks.extend(nbr_chunks)
            spliced += len(nbr_chunks)
            budget -= len(nbr_chunks)
        return spliced

    def _facet_years(self, state: OrchestratorState, limit: int = 4) -> list[int]:
        """Most-frequent years present in the current retrieval, for facet-partitioned
        enumerate. Mines facets on demand when the clarification stage didn't run."""
        facets = state.facets
        if facets is None:
            results = [
                {"metadatas": [c.metadata for c in ro.chunks]}
                for ro in state.retrieval_results.values()
            ]
            facets = self._facet_miner.mine(results)
        years: list[int] = []
        for fv in facets.years[:limit]:
            token = str(fv.value)[:4]
            if token.isdigit():
                years.append(int(token))
        return years

    @staticmethod
    def _seen_chunk_ids(state: OrchestratorState) -> set[str]:
        """Every chunk id already surfaced by any retrieval this run (initial planner
        stage + every reflect round — retrieval_results accumulates via _merge_new_chunks
        and never resets). Fed to _run_retrieval(exclude_ids=...) so a reflect hop holds
        out the exact chunks already in the pool and fills its fetch_k with novel ones.
        Chunk ids are globally unique (``{document_id}_{index}``), so a single flat set
        across collections needs no per-collection scoping."""
        return {
            c.chunk_id
            for ro in state.retrieval_results.values()
            for c in ro.chunks
        }

    def _enumerate_expand(
        self,
        state: OrchestratorState,
        tracer: PipelineTracer,
        fetch_k: int,
        base_plan: Optional[SearchPlan] = None,
    ) -> "_RequeryResult":
        """Exhaustive gather for comprehensive/enumerate queries.

        Two levers the plain re-query lacks:
          1. Depth — searches run at ``fetch_k`` (escalated by the caller on dry
             rounds), reaching ranks below the initial top-K.
          2. Facet partition — one extra pass per mined year (year-scoped filter),
             so each year's own top-K surfaces instead of only the globally
             dominant region. Requires `enumerate_facet_partition`.

        ``base_plan`` short-circuits the broaden-LLM call: a depth-escalation round
        re-runs the previous round's drafts deeper, so the caller passes that plan
        back in instead of paying an LLM round-trip that would (deterministically,
        temperature 0) regenerate the same drafts.

        Already-tried (query × filter × depth) searches are pruned via the
        tried-ledger; chunks from every remaining pass are deduped into the live
        pool via _merge_new_chunks.
        """
        from src.api import db as term_db

        if base_plan is None:
            missing_aspects = state.evidence_decision.missing_aspects if state.evidence_decision else None
            broader = self._planner.broaden(
                state.user_query,
                state.planner_output,
                tracer,
                selected_collections=state.selected_collections or self._production,
                result_count=len(state.assembled_chunks),
                missing_aspects=missing_aspects,
                rejected_hypotheses=term_db.list_rejected_term_hypotheses(),
                tried_queries=self._tried_query_texts(state),
            )
            if broader is not None and broader.resources:
                # Carry the original query_type so _flat_plans_for keeps the comprehensive
                # depth (see the regression note in _requery_expand).
                broader.query_type = state.planner_output.query_type if state.planner_output else broader.query_type
                base_plan = broader
            else:
                base_plan = state.planner_output
        # Initial plans never carry one; a reused broaden plan keeps its hypothesis
        # alive so a deeper round that finally succeeds can still record it.
        term_hypothesis = base_plan.term_hypothesis if base_plan else None

        allowed = ([r.collection for r in base_plan.resources] if base_plan else None) \
            or [p.collection_name for p in state.collection_plans]
        if not allowed:
            return _RequeryResult(added=0, draft_texts={}, term_hypothesis=term_hypothesis, plan=base_plan)

        # Pass 1 = unfiltered (deeper global reach); passes 2..N = one per mined year.
        filter_variants: list[Optional[dict]] = [None]
        if self._config.retrieval_budget.enumerate_facet_partition:
            for year in self._facet_years(state):
                wf = where_year_filter([year])
                if wf is not None:
                    filter_variants.append(wf)

        query_text = (base_plan.refined_query if base_plan else None) or state.user_query
        added_total = 0
        pruned_total = 0
        draft_texts: dict[str, list[str]] = {}
        for variant in filter_variants:
            plans = self._flat_plans_for(base_plan, allowed, fetch_k=fetch_k, override_filter=variant)
            plans, pruned = self._dedupe_and_register(state, plans, query_text)
            pruned_total += pruned
            if not plans:
                continue
            results = self._run_retrieval(plans, query_text)
            if not results:
                continue
            added_total += self._merge_new_chunks(state, results, plans)
            for p in plans:
                bucket = draft_texts.setdefault(p.collection_name, [])
                for d in p.query_drafts:
                    if d not in bucket:
                        bucket.append(d)

        if added_total > 0:
            state.expanded = True
        state.expand_iterations += 1
        return _RequeryResult(added=added_total, draft_texts=draft_texts,
                              term_hypothesis=term_hypothesis, plan=base_plan, pruned=pruned_total)

    def _sira_sayisi_filter_for(
        self, state: OrchestratorState, collections: list[str],
    ) -> Optional[dict]:
        """Metadata omurgası — reflect'in çıkardığı sıra sayısı çıpasından pozitif
        ``{'sira_sayisi': {'$eq': N}}`` where-filtresi kur (Hop-2'nin KESİN filtresi:
        kanun adı geçmese bile kimliksiz roll-call/görüşme/rapor bölgesini getirir).

        None (no-op) döner ve o zaman hop bugünkü filtresiz semantik aramaya düşer:
        çıpa yoksa, sıra sayısı int'e çevrilemiyorsa, ya da hedef koleksiyonlardan
        biri ``sira_sayisi`` metadata'sını taşımıyorsa. Son guard, ``override_filter``
        tek ``_flat_plans_for`` çağrısında TÜM koleksiyonlara aynen uygulandığından
        şart: sira_sayisi taşımayan bir koleksiyona filtre dayatmak onu sıfırlardı
        (``_seen_chunk_ids`` chunk-id dışlamasıyla dikey çalışır — biri hangi chunk'ların
        ELENDİĞİNİ, diğeri hangi bölgenin ARANDIĞINI belirler)."""
        raw = (state.extracted_anchors or {}).get("sira_sayisi")
        if raw is None:
            return None
        try:
            n = int(str(raw).strip())
        except (TypeError, ValueError):
            return None  # "S.S. 5" gibi çözülmemiş biçim → filtresiz semantiğe bırak
        for name in collections:
            spec = COLLECTIONS.get(name)
            allowed = FILTER_APPLICABILITY.get(spec.doc_type) if spec else None
            if allowed is None or "sira_sayisi" not in allowed:
                return None
        return build_chroma_where(FilterCriteria(sira_sayisi=n), collections[0])

    def _reflect_retrieve(
        self, state: OrchestratorState, plan: SearchPlan, tracer: PipelineTracer,
        exclude_seen: bool = False,
        window_expand: bool = False,
    ) -> "_RequeryResult":
        """Execute one reflect hop's SearchPlan: retrieve anew, merge unique chunks.

        Carries the in-flight query_type onto the reflect plan so the retrieval budget
        matches the strategy — the reflect JSON has no query_type field, so _parse_plan
        defaults next_plan.query_type to "fact", the same starvation _requery_expand
        guards against (see the regression note there). Comprehensive-adaptive strategies
        (e.g. kanun_gorusmeleri) delegate to _enumerate_expand so the facet-partition +
        depth-escalation gather that makes "tüm görüşmeler" exhaustive is preserved;
        the reflect hop rides in as base_plan (no extra broaden-LLM call).

        ``exclude_seen`` (from strategy["exclude_seen_chunks"], e.g. kanun_kabul_oylama)
        holds the exact chunk ids already surfaced this run out of this hop's ranked pool
        so fetch_k fills with novel chunks (see _run_retrieval's exclude_ids). Deliberately
        chunk-id, NOT session $nin: excluding a whole birleşim can black out an unreached
        target (e.g. an identity-less roll-call vote table) that lives inside a sitting
        whose other chunks were already touched. Consumed only by the non-comprehensive
        branch below — kanun_gorusmeleri delegates to _enumerate_expand and does not opt in.
        """
        if state.planner_output:
            plan.query_type = state.planner_output.query_type

        if plan.query_type == "comprehensive":
            fetch_k = self._config.retrieval_budget.budget_for("comprehensive").fetch_k
            return self._enumerate_expand(state, tracer, fetch_k=fetch_k, base_plan=plan)

        allowed = [r.collection for r in plan.resources]
        if not allowed:
            return _RequeryResult(added=0, draft_texts={}, term_hypothesis=None, plan=plan)
        query_text = plan.refined_query or state.user_query
        # Snapshot the seen pool BEFORE this round's merge, so it reflects the hop's start.
        exclude_ids = self._seen_chunk_ids(state) if exclude_seen else None
        excluded_n = len(exclude_ids) if exclude_ids else 0
        # Metadata omurgası: sıra sayısı çıpası çözüldüyse bu hop'u kesin where-filtresine
        # bağla (kanun adı geçmese bile kimliksiz roll-call bölgesini getirir). exclude_ids
        # (chunk-id dışlama) ile dikey: biri neyi ELER, diğeri hangi bölgeyi ARAR.
        sira_filter = self._sira_sayisi_filter_for(state, allowed)
        plans = self._flat_plans_for(plan, allowed, override_filter=sira_filter)
        plans, pruned = self._dedupe_and_register(state, plans, query_text)
        draft_texts = {p.collection_name: list(p.query_drafts) for p in plans}
        results = self._run_retrieval(plans, query_text, exclude_ids=exclude_ids) if plans else {}
        if not results:
            state.expand_iterations += 1
            return _RequeryResult(added=0, draft_texts=draft_texts, term_hypothesis=None,
                                  plan=plan, pruned=pruned, excluded_seen_count=excluded_n,
                                  sira_filter=sira_filter)
        # Window-expand: çıpanın (en yüksek skorlu chunk) belge-içi okuma-sırası komşularını
        # merge ÖNCESİ pool'a splice et (kimliksiz roll-call'ı self-identifying duyurunun
        # yanına getirir). Strateji flag'i + global knob kapalıysa no-op.
        window_added = self._expand_windows(state, results, window_expand)
        added = self._merge_new_chunks(state, results, plans)
        if added > 0:
            state.expanded = True
        state.expand_iterations += 1
        return _RequeryResult(added=added, draft_texts=draft_texts, term_hypothesis=None,
                              plan=plan, pruned=pruned, excluded_seen_count=excluded_n,
                              sira_filter=sira_filter, window_added=window_added)

    def _compact_evidence(self, state: OrchestratorState) -> str:
        """Deterministic compact summary of gathered evidence for the reflect LLM.

        The full chunk pool won't fit the reflect context, so take the top-N assembled
        chunks (already rerank-ordered) and emit a short header + head-truncated body per
        chunk. esas_no/sıra sayısı usually appear near a report/agenda chunk's head, so
        truncating from the head keeps them readable. Bounded by the reflect.evidence_* caps.
        """
        cfg = self._config.reflect
        parts: list[str] = []
        total = 0
        for c in state.assembled_chunks[: cfg.evidence_max_chunks]:
            meta = c.metadata or {}
            tags = [c.source_title or c.document_id]
            if meta.get("date"):
                tags.append(str(meta.get("date")))
            no = meta.get("esas_no") or meta.get("sira_sayisi")
            if no:
                tags.append(f"no={no}")
            head = " | ".join(str(t) for t in tags if t)
            body = (c.text or "")[: cfg.evidence_char_cap_per_chunk]
            block = f"[{head}]\n{body}"
            if total + len(block) > cfg.evidence_total_char_cap and parts:
                break
            parts.append(block)
            total += len(block)
        return "\n\n".join(parts)

    # =============================================================== helpers

    @staticmethod
    def _fuse_draft_chunks(draft_lists: list[list["Chunk"]], k: int = 60) -> list["Chunk"]:
        """RRF-merge the ranked chunk lists produced by a collection's query drafts."""
        if not draft_lists:
            return []
        if len(draft_lists) == 1:
            return draft_lists[0]

        scores: dict[str, float] = {}
        best: dict[str, Chunk] = {}
        for chunks in draft_lists:
            for rank, ch in enumerate(chunks, start=1):
                scores[ch.chunk_id] = scores.get(ch.chunk_id, 0.0) + 1.0 / (k + rank)
                if ch.chunk_id not in best or ch.rerank_score > best[ch.chunk_id].rerank_score:
                    best[ch.chunk_id] = ch
        ordered = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [best[cid] for cid, _ in ordered]

    @staticmethod
    def _dict_to_chunks(result: dict, collection_name: str) -> list[Chunk]:
        docs = result.get("documents", []) or []
        metas = result.get("metadatas", []) or []
        dists = result.get("distances", []) or []
        out: list[Chunk] = []
        for i, (doc, meta) in enumerate(zip(docs, metas)):
            dist = dists[i] if i < len(dists) else 0.0
            out.append(Chunk(
                chunk_id=meta.get("chunk_id") or f"{collection_name}_{i}",
                document_id=meta.get("document_id") or meta.get("chunk_id") or f"{collection_name}_{i}",
                collection_name=collection_name,
                doc_type=meta.get("doc_type") or "unknown",
                source_title=meta.get("source_title") or meta.get("title") or "",
                text=doc,
                score=1.0 - float(dist),
                rerank_score=float(meta.get("rerank_score", 1.0 - float(dist))),
                metadata=meta,
            ))
        return out

    @staticmethod
    def _apply_reading_order(state: OrchestratorState) -> None:
        """Reorder ``assembled_chunks`` into reading order for window-expanded documents.

        Window-expand splices neighbors at the TAIL of a collection block (via merge), and
        neither _assemble_passthrough nor _build_context reorders — so an anchor and its
        neighbors would present scattered. This single-shot pass (run once at answering,
        the terminal node) groups each expanded document's chunks contiguously in
        chunk-order at the anchor block's original (highest-rerank) position, leaving every
        other chunk's relative order untouched. Context AND citations read the reordered
        list, so ``[i]`` markers stay aligned with ``out.sources``. No-op when nothing was
        expanded (default). Field-free: order is parsed from chunk_id.
        """
        keys = set(state.window_anchor_keys)
        if not keys:
            return

        def _dockey(c) -> Optional[str]:
            parsed = _parse_chunk_order(c.chunk_id)
            return f"{c.collection_name}\x1f{parsed[0]}" if parsed else None

        buckets: dict[str, list] = {}
        for c in state.assembled_chunks:
            k = _dockey(c)
            if k in keys:
                buckets.setdefault(k, []).append(c)
        for chunks in buckets.values():
            chunks.sort(key=lambda c: _parse_chunk_order(c.chunk_id)[1])

        rebuilt: list = []
        emitted: set[str] = set()
        for c in state.assembled_chunks:
            k = _dockey(c)
            if k in keys:
                if k not in emitted:  # emit this doc's whole sorted bucket at first sighting
                    rebuilt.extend(buckets[k])
                    emitted.add(k)
            else:
                rebuilt.append(c)
        state.assembled_chunks = rebuilt

    @staticmethod
    def _build_context(state: OrchestratorState) -> str:
        blocks = []
        # Adaptive multi-hop runs pin the resolved identity (esas_no/sıra sayısı) so the
        # answer LLM can verify each record belongs to the ASKED law and never attribute
        # another law's vote/gerekçe from the same session.
        if state.extracted_anchors:
            anchor_str = ", ".join(
                f"{k}={v}" for k, v in state.extracted_anchors.items() if v not in (None, "")
            )
            if anchor_str:
                blocks.append(
                    "DOĞRULANAN KİMLİK (bu araştırmada çıpalanan varlıklar — kullandığın her "
                    f"kaydın SORULAN kanuna ait olduğunu bununla doğrula): {anchor_str}"
                )
        for i, c in enumerate(state.assembled_chunks, start=1):
            blocks.append(f"[{i}] ({c.collection_name}/{c.document_id}/{c.chunk_id})\n{c.text}")
        return "\n\n".join(blocks)

    def _build_refuse_output(self, state: OrchestratorState, reason: str, tracer: PipelineTracer) -> AgentOutput:
        message = _REFUSE_MESSAGES.get(reason, _REFUSE_MESSAGES["judge_refuse"])
        if state.evidence_decision is None and reason == "no_allowed_collections":
            state.evidence_decision = EvidenceDecision(
                sufficient=False, confidence=0.0, action="refuse",
                missing_aspects=[reason], judge_type="heuristic",
            )
        return AgentOutput(
            answer=message,
            plan=state.planner_output,
            trace=tracer.events,
            sources=[],
            policy_result=state.policy_result,
            evidence_decision=state.evidence_decision,
            assembly=state.balanced_context,
            expanded=state.expanded,
            clarification=state.clarification,
            rabbit_holes=state.rabbit_holes,
        )

    def _conversational_output(
        self,
        query: str,
        chat_history: Optional[list],
        tracer: PipelineTracer,
        stream_callback: Optional[callable] = None,
    ) -> AgentOutput:
        with tracer.phase("answering") as ctx:
            ans_cfg = self._config.answering
            block_name = ans_cfg.block
            model_key = ans_cfg.model_key

            client = self._pool.get_client(block_name)
            model = self._pool.get_model_for_block(block_name, model_key)

            sys_prompt = CONVERSATIONAL_SYS_PROMPT

            history_messages = []
            for m in (chat_history or []):
                role = m.get("role") or "user"
                msg_content = m.get("content") or ""
                if not msg_content:
                    continue
                if role == "assistant":
                    msg_content = msg_content[:1500]
                history_messages.append({"role": role, "content": msg_content})

            stream = client.chat(
                model=model,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    *history_messages,
                    {"role": "user", "content": query},
                ],
                options={
                    "temperature": ans_cfg.temperature,
                    "num_predict": min(ans_cfg.num_predict, self._config.get_block(block_name).max_num_predict),
                },
                stream=True,
                # think açıkça geçilmeli: langchain-ollama, reasoning truthy
                # değilken sunucudan gelen thinking deltalarını sessizce düşürür
                # (raw istemci koşulsuz geçiriyordu). AnswerTool ile aynı kural.
                think=ans_cfg.think if ans_cfg.think is not None else False,
            )

            thinking = ""
            content = ""
            for chunk in stream:
                if hasattr(chunk.message, "thinking") and chunk.message.thinking:
                    thinking += chunk.message.thinking
                if hasattr(chunk.message, "content") and chunk.message.content:
                    token = chunk.message.content
                    content += token
                    if stream_callback is not None:
                        try:
                            stream_callback({"type": "content", "content": token})
                        except Exception:
                            pass

            if ctx:
                ctx.update_details(answer_chars=len(content))
                if self._config.expose_thinking:
                    if thinking:
                        ctx.update_details(thinking=thinking)
                    ctx.update_details(answer_preview=content[:600])

            return AgentOutput(
                answer=content,
                thinking=thinking,
                scope="conversational",
                trace=tracer.events,
            )
