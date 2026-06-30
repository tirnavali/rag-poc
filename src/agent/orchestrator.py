"""OrchestratorAgent — the single agent pipeline.

Unifies the former legacy/orchestrator split into one explicit state machine:

  bad_words? → intent (scope + tool/db) → probe + facets + clarification →
  planning → policy? → allocation? → retrieve → assemble → judge →
  (bounded re-query → re-assemble → re-judge) → answer → sanitize → cite

Stage-2 gates (`bad_words_filter`, `policy`, `allocation`) are toggled per-stage
in pipeline.yaml; when disabled the orchestrator supplies sensible fallbacks
(allowed = planner suggestions; flat single-pool execution plans).
"""
from __future__ import annotations

import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from src.agent.allocator import AllocationPlanner
from src.agent.assembler import BalancedContextAssembler
from src.agent.bad_words_filter import BadWordsFilter
from src.agent.citations import CitationBuilder
from src.agent.clarifier import AmbiguityGate, FacetMiner, QueryRefiner
from src.agent.classifier import ScopeClassifier
from src.agent.expander import ExpansionPlanner
from src.agent.judge import EvidenceJudge
from src.agent.planner import Planner
from src.agent.policy import PolicyEnforcer
from src.agent.sanitizer import SanitizerAgent
from src.agent.schemas import (
    AgentOutput,
    Chunk,
    ClarificationResult,
    CollectionExecutionPlan,
    EvidenceDecision,
    OrchestratorState,
    PolicyResult,
    RetrievalOutput,
    SearchPlan,
)
from src.agent.suggester import Suggester
from src.agent.tools import AnswerTool, SearchTool
from src.agent.tracer import PipelineTracer
from src.common.llm_client_pool import LLMClientPool
from src.config.collections import COLLECTIONS, get_production_collection_keys
from src.config.pipeline_loader import PipelineConfig


_REFUSE_MESSAGES = {
    "no_allowed_collections": (
        "Seçili koleksiyonlarda bu konu için arama yapılamaz. "
        "Başlangıçta farklı koleksiyonlar seçin."
    ),
    "judge_refuse": "Yetkili kaynaklarla yanıt veremiyorum.",
    "clarify": "Sorunuzu netleştirir misiniz? Yeterli kanıt bulunamadı.",
}


class OrchestratorAgent:
    """The single agentic RAG pipeline (see module docstring for the stage order)."""

    def __init__(self, config: PipelineConfig, client_pool: LLMClientPool, filter_extractor=None) -> None:
        self._config = config
        self._pool = client_pool
        self._planner = Planner(config, client_pool, filter_extractor)
        self._policy = PolicyEnforcer(config.policy)
        self._allocator = AllocationPlanner(config.allocation)
        self._search_tool = SearchTool(config, client_pool)
        self._assembler = BalancedContextAssembler(config.allocation)
        self._judge = EvidenceJudge(config.judge, client_pool)
        self._expander = ExpansionPlanner()
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
    ) -> AgentOutput:
        state = OrchestratorState(request_id=str(uuid.uuid4()), user_query=query)
        tracer = PipelineTracer(on_phase=on_phase, on_phase_end=on_phase_end)
        session_collections = session_collections or []

        # ---- Stage 0: bad-words gate (stage-2; off by default) ----
        if self._bad_words is not None:
            bw = self._bad_words.check(query)
            with tracer.phase("bad_words_filter", details={"matched": bw.matched, "matched_terms": bw.matched_terms}):
                pass
            if bw.matched:
                return AgentOutput(
                    answer=self._config.bad_words_filter.response_message,
                    scope="bad_word",
                    trace=tracer.events,
                )

        # ---- Stage 1: intent (scope + tool/db selection) ----
        if self._classifier is not None:
            scope_result = self._classifier.classify(query, tracer)
            if (
                scope_result.scope == "off_domain"
                and scope_result.confidence >= self._config.classifier.confidence_threshold
            ):
                return self._off_domain_output(query, tracer)
            if scope_result.scope == "conversational":
                return self._conversational_output(query, chat_history, tracer, stream_callback)
            # Validate intent's collection picks against the production universe.
            prod = set(self._production)
            state.selected_collections = [
                c for c in scope_result.selected_collections if c in COLLECTIONS and c in prod
            ]

        # ---- Stage 1.5/1.6: probe + facets + grounded clarification ----
        if self._config.clarification.enabled:
            self._clarify(state, tracer, clarification_callback, deep_mode)

        # ---- Stage 2: planning (intent + constraints → diversified plan) ----
        with tracer.phase("planning") as ctx:
            state.planner_output = self._planner.plan(
                query,
                tracer,
                # Restrict the planner's catalog + routing to the production universe.
                selected_collections=state.selected_collections or self._production,
                constraints=state.applied_constraints or None,
                max_variants=self._config.planner.normal_max_query_variants,
            )
            if ctx and state.planner_output:
                ctx.update_details(
                    intent=state.planner_output.intent,
                    query_type=state.planner_output.query_type,
                    collections=[r.collection for r in state.planner_output.resources],
                    drafts={r.collection: [d.text for d in r.query_drafts]
                            for r in state.planner_output.resources},
                )
                if self._config.expose_thinking and state.planner_output.reasoning:
                    ctx.update_details(reasoning=state.planner_output.reasoning)

        # ---- Stage 2a: policy (stage-2; off → allow planner suggestions) ----
        with tracer.phase("policy") as ctx:
            if self._config.policy.enabled:
                self._policy.run(state, session_collections)
            else:
                suggested = [r.collection for r in state.planner_output.resources] if state.planner_output else []
                state.policy_result = PolicyResult(allowed_collections=suggested, denied_collections=[])
            if ctx and state.policy_result:
                ctx.update_details(
                    enabled=self._config.policy.enabled,
                    allowed=state.policy_result.allowed_collections,
                    denied=state.policy_result.denied_collections,
                )
        if not state.policy_result.allowed_collections:
            return self._build_refuse_output(state, "no_allowed_collections", tracer)

        # ---- Stage 2b: allocation (stage-2; off → flat single-pool plans) ----
        with tracer.phase("allocation") as ctx:
            if self._config.allocation.enabled:
                self._allocator.run(state)
            else:
                state.collection_plans = self._flat_plans_for(
                    state.planner_output, state.policy_result.allowed_collections
                )
            if ctx:
                ctx.update_details(
                    enabled=self._config.allocation.enabled,
                    plans=[
                        {"collection": p.collection_name, "primary": p.retrieval_budget,
                         "reserve": p.reserve_budget, "fetch_k": p.fetch_k}
                        for p in state.collection_plans
                    ],
                )
        if not state.collection_plans:
            return self._build_refuse_output(state, "no_allowed_collections", tracer)

        # ---- Stage 3: retrieval (parallel fan-out + RRF + rerank) ----
        with tracer.phase("retrieval") as ctx:
            state.retrieval_results = self._run_retrieval(state.collection_plans, self._fallback_query(state))
            if ctx:
                ctx.update_details(per_collection={
                    name: {"fetched": ro.fetched_count, "returned": ro.returned_count, "latency_ms": ro.latency_ms}
                    for name, ro in state.retrieval_results.items()
                })

        # ---- Stage 4: assembly ----
        with tracer.phase("assembly") as ctx:
            self._assembler.run(state)
            if ctx:
                ctx.update_details(
                    primary_count=len(state.assembled_chunks),
                    collection_coverage=len({c.collection_name for c in state.assembled_chunks}),
                )

        # ---- Stage 5: judge ----
        with tracer.phase("judge") as ctx:
            self._judge.run(state)
            if ctx and state.evidence_decision:
                ctx.update_details(
                    judge_type=state.evidence_decision.judge_type,
                    action=state.evidence_decision.action,
                    confidence=state.evidence_decision.confidence,
                    missing_aspects=state.evidence_decision.missing_aspects,
                )
                if self._config.expose_thinking and state.evidence_decision.reasoning:
                    ctx.update_details(reasoning=state.evidence_decision.reasoning)

        # ---- Stage 5.1: bounded re-query expansion ----
        max_iters = self._config.judge.max_expand_iterations
        if state.evidence_decision.action == "expand" and max_iters > 0:
            with tracer.phase("expansion") as ctx:
                if self._config.judge.expand_strategy == "requery":
                    self._requery_expand(state, tracer)
                    self._assembler.run(state)
                else:
                    self._expander.run(state)
                if ctx:
                    ctx.update_details(expanded=state.expanded, post_count=len(state.assembled_chunks))
            with tracer.phase("judge_post_expand") as ctx:
                self._judge.run(state)
                if ctx and state.evidence_decision:
                    ctx.update_details(
                        judge_type=state.evidence_decision.judge_type,
                        action=state.evidence_decision.action,
                    )

        action = state.evidence_decision.action
        if action == "clarify":
            return self._build_refuse_output(state, "clarify", tracer)
        if action == "refuse":
            return self._build_refuse_output(state, "judge_refuse", tracer)

        # ---- Stage 6: answer → sanitize → cite ----
        with tracer.phase("answering") as ctx:
            context = self._build_context(state)
            thinking, answer = self._answer_tool.generate(query=query, context=context, chat_history=chat_history)
            state.final_answer = answer
            if stream_callback is not None:
                try:
                    stream_callback({"type": "content", "content": answer})
                except Exception:
                    pass
            if ctx:
                ctx.update_details(answer_chars=len(answer), context_chars=len(context))
                if self._config.expose_thinking:
                    if thinking:
                        ctx.update_details(thinking=thinking)
                    ctx.update_details(answer_preview=answer[:600])

        with tracer.phase("validation") as ctx:
            validation = self._sanitizer.validate(
                query=query,
                answer=answer,
                sources=[c.metadata for c in state.assembled_chunks],
                context=context,
            )
            # Non-destructive: validation is an advisory quality signal only. We do
            # NOT overwrite the answering agent's (strong-model) text with the
            # sanitizer's (weaker-model) corrected_answer, which tended to condense
            # and crop long answers. passes/issues stay visible in the trace.
            if ctx and validation:
                ctx.update_details(passes=getattr(validation, "passes", True))
                if self._config.expose_thinking:
                    issues = getattr(validation, "issues", None)
                    if issues:
                        ctx.update_details(issues=issues)
                    if not getattr(validation, "passes", True) and getattr(validation, "corrected_answer", None):
                        # Surface the suggestion for transparency, but don't apply it.
                        ctx.update_details(suggested_correction=validation.corrected_answer[:600])

        with tracer.phase("citation") as ctx:
            state.citations = CitationBuilder.build(state.assembled_chunks)
            if ctx:
                ctx.update_details(citation_count=len(state.citations))

        return AgentOutput(
            answer=state.final_answer,
            thinking=thinking,
            plan=state.planner_output,
            validation=validation,
            trace=tracer.events,
            sources=state.citations,
            policy_result=state.policy_result,
            evidence_decision=state.evidence_decision,
            assembly=state.balanced_context,
            expanded=state.expanded,
            clarification=state.clarification,
        )

    # ============================================================== clarify

    def _clarify(self, state, tracer, clarification_callback, deep_mode) -> None:
        """Probe-retrieve, mine facets, and (if ambiguous) narrow year/scope/topic."""
        cfg = self._config.clarification
        probe_cols = state.selected_collections or self._production

        with tracer.phase("probe") as ctx:
            probe_results = self._probe(state.user_query, probe_cols, cfg.probe_k)
            state.facets = self._facet_miner.mine(probe_results)
            if ctx:
                ctx.update_details(
                    probed=len(probe_cols),
                    hits=state.facets.total,
                    years=len(state.facets.years),
                    topics=len(state.facets.topics),
                )

        if not self._gate.is_ambiguous(state.facets, state.user_query):
            return

        max_turns = cfg.max_turns_deep if deep_mode else cfg.max_turns_normal
        with tracer.phase("clarification") as ctx:
            if clarification_callback is not None and max_turns > 0:
                questions = self._refiner.build_questions(state.user_query, state.facets, tracer)
                if questions:
                    try:
                        answers = clarification_callback(questions)
                    except Exception:
                        answers = None
                    constraints = self._refiner.resolve(questions, answers)
                    state.applied_constraints = constraints
                    state.clarify_turns = 1
                    state.clarification = ClarificationResult(
                        asked=True, turns=1, questions=questions,
                        year=constraints.get("year"),
                        collections=constraints.get("collections", []),
                        topic=constraints.get("topic"),
                    )
            else:
                # Non-interactive: auto-apply the strongest facet + assumption note.
                constraints, note = self._refiner.auto_constraints(state.facets)
                state.applied_constraints = constraints
                state.clarification = ClarificationResult(
                    auto_applied=bool(constraints), note=note,
                    year=constraints.get("year"), topic=constraints.get("topic"),
                )
            if ctx and state.clarification:
                ctx.update_details(
                    asked=state.clarification.asked,
                    auto_applied=state.clarification.auto_applied,
                    constraints=state.applied_constraints,
                )

    def _probe(self, query: str, collections: list[str], probe_k: int) -> list[dict]:
        """Cheap broad retrieval over `collections` to mine facets from."""
        def _one(name):
            try:
                return self._search_tool.search(collection_key=name, query_text=query, filters=None, top_k=probe_k)
            except Exception:
                return {"documents": [], "metadatas": [], "distances": []}

        if not collections:
            return []
        with ThreadPoolExecutor(max_workers=min(8, len(collections))) as ex:
            return list(ex.map(_one, collections))

    def _off_domain_output(self, query: str, tracer: PipelineTracer) -> AgentOutput:
        suggestions = self._suggester.suggest(query, tracer)
        template = (
            self._config.off_domain_response_template
            or "Bu sistem alan dışında.\n1. {suggestion_0}\n2. {suggestion_1}\n3. {suggestion_2}"
        )
        padded = (suggestions + ["", "", ""])[:3]
        answer = template.format(suggestion_0=padded[0], suggestion_1=padded[1], suggestion_2=padded[2])
        return AgentOutput(answer=answer, scope="off_domain", suggestions=suggestions, trace=tracer.events)

    # ====================================================== allocation fallback

    def _flat_plans_for(self, search_plan: Optional[SearchPlan], allowed: list[str]) -> list[CollectionExecutionPlan]:
        """Build flat single-pool execution plans (allocation disabled).

        Reuses AllocationPlanner's filter/draft mapping but with a flat budget:
        all fused hits flow into the primary pool (reserve=0), capped downstream
        by `allocation.max_total_primary`.
        """
        filters_by = AllocationPlanner._collect_first_filters(search_plan) if search_plan else {}
        drafts_by = AllocationPlanner._collect_draft_texts(search_plan) if search_plan else {}
        qt = search_plan.query_type if search_plan else "fact"
        fetch_k = self._config.allocation.budget_for(qt).fetch_k
        plans = []
        for idx, name in enumerate(allowed):
            plans.append(CollectionExecutionPlan(
                collection_name=name,
                priority=idx + 1,
                retrieval_budget=fetch_k,
                reserve_budget=0,
                fetch_k=fetch_k,
                filters=filters_by.get(name, {}),
                query_drafts=drafts_by.get(name, []),
                route_reason="flat_allocation_disabled",
            ))
        return plans

    def _fallback_query(self, state: OrchestratorState) -> str:
        refined = state.planner_output.refined_query if state.planner_output else None
        return refined or state.user_query

    # ============================================================ retrieval

    def _run_retrieval(self, plans: list[CollectionExecutionPlan], fallback_query: str) -> dict[str, RetrievalOutput]:
        """Fan out one search per (collection × draft), RRF-fuse per collection.

        Returns a fresh RetrievalOutput map; does not mutate state, so the same
        helper serves both the initial retrieval and bounded re-query expansion.
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
            primary = fused[: plan.retrieval_budget]
            reserve = fused[plan.retrieval_budget: plan.retrieval_budget + plan.reserve_budget]
            results[plan.collection_name] = RetrievalOutput(
                collection_name=plan.collection_name,
                chunks=primary,
                reserve_chunks=reserve,
                fetched_count=len(fused),
                returned_count=len(primary),
                latency_ms=latency_by_plan[pi],
                filter_applied=plan.filters or {},
            )
        return results

    def _requery_expand(self, state: OrchestratorState, tracer: PipelineTracer) -> None:
        """Bounded re-query: broaden the plan, retrieve anew, merge unique chunks.

        Unlike reserve-promotion this issues fresh vector searches. New chunks are
        appended to the existing primary buffers and the matching execution plan's
        budget is raised so the re-assembly step can surface them.
        """
        broader = self._planner.broaden(
            state.user_query,
            state.planner_output,
            tracer,
            selected_collections=state.selected_collections or self._production,
        )
        if broader is None or not broader.resources:
            return

        # A hallucinated collection key fails safely inside SearchTool (KeyError →
        # caught and dropped in _run_retrieval), so no catalog pre-filter is needed.
        allowed = [r.collection for r in broader.resources]
        requery_plans = self._flat_plans_for(broader, allowed)
        new_results = self._run_retrieval(requery_plans, broader.refined_query or state.user_query)
        if not new_results:
            return

        existing_by = {p.collection_name: p for p in state.collection_plans}
        plan_by_name = {p.collection_name: p for p in requery_plans}
        added_total = 0
        for name, rr_new in new_results.items():
            rr = state.retrieval_results.get(name)
            if rr is None:
                state.retrieval_results[name] = rr_new
                state.collection_plans.append(plan_by_name[name])
                added_total += len(rr_new.chunks)
                continue
            seen = {c.chunk_id for c in rr.chunks}
            added = [c for c in rr_new.chunks if c.chunk_id not in seen]
            if added:
                rr.chunks = rr.chunks + added
                if name in existing_by:
                    existing_by[name].retrieval_budget += len(added)
                added_total += len(added)

        if added_total > 0:
            state.expanded = True
        state.expand_iterations += 1

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
    def _build_context(state: OrchestratorState) -> str:
        blocks = []
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

            sys_prompt = (
                "Sen yardımsever bir yapay zeka arşiv asistanısın. "
                "Kullanıcı ile olan geçmiş konuşmana (hafızaya) ve güncel sorusuna dayanarak doğrudan ve doğal bir yanıt ver. "
                "Arşiv taraması yapmana gerek yoktur. Kısa, samimi ve Türkçe cevap ver."
            )

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
