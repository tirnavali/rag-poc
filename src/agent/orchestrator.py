"""OrchestratorAgent — explicit state-machine pipeline replacing PlanningAgent retry loops."""
from __future__ import annotations

import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from src.agent.allocator import AllocationPlanner
from src.agent.assembler import BalancedContextAssembler
from src.agent.citations import CitationBuilder
from src.agent.expander import ExpansionPlanner
from src.agent.judge import EvidenceJudge
from src.agent.planner import Planner
from src.agent.policy import PolicyEnforcer
from src.agent.sanitizer import SanitizerAgent
from src.agent.schemas import (
    AgentOutput,
    Chunk,
    EvidenceDecision,
    OrchestratorState,
    RetrievalOutput,
)
from src.agent.tools import AnswerTool, SearchTool
from src.agent.tracer import PipelineTracer
from src.common.llm_client_pool import LLMClientPool
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
    """Runs the new state-machine pipeline.

    Components: Planner → Policy → Allocator → Retrieve → Assembler →
    Judge → (Expand → Re-Judge) → Answer → Sanitizer → Citation.
    """

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

    def run(
        self,
        query: str,
        session_collections: list[str],
        stream_callback: Optional[callable] = None,
    ) -> AgentOutput:
        state = OrchestratorState(
            request_id=str(uuid.uuid4()),
            user_query=query,
        )
        tracer = PipelineTracer()

        with tracer.phase("planning") as ctx:
            state.planner_output = self._planner.plan(query, tracer)
            if ctx and state.planner_output:
                ctx.update_details(
                    intent=state.planner_output.intent,
                    query_type=state.planner_output.query_type,
                    suggested_collections=[r.collection for r in state.planner_output.resources],
                )

        with tracer.phase("policy") as ctx:
            self._policy.run(state, session_collections)
            if ctx and state.policy_result:
                ctx.update_details(
                    allowed=state.policy_result.allowed_collections,
                    denied=state.policy_result.denied_collections,
                )
        if not state.policy_result.allowed_collections:
            return self._build_refuse_output(state, "no_allowed_collections", tracer)

        with tracer.phase("allocation") as ctx:
            self._allocator.run(state)
            if ctx:
                ctx.update_details(
                    plans=[
                        {"collection": p.collection_name, "primary": p.retrieval_budget,
                         "reserve": p.reserve_budget, "fetch_k": p.fetch_k,
                         "drafts": p.query_drafts or [state.user_query]}
                        for p in state.collection_plans
                    ],
                    query_type=state.planner_output.query_type if state.planner_output else "fact",
                )
        if not state.collection_plans:
            return self._build_refuse_output(state, "no_allowed_collections", tracer)

        with tracer.phase("retrieval") as ctx:
            self._retrieve_all(state)
            if ctx:
                ctx.update_details(
                    per_collection={
                        name: {
                            "fetched": ro.fetched_count,
                            "returned": ro.returned_count,
                            "latency_ms": ro.latency_ms,
                        }
                        for name, ro in state.retrieval_results.items()
                    }
                )

        with tracer.phase("assembly") as ctx:
            self._assembler.run(state)
            if ctx:
                ctx.update_details(
                    primary_count=len(state.assembled_chunks),
                    collection_coverage=len({c.collection_name for c in state.assembled_chunks}),
                )

        with tracer.phase("judge") as ctx:
            self._judge.run(state)
            if ctx and state.evidence_decision:
                ctx.update_details(
                    judge_type=state.evidence_decision.judge_type,
                    action=state.evidence_decision.action,
                    confidence=state.evidence_decision.confidence,
                )

        max_iters = self._config.judge.max_expand_iterations
        if state.evidence_decision.action == "expand" and max_iters > 0:
            with tracer.phase("expansion") as ctx:
                self._expander.run(state)
                if ctx:
                    ctx.update_details(
                        expanded=state.expanded,
                        post_count=len(state.assembled_chunks),
                    )
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

        with tracer.phase("answering") as ctx:
            context = self._build_context(state)
            # Token-level streaming through AnswerTool is not yet exposed; the
            # callback is fired once with the completed answer so downstream
            # consumers can branch on stream vs. blocking behavior.
            thinking, answer = self._answer_tool.generate(query=query, context=context)
            state.final_answer = answer
            if stream_callback is not None:
                try:
                    stream_callback({"type": "content", "content": answer})
                except Exception:
                    pass
            if ctx:
                ctx.update_details(answer_chars=len(answer), context_chars=len(context))

        with tracer.phase("validation") as ctx:
            validation = self._sanitizer.validate(
                query=query,
                answer=answer,
                sources=[c.metadata for c in state.assembled_chunks],
                context=context,
            )
            if validation and not getattr(validation, "passes", True) and getattr(validation, "corrected_answer", None):
                state.final_answer = validation.corrected_answer
            if ctx and validation:
                ctx.update_details(passes=getattr(validation, "passes", True))

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
        )

    def _retrieve_all(self, state: OrchestratorState) -> None:
        plans = [p for p in state.collection_plans if p.enabled]
        if not plans:
            return

        # Fan out one search per (collection × planner draft). The planner emits
        # several query rewrites per collection; running them all and RRF-fusing
        # the ranked lists is what makes that query expansion actually count.
        # A collection with no drafts degrades to a single search on the raw
        # query. All searches share one pool so collections AND drafts run
        # concurrently (each search is I/O-bound: Chroma + Ollama rerank).
        # When a collection has no planner drafts, fall back to the filter-words-
        # stripped refined_query (FilterExtractor) rather than the raw query, so
        # filter tokens don't pollute the vector search — mirroring RAGService.
        refined = state.planner_output.refined_query if state.planner_output else None
        fallback_query = refined or state.user_query
        tasks: list[tuple[int, str]] = []  # (plan_index, query_text)
        for pi, plan in enumerate(plans):
            drafts = plan.query_drafts or [fallback_query]
            for draft in drafts:
                tasks.append((pi, draft))

        def _one(task):
            pi, query_text = task
            plan = plans[pi]
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

        # Ranked chunk lists per collection (one list per successful draft).
        draft_lists: dict[int, list[list[Chunk]]] = {pi: [] for pi in range(len(plans))}
        latency_by_plan: dict[int, float] = {pi: 0.0 for pi in range(len(plans))}

        with ThreadPoolExecutor(max_workers=min(16, max(1, len(tasks)))) as ex:
            for pi, payload, latency in ex.map(_one, tasks):
                latency_by_plan[pi] = max(latency_by_plan[pi], latency)
                name = plans[pi].collection_name
                if isinstance(payload, Exception):
                    state.errors.append(f"retrieval_failed:{name}:{type(payload).__name__}")
                else:
                    draft_lists[pi].append(payload)

        for pi, plan in enumerate(plans):
            fused = self._fuse_draft_chunks(draft_lists[pi])
            primary = fused[: plan.retrieval_budget]
            reserve_end = plan.retrieval_budget + plan.reserve_budget
            reserve = fused[plan.retrieval_budget:reserve_end]
            state.retrieval_results[plan.collection_name] = RetrievalOutput(
                collection_name=plan.collection_name,
                chunks=primary,
                reserve_chunks=reserve,
                fetched_count=len(fused),
                returned_count=len(primary),
                latency_ms=latency_by_plan[pi],
                filter_applied=plan.filters or {},
            )

    @staticmethod
    def _fuse_draft_chunks(draft_lists: list[list["Chunk"]], k: int = 60) -> list["Chunk"]:
        """RRF-merge the ranked chunk lists produced by a collection's query drafts.

        Each draft is one ranked list; a chunk's fused score sums ``1/(k+rank)``
        over the drafts that surfaced it, so chunks found by several rewrites rank
        highest. Dedup is by ``chunk_id``, keeping the instance with the best
        rerank score. Zero drafts → empty; a single draft passes through unchanged
        (preserving the reranker's order).
        """
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
            blocks.append(
                f"[{i}] ({c.collection_name}/{c.document_id}/{c.chunk_id})\n{c.text}"
            )
        return "\n\n".join(blocks)

    def _build_refuse_output(
        self,
        state: OrchestratorState,
        reason: str,
        tracer: PipelineTracer,
    ) -> AgentOutput:
        message = _REFUSE_MESSAGES.get(reason, _REFUSE_MESSAGES["judge_refuse"])
        if state.evidence_decision is None and reason == "no_allowed_collections":
            state.evidence_decision = EvidenceDecision(
                sufficient=False,
                confidence=0.0,
                action="refuse",
                missing_aspects=[reason],
                judge_type="heuristic",
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
        )
