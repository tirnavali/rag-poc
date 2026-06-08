# Agentic RAG Orchestrator — Design Spec

**Date:** 2026-05-24
**Status:** Design (awaiting implementation plan)
**Goal:** Replace the existing planner's retry loops with an explicit orchestrator pipeline: Planner → Policy → Allocation → per-collection Retrieve → Balanced Assembly → Evidence Judge → (Expand | Answer | Clarify | Refuse) → Citation.

---

## 1. Background and Motivation

The current `src/agent/planner.py` (`PlanningAgent`) already provides:

- LLM-driven `SearchPlan` generation (collections + query drafts + filters)
- Parallel/sequential `SearchTool` execution
- A pre-answer `re_retrieval` loop (triggers when result count is low)
- A post-answer `SanitizerAgent` validation loop
- A `quality_re_retrieval` loop that re-fetches when sanitizer fails

Two issues motivate this redesign:

1. **No explicit pre-answer evidence gate.** `re_retrieval` triggers only on result *count*, not on cross-collection coverage or score quality. Borderline-evidence answers slip through and are then post-hoc rewritten by the sanitizer, which is expensive and loses recall.
2. **No collection-level budget control.** When multiple collections return results, the assembler just RRF-fuses and picks top-k. There is no guarantee of cross-collection balance, no document-level dedup, and no reserve buffer for expansion.

This spec adds an explicit orchestrator state machine on top of the existing components, replaces the two retrieval-side retry loops (`re_retrieval`, `quality_re_retrieval`) with a single pre-answer `EvidenceJudge` + reserve-buffer `ExpansionPlanner`, and keeps the post-answer `SanitizerAgent` as an orthogonal text-level validator.

## 2. Scope

**In scope:**
- New stages: Policy, Allocation, Balanced Assembly, Evidence Judge, Expansion, Citation Builder
- New Pydantic schemas under `src/agent/schemas.py`
- New entrypoint `OrchestratorAgent` in `src/agent/orchestrator.py`
- `pipeline.yaml` additions: `policy:`, `allocation:`, `judge:` blocks; corresponding loader classes
- Refactor: `PlanningAgent` strips retrieval-side retry logic, exposes a thin `Planner.plan(query) -> SearchPlan` API
- Feature flag for staged rollout; legacy code path retained for one release

**Out of scope:**
- Authentication/role-based collection access (deferred; all current collections are public)
- New ingestion or embedding model changes
- UI redesign beyond two new chat spinners (`Kısa değerlendirme...`, `Kanıt genişletiliyor...`)
- Cross-collection re-ranking with a new reranker model (reuses existing `CrossEncoderReranker`)

## 3. Architecture Overview

```
[User Query]
   |
   v
[Planner LLM] ────────────► SearchPlan (existing)
   |
   v
[Policy]      ────────────► PolicyResult        (NEW)  session ∩ planner-suggested
   |
   v
[Allocator]   ────────────► CollectionExecutionPlan[]  (NEW) budgets from pipeline.yaml
   |
   v
[Retrieve per-collection]    (existing SearchTool, parallel; fetch_k >= primary+reserve+slack)
   |
   v
[BalancedContextAssembler]  ─► assembled_chunks (primary slots), reserves held in retrieval_results
   |
   v
[EvidenceJudge]   action ∈ {answer, expand, clarify, refuse}
   |
   ├─ expand  ─► [ExpansionPlanner] (pull reserves only) ─► re-Judge (max 1 iteration)
   ├─ clarify ─► clarify message, skip answer
   ├─ refuse  ─► refuse message, skip answer
   └─ answer  ─► [AnswerGenerator stream] ─► [SanitizerAgent] (existing; kept as text validator)
                       |
                       v
                  [CitationBuilder]
                       |
                       v
                  AgentOutput (final_answer + citations + trace)
```

**Replaces in existing planner:**
- `re_retrieval` loop (low result count trigger)
- `quality_re_retrieval` loop (post-sanitizer-failure re-fetch)

**Keeps:**
- `SearchPlan` schema (with one new field; see §4)
- `SearchTool`, `ContextBuilderTool`, `AnswerTool` (in `src/agent/tools.py`)
- `SanitizerAgent` (post-answer text validation only)
- `PipelineTracer`, `LLMClientPool`
- `MultiSourceRetriever` for non-orchestrator code paths (untouched)

## 4. Schema Changes (`src/agent/schemas.py`)

### 4.1 Augment existing `SearchPlan`

Add `query_type` field (orthogonal to `intent`; drives allocator):

```python
class SearchPlan(BaseModel):
    intent: Literal["factual", "comparative", "analytical", "temporal", "unknown"]
    query_type: Literal["fact", "summary", "comparison", "reasoning", "policy"] = "fact"  # NEW
    resources: list[CollectionSearchPlan]
    reasoning: str
```

The planner prompt is updated to emit `query_type`; default `fact` if the LLM omits it.

### 4.2 New models

```python
class PolicyResult(BaseModel):
    allowed_collections: list[str]
    denied_collections: list[str] = Field(default_factory=list)
    reason_by_collection: dict[str, str] = Field(default_factory=dict)


class CollectionExecutionPlan(BaseModel):
    collection_name: str
    priority: int = 1
    retrieval_budget: int       # primary slots assembled into context
    reserve_budget: int         # held back; consumed on expand
    fetch_k: int                # vector top-N (>= primary + reserve + slack)
    filters: dict = Field(default_factory=dict)
    enabled: bool = True
    route_reason: str = ""


class Chunk(BaseModel):
    chunk_id: str
    document_id: str
    collection_name: str
    doc_type: str
    source_title: str
    text: str
    score: float
    rerank_score: float = 0.0
    metadata: dict = Field(default_factory=dict)


class RetrievalOutput(BaseModel):
    collection_name: str
    chunks: list[Chunk]
    reserve_chunks: list[Chunk] = Field(default_factory=list)
    fetched_count: int
    returned_count: int
    latency_ms: float
    filter_applied: dict = Field(default_factory=dict)


class ContextAssemblyItem(BaseModel):
    chunk_id: str
    collection_name: str
    document_id: str
    slot_type: Literal["primary", "supporting", "diversity", "reserve"]
    assembly_reason: str
    order_index: int


class EvidenceDecision(BaseModel):
    sufficient: bool
    confidence: float
    missing_aspects: list[str] = Field(default_factory=list)
    action: Literal["answer", "expand", "clarify", "refuse"]
    judge_type: Literal["heuristic", "llm"] = "heuristic"


class OrchestratorState(BaseModel):
    request_id: str
    user_query: str
    normalized_query: str = ""

    planner_output: Optional[SearchPlan] = None
    policy_result: Optional[PolicyResult] = None
    collection_plans: list[CollectionExecutionPlan] = Field(default_factory=list)

    retrieval_results: dict[str, RetrievalOutput] = Field(default_factory=dict)

    assembled_chunks: list[Chunk] = Field(default_factory=list)
    balanced_context: list[ContextAssemblyItem] = Field(default_factory=list)

    evidence_decision: Optional[EvidenceDecision] = None
    expanded: bool = False

    final_answer: str = ""
    citations: list[dict] = Field(default_factory=list)
    trace: list[AgentTraceEvent] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
```

### 4.3 Augment `AgentOutput`

```python
class AgentOutput(BaseModel):
    answer: str
    thinking: str = ""
    plan: Optional[SearchPlan] = None
    validation: Optional[ValidationResult] = None
    trace: list[AgentTraceEvent] = Field(default_factory=list)
    sources: list[dict] = Field(default_factory=list)

    # NEW
    policy_result: Optional[PolicyResult] = None
    evidence_decision: Optional[EvidenceDecision] = None
    assembly: list[ContextAssemblyItem] = Field(default_factory=list)
    expanded: bool = False

    # Deprecated (kept one release for client compat; always False under orchestrator)
    re_retrieved: bool = False
    quality_re_retrieved: bool = False
```

## 5. Module Layout

| Path | Action | Responsibility |
|---|---|---|
| `src/agent/schemas.py` | Modify | Add Policy/Allocation/Assembly/Evidence/State models; augment SearchPlan and AgentOutput |
| `src/agent/policy.py` | Create | `PolicyEnforcer.run(state, session_collections)` |
| `src/agent/allocator.py` | Create | `AllocationPlanner.run(state)` reads `AllocationConfig` from YAML |
| `src/agent/assembler.py` | Create | `BalancedContextAssembler.run(state)` — doc-dedup, slot_type, max_per_document, max_total_primary |
| `src/agent/judge.py` | Create | `EvidenceJudge.run(state)` — hybrid heuristic + LLM borderline path |
| `src/agent/expander.py` | Create | `ExpansionPlanner.run(state)` — consume reserves only, no new fetch |
| `src/agent/citations.py` | Create | `CitationBuilder.build(chunks) -> list[dict]` |
| `src/agent/orchestrator.py` | Create | `OrchestratorAgent.run(query, session_collections) -> AgentOutput` |
| `src/agent/planner.py` | Modify | Strip `_needs_reretrieval`, `_needs_quality_reretrieval`, gap-fill prompt, retry loops. Expose `Planner.plan(query) -> SearchPlan` |
| `src/config/pipeline_loader.py` | Modify | Add `PolicyConfig`, `AllocationConfig`, `JudgeConfig`; load from YAML |
| `pipeline.yaml` | Modify | Add `policy:`, `allocation:`, `judge:` blocks |
| `src/generator/service.py` | Modify | Dispatch to `OrchestratorAgent` when `orchestrator.enabled: true`; else legacy `PlanningAgent` |

## 6. Component Specifications

### 6.1 `PolicyEnforcer` (`src/agent/policy.py`)

```python
class PolicyEnforcer:
    def __init__(self, config: PolicyConfig) -> None: ...

    def run(self, state: OrchestratorState, session_collections: list[str]) -> OrchestratorState:
        suggested = [r.collection for r in state.planner_output.resources]
        allowed = [c for c in suggested if c in session_collections]
        denied = [c for c in suggested if c not in session_collections]

        state.policy_result = PolicyResult(
            allowed_collections=allowed,
            denied_collections=denied,
            reason_by_collection={c: "not_in_session_selection" for c in denied},
        )
        if not allowed:
            state.errors.append("policy_no_allowed_collections")
        return state
```

Auth/role-based rules are deferred. When added, they layer onto the same intersection result with reasons.

### 6.2 `AllocationPlanner` (`src/agent/allocator.py`)

Reads `AllocationConfig` (from YAML) and builds one `CollectionExecutionPlan` per allowed collection. Budget lookup: `config.by_query_type.get(state.planner_output.query_type, config.defaults)`.

```python
class AllocationPlanner:
    def __init__(self, config: AllocationConfig) -> None: ...

    def run(self, state: OrchestratorState) -> OrchestratorState:
        if not state.policy_result or not state.policy_result.allowed_collections:
            state.errors.append("allocation_no_allowed_collections")
            return state

        qt = state.planner_output.query_type
        budget = self._config.budget_for(qt)  # {primary, reserve, fetch_k}

        plans = []
        for idx, name in enumerate(state.policy_result.allowed_collections):
            # Resources filters from planner take priority over empty defaults
            filters = self._collect_filters_for(state.planner_output, name)
            plans.append(CollectionExecutionPlan(
                collection_name=name,
                priority=idx + 1,
                retrieval_budget=budget.primary,
                reserve_budget=budget.reserve,
                fetch_k=budget.fetch_k,
                filters=filters,
                route_reason="planner_suggested_and_session_allowed",
            ))
        state.collection_plans = plans
        return state
```

### 6.3 Per-collection retrieval

Reuses `src/agent/tools.py:SearchTool.search()`. The orchestrator dispatches one search per plan (parallel by default via `ThreadPoolExecutor` already used in `PlanningAgent`):

```python
for plan in state.collection_plans:
    result_dict = search_tool.search(
        collection_key=plan.collection_name,
        query_text=state.user_query,        # or planner-drafted text if planner emits one
        filters=plan.filters,
        top_k=plan.fetch_k,
    )
    chunks = _to_chunk_list(result_dict, plan.collection_name)
    primary, reserve = chunks[:plan.retrieval_budget], chunks[plan.retrieval_budget:plan.retrieval_budget + plan.reserve_budget]
    state.retrieval_results[plan.collection_name] = RetrievalOutput(
        collection_name=plan.collection_name,
        chunks=primary,
        reserve_chunks=reserve,
        fetched_count=len(chunks),
        returned_count=len(primary),
        latency_ms=...,
        filter_applied=plan.filters,
    )
```

The reranker (already wired in `SearchTool`) runs before primary/reserve split, so reserves are the next-best-ranked candidates, not random leftovers.

### 6.4 `BalancedContextAssembler` (`src/agent/assembler.py`)

Per-collection iteration in priority order. Tracks `seen_document_ids` globally for cross-collection doc-dedup. Honors `max_per_document` and `max_total_primary` from `AllocationConfig`.

```python
class BalancedContextAssembler:
    def __init__(self, config: AllocationConfig) -> None: ...

    def run(self, state: OrchestratorState) -> OrchestratorState:
        seen_doc_ids: set[str] = set()
        per_doc_count: dict[str, int] = {}
        assembled: list[Chunk] = []
        items: list[ContextAssemblyItem] = []
        total = 0

        for plan in sorted(state.collection_plans, key=lambda p: p.priority):
            rr = state.retrieval_results.get(plan.collection_name)
            if not rr:
                continue

            taken = 0
            for chunk in rr.chunks:
                if total >= self._config.max_total_primary:
                    break
                if per_doc_count.get(chunk.document_id, 0) >= self._config.max_per_document:
                    continue
                if taken >= plan.retrieval_budget:
                    break

                assembled.append(chunk)
                items.append(ContextAssemblyItem(
                    chunk_id=chunk.chunk_id,
                    collection_name=chunk.collection_name,
                    document_id=chunk.document_id,
                    slot_type="primary",
                    assembly_reason="collection_budget_fill",
                    order_index=len(items),
                ))
                seen_doc_ids.add(chunk.document_id)
                per_doc_count[chunk.document_id] = per_doc_count.get(chunk.document_id, 0) + 1
                taken += 1
                total += 1

        state.assembled_chunks = assembled
        state.balanced_context = items
        return state
```

### 6.5 `EvidenceJudge` (`src/agent/judge.py`)

Hybrid: heuristic first, LLM only in borderline band.

```python
class EvidenceJudge:
    def __init__(self, config: JudgeConfig, client_pool: LLMClientPool) -> None: ...

    def run(self, state: OrchestratorState) -> OrchestratorState:
        chunks = state.assembled_chunks
        coverage = len({c.collection_name for c in chunks})
        h = self._config.heuristic

        if len(chunks) == 0:
            state.evidence_decision = EvidenceDecision(
                sufficient=False, confidence=0.0, action="clarify",
                missing_aspects=["no_results"], judge_type="heuristic",
            )
            return state

        if len(chunks) >= h.min_chunks and coverage >= h.min_collection_coverage:
            state.evidence_decision = EvidenceDecision(
                sufficient=True, confidence=0.85, action="answer", judge_type="heuristic",
            )
            return state

        llm = self._config.llm
        in_band = llm.borderline_band[0] <= len(chunks) <= llm.borderline_band[1]
        if llm.enabled and in_band:
            state.evidence_decision = self._llm_judge(state)
            return state

        state.evidence_decision = EvidenceDecision(
            sufficient=False, confidence=0.4, action="expand",
            missing_aspects=["insufficient_chunks"], judge_type="heuristic",
        )
        return state
```

**LLM judge prompt** (Turkish, returns JSON):

> Sen bir kanıt yeterlilik değerlendirme uzmanısın. Aşağıdaki soruya verilen bağlam parçaları yeterli mi?
> Soru: `{query}`
> Niyet: `{intent}` / Sorgu tipi: `{query_type}`
> Bağlam (özet):
> ```
> [1] (koleksiyon/doc) {text_excerpt}
> [2] ...
> ```
> Çıktı (JSON):
> `{"sufficient": bool, "confidence": float, "action": "answer|expand|clarify|refuse", "missing_aspects": ["..."]}`

Model: small/fast model (`fast-01` block, `model_key: judge`). Timeout 5s; on timeout fall back to heuristic `expand` for borderline cases.

### 6.6 `ExpansionPlanner` (`src/agent/expander.py`)

Consumes `reserve_chunks` from each `RetrievalOutput`. No new vector calls.

```python
class ExpansionPlanner:
    def run(self, state: OrchestratorState) -> OrchestratorState:
        if not state.evidence_decision or state.evidence_decision.action != "expand":
            return state

        current_doc_ids = {c.document_id for c in state.assembled_chunks}
        added_total = 0

        for plan in sorted(state.collection_plans, key=lambda p: p.priority):
            rr = state.retrieval_results.get(plan.collection_name)
            if not rr:
                continue
            added = 0
            for chunk in rr.reserve_chunks:
                if chunk.document_id in current_doc_ids:
                    continue
                if added >= plan.reserve_budget:
                    break
                state.assembled_chunks.append(chunk)
                state.balanced_context.append(ContextAssemblyItem(
                    chunk_id=chunk.chunk_id,
                    collection_name=chunk.collection_name,
                    document_id=chunk.document_id,
                    slot_type="reserve",
                    assembly_reason="evidence_expansion",
                    order_index=len(state.balanced_context),
                ))
                current_doc_ids.add(chunk.document_id)
                added += 1
                added_total += 1
        state.expanded = added_total > 0
        return state
```

Max one expansion iteration (controlled by `judge.max_expand_iterations`); after expansion, re-run `EvidenceJudge` once. If still insufficient → action becomes `clarify` (LLM judge) or proceed to answer with low-confidence flag (heuristic).

### 6.7 `CitationBuilder` (`src/agent/citations.py`)

```python
class CitationBuilder:
    @staticmethod
    def build(chunks: list[Chunk]) -> list[dict]:
        return [
            {
                "index": i + 1,
                "collection_name": c.collection_name,
                "document_id": c.document_id,
                "chunk_id": c.chunk_id,
                "source_title": c.source_title,
                "doc_type": c.doc_type,
                "metadata": c.metadata,
            }
            for i, c in enumerate(chunks)
        ]
```

### 6.8 `OrchestratorAgent` (`src/agent/orchestrator.py`)

```python
class OrchestratorAgent:
    def __init__(self, config: PipelineConfig, client_pool: LLMClientPool) -> None:
        self._planner = Planner(config, client_pool)
        self._policy = PolicyEnforcer(config.policy)
        self._allocator = AllocationPlanner(config.allocation)
        self._search_tool = SearchTool(config, client_pool)
        self._assembler = BalancedContextAssembler(config.allocation)
        self._judge = EvidenceJudge(config.judge, client_pool)
        self._expander = ExpansionPlanner()
        self._answer_tool = AnswerTool(config, client_pool)
        self._sanitizer = SanitizerAgent(client_pool, config)
        self._tracer_factory = lambda req_id: PipelineTracer(req_id)

    def run(self, query: str, session_collections: list[str], stream_callback=None) -> AgentOutput:
        state = OrchestratorState(request_id=_uuid(), user_query=query)
        tracer = self._tracer_factory(state.request_id)

        # 1. Plan
        state.planner_output = self._planner.plan(query, tracer)

        # 2. Policy
        self._policy.run(state, session_collections)
        if not state.policy_result.allowed_collections:
            return self._make_refuse(state, "no_allowed_collections")

        # 3. Allocate
        self._allocator.run(state)

        # 4. Retrieve (parallel)
        self._retrieve_all(state, tracer)

        # 5. Assemble
        self._assembler.run(state)

        # 6. Judge
        self._judge.run(state)

        # 7. Expand once if needed, then re-judge
        if state.evidence_decision.action == "expand":
            self._expander.run(state)
            self._judge.run(state)  # second judgement; action may flip to answer or clarify

        # 8. Dispatch
        action = state.evidence_decision.action
        if action == "clarify":
            return self._make_clarify(state)
        if action == "refuse":
            return self._make_refuse(state, "judge_refuse")

        # action ∈ {answer, expand-then-answer-with-low-confidence}
        state.final_answer, validation = self._generate_and_validate(state, tracer, stream_callback)
        state.citations = CitationBuilder.build(state.assembled_chunks)

        return AgentOutput(
            answer=state.final_answer,
            plan=state.planner_output,
            validation=validation,
            trace=state.trace,
            sources=state.citations,
            policy_result=state.policy_result,
            evidence_decision=state.evidence_decision,
            assembly=state.balanced_context,
            expanded=state.expanded,
        )
```

The answering step calls `AnswerTool.stream()` (existing) then `SanitizerAgent.validate()` (existing). If sanitizer says fail and has `corrected_answer`, swap it in (existing behavior). No `quality_re_retrieval` re-fetch loop.

## 7. `pipeline.yaml` Additions

```yaml
orchestrator:
  enabled: true       # feature flag; false => legacy PlanningAgent.process()

policy:
  mode: session_intersection   # intersect session selection with planner suggestions
  # auth_rules: deferred

allocation:
  defaults: { primary: 2, reserve: 2, fetch_k: 10 }
  by_query_type:
    fact:       { primary: 2, reserve: 2, fetch_k: 10 }
    summary:    { primary: 3, reserve: 2, fetch_k: 10 }
    comparison: { primary: 3, reserve: 2, fetch_k: 12 }
    reasoning:  { primary: 3, reserve: 2, fetch_k: 12 }
    policy:     { primary: 3, reserve: 3, fetch_k: 15 }
  max_per_document: 1
  max_total_primary: 12

judge:
  mode: hybrid
  heuristic:
    min_chunks: 4
    min_collection_coverage: 2
    min_rerank_score: 0.0
  llm:
    enabled: true
    block: fast-01
    model_key: judge
    borderline_band: [2, 4]
    max_borderline_score_floor: 0.35
    timeout_seconds: 5
  max_expand_iterations: 1
  on_low_confidence: expand     # else: clarify | refuse
```

`PipelineConfig` gains `policy: PolicyConfig`, `allocation: AllocationConfig`, `judge: JudgeConfig` attributes loaded from these blocks. Each config class lives in `src/config/pipeline_loader.py` and follows the existing `PlannerConfig`/`SanitizerConfig` pattern.

## 8. Streaming and UX

- Heuristic judge path: <50ms, transparent to user.
- LLM judge path (borderline only): show chat spinner `Kısa değerlendirme...` until decision returns.
- Expand path: show `Kanıt genişletiliyor...`, run reserve fetch + re-judge, then stream answer.
- Clarify path: print clarify message (no stream), e.g. `Sorunuzu netleştirir misiniz? Eksik bilgi: {missing_aspects}`.
- Refuse path: print refuse message, e.g. `Yetkili kaynaklarla yanıt veremiyorum.`.
- Answer path: stream tokens via `AnswerTool.stream()`, then append citations block.

## 9. Integration with `RAGService` and Chat

`src/generator/service.py:RAGService` gains a feature-flag branch:

```python
def ask_stream(self, query: str, session_collections: list[str], ...):
    if self._config.orchestrator.enabled:
        return self._orchestrator.run(query, session_collections, stream_callback=...)
    return self._legacy_planner.process(query, ...)   # existing path
```

`src/ui/chat.py` already exposes session-selected collections via the collection selector. It passes that list into `ask_stream()`. No UI changes beyond the two new spinners.

## 10. Testing Strategy

| Test file | Coverage |
|---|---|
| `tests/test_policy.py` | session-intersection, empty allowed list, deny reasons populated |
| `tests/test_allocator.py` | YAML budget lookup per query_type, default fallback, filter inheritance from SearchPlan |
| `tests/test_assembler.py` | doc-dedup across collections, slot_type assignment, max_per_document, max_total_primary, priority ordering |
| `tests/test_evidence_judge.py` | heuristic pass/fail bands, borderline LLM path (mocked LLM), action mapping for each input shape |
| `tests/test_expander.py` | reserve consumed in order, no new fetch invoked, current_doc_ids tracked, expansion idempotent on second invocation |
| `tests/test_orchestrator.py` | end-to-end with mocked SearchTool + LLM clients: happy path, one expand cycle, clarify path, refuse path, post-answer sanitizer rewrite still works |
| Modified: `tests/test_planner.py` | Planner.plan() returns SearchPlan only; remove tests for `_needs_reretrieval`, `_needs_quality_reretrieval`, gap-fill prompt |
| Untouched: `tests/test_vector_retriever.py`, `tests/test_minutes_*.py`, `tests/test_query_routing.py` |

Add `tests/fixtures/orchestrator/` for canned `SearchPlan`, `RetrievalResult`, and judge-prompt LLM responses.

## 11. Migration / Rollout

1. **Add infra without flipping flag.** Land schemas, new modules, YAML keys with `orchestrator.enabled: false`. All new tests pass; existing tests unchanged.
2. **Implement components in order:** Policy → Allocator → Assembler → Judge (heuristic only) → Expander → Citations → Orchestrator (without LLM judge).
3. **Wire `RAGService` dispatch behind the flag.** Validate manually in chat with flag on for a single session.
4. **Add LLM judge path.** Enable `judge.llm.enabled: true`. Tune `borderline_band` from evaluator results.
5. **Run evaluator goldens** (`src/evaluator/`) with flag on vs off; compare retrieval metrics, latency, judge action distribution.
6. **Flip default to `orchestrator.enabled: true`** in `pipeline.yaml` once goldens pass.
7. **Deprecation release:** mark `re_retrieval`, `quality_re_retrieval` YAML keys as deprecated; warn on load. Keep `PlanningAgent.process()` body for one release.
8. **Removal release:** delete `PlanningAgent.process()` body, alias to `OrchestratorAgent.run()`; drop deprecated YAML keys and `re_retrieved` / `quality_re_retrieved` fields from `AgentOutput`.

## 12. Error Handling

- Empty `policy_result.allowed_collections` → return refuse with reason; never call retrieve.
- Empty `assembled_chunks` after assembly → judge emits `clarify` (heuristic, no LLM call).
- LLM judge timeout → fall back to heuristic `expand` for borderline; if expand impossible (no reserves), emit `clarify`.
- Search failure for one collection → log error in `state.errors`, continue with remaining collections.
- Reranker failure → log, fall back to raw vector-distance order (existing `SearchTool` behavior).
- Sanitizer failure → existing behavior preserved (single rewrite attempt; if still fail, return original answer with `validation.passes = False`).

## 13. Tracing

Each stage emits one `AgentTraceEvent`:

| Phase | Details payload |
|---|---|
| `planning` | `{intent, query_type, suggested_collections, latency_ms}` |
| `policy` | `{allowed, denied, reason_by_collection}` |
| `allocation` | `{plans: [{collection, primary, reserve, fetch_k}], query_type}` |
| `retrieval` | `{per_collection: {name: {fetched, returned, latency_ms}}}` |
| `assembly` | `{primary_count, doc_dedupe_drops, slot_distribution}` |
| `judge` | `{judge_type, action, confidence, missing_aspects, chunk_count, coverage}` |
| `expansion` | `{added_count, per_collection_added}` |
| `judge_post_expand` | same as `judge` |
| `answering` | `{model, latency_ms, token_count, stream}` |
| `validation` | existing sanitizer trace |
| `citation` | `{citation_count}` |

Trace events accumulate on `OrchestratorState.trace` and are surfaced in `AgentOutput.trace`.

## 14. Guardrails (out of scope; see separate spec)

Input/output safety, PII redaction, prompt-injection detection, refuse-action trigger taxonomy, citation grounding strictness, and rate limiting are designed in a sibling spec: `docs/superpowers/specs/2026-05-24-guardrails-design.md`. Hooks land in this orchestrator pipeline at four insertion points:

- Pre-Planner: input guard (injection / off-topic gate)
- Pre-Answer: refuse trigger taxonomy consumed by `EvidenceJudge`
- Post-Answer (pre-citation): output guard (PII redaction, toxicity)
- Post-Sanitizer: citation grounding hard rule

This spec leaves `EvidenceDecision.action == "refuse"` reachable but underspecified; the guardrails spec defines the concrete triggers.

## 15. Open Questions (resolve during implementation)

- Should `query_type` be emitted by the existing planner LLM, or derived heuristically from `intent` + query string features? Default: ask the LLM (one extra JSON field); fallback to heuristic `fact`.
- LLM judge model: reuse the sanitizer/answer model, or a dedicated smaller model? Default: try the answer model first; switch if latency budget broken.
- Should `max_total_primary` adapt to context-window remaining after system prompt + query? Default: static for now; revisit after eval.
