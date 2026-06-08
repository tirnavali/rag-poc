# Agentic Orchestrator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the existing `PlanningAgent`'s retrieval-side retry loops with an explicit orchestrator pipeline: Planner → Policy → Allocation → per-collection Retrieve → Balanced Assembly → Evidence Judge → (Expand | Answer | Clarify | Refuse) → Citation. Sanitizer kept as orthogonal post-answer text validator.

**Architecture:** New `OrchestratorAgent` in `src/agent/orchestrator.py` runs a state machine over `OrchestratorState`. Existing `SearchPlan`, `SearchTool`, `AnswerTool`, `SanitizerAgent`, `PipelineTracer`, `LLMClientPool` are reused. New per-stage modules under `src/agent/`: `policy.py`, `allocator.py`, `assembler.py`, `judge.py`, `expander.py`, `citations.py`. Config additions in `pipeline.yaml` (`policy:`, `allocation:`, `judge:`, `orchestrator:`) and `src/config/pipeline_loader.py`. Old path stays behind `orchestrator.enabled: false` flag for one release.

**Tech Stack:** Python 3.11+, Pydantic v2, pytest, PyYAML, existing `LLMClientPool` (Ollama HTTP), ChromaDB embedded.

---

## File Structure

| Path | Action | Responsibility |
|---|---|---|
| `src/agent/schemas.py` | Modify | Add `PolicyResult`, `CollectionExecutionPlan`, `Chunk`, `RetrievalOutput`, `ContextAssemblyItem`, `EvidenceDecision`, `OrchestratorState`; augment `SearchPlan` (`query_type`) and `AgentOutput` |
| `src/config/pipeline_loader.py` | Modify | Add `PolicyConfig`, `AllocationConfig`, `JudgeConfig`, `OrchestratorConfig`; wire into `PipelineConfig.__init__` |
| `pipeline.yaml` | Modify | Add `orchestrator:`, `policy:`, `allocation:`, `judge:` blocks |
| `src/agent/policy.py` | Create | `PolicyEnforcer.run(state, session_collections) -> OrchestratorState` |
| `src/agent/allocator.py` | Create | `AllocationPlanner.run(state) -> OrchestratorState` |
| `src/agent/assembler.py` | Create | `BalancedContextAssembler.run(state) -> OrchestratorState` |
| `src/agent/judge.py` | Create | `EvidenceJudge.run(state) -> OrchestratorState` — heuristic + optional LLM |
| `src/agent/expander.py` | Create | `ExpansionPlanner.run(state) -> OrchestratorState` |
| `src/agent/citations.py` | Create | `CitationBuilder.build(chunks) -> list[dict]` |
| `src/agent/orchestrator.py` | Create | `OrchestratorAgent.run(query, session_collections) -> AgentOutput` |
| `src/agent/planner.py` | Modify | Extract `Planner.plan(query) -> SearchPlan`; strip `_needs_reretrieval`, `_needs_quality_reretrieval`, `_generate_broader_plan`, `_generate_gap_fill_plan`, retry loops from `PlanningAgent.run()` (kept for legacy path) |
| `src/generator/service.py` | Modify | Dispatch to `OrchestratorAgent` when `config.orchestrator.enabled`; legacy `PlanningAgent` otherwise |
| `tests/test_orchestrator_schemas.py` | Create | Unit tests for new Pydantic models |
| `tests/test_pipeline_loader_orchestrator.py` | Create | Unit tests for new config classes |
| `tests/test_policy.py` | Create | `PolicyEnforcer` cases |
| `tests/test_allocator.py` | Create | `AllocationPlanner` cases |
| `tests/test_assembler.py` | Create | `BalancedContextAssembler` cases |
| `tests/test_evidence_judge.py` | Create | `EvidenceJudge` heuristic + mocked LLM |
| `tests/test_expander.py` | Create | `ExpansionPlanner` reserve consumption |
| `tests/test_citations.py` | Create | `CitationBuilder.build()` shape |
| `tests/test_orchestrator.py` | Create | End-to-end with mocked SearchTool + LLM |
| `tests/test_service_dispatch.py` | Create | `RAGService` flag branching |

---

## Pre-flight Checks

- [ ] **Step P1: Verify baseline tests green on current branch**

Run: `python -m pytest tests/ -x -q`
Expected: all pass (or note known failures in a comment before starting).

- [ ] **Step P2: Verify branch is `feature/agent-pipeline`**

Run: `git branch --show-current`
Expected: `feature/agent-pipeline`. If not, stop and ask.

- [ ] **Step P3: Verify spec is present**

Run: `ls docs/superpowers/specs/2026-05-24-agentic-orchestrator-design.md`
Expected: file exists.

---

## Task 1: Augment `SearchPlan` with `query_type`

**Files:**
- Modify: `src/agent/schemas.py`
- Test: `tests/test_orchestrator_schemas.py`

- [ ] **Step 1.1: Write failing test for `SearchPlan.query_type` default**

Create `tests/test_orchestrator_schemas.py` with:

```python
"""Unit tests for orchestrator-related Pydantic schemas."""
from __future__ import annotations

import pytest

from src.agent.schemas import SearchPlan, CollectionSearchPlan, SearchQueryDraft


def test_search_plan_query_type_defaults_to_fact():
    plan = SearchPlan(
        intent="factual",
        resources=[
            CollectionSearchPlan(
                collection="gazete_arsivi",
                query_drafts=[SearchQueryDraft(text="test")],
            )
        ],
        reasoning="r",
    )
    assert plan.query_type == "fact"


def test_search_plan_query_type_accepts_each_literal():
    for qt in ("fact", "summary", "comparison", "reasoning", "policy"):
        plan = SearchPlan(
            intent="unknown",
            query_type=qt,
            resources=[
                CollectionSearchPlan(
                    collection="c", query_drafts=[SearchQueryDraft(text="t")]
                )
            ],
            reasoning="r",
        )
        assert plan.query_type == qt


def test_search_plan_query_type_rejects_other():
    with pytest.raises(Exception):  # ValidationError
        SearchPlan(
            intent="unknown",
            query_type="invalid_type",
            resources=[
                CollectionSearchPlan(
                    collection="c", query_drafts=[SearchQueryDraft(text="t")]
                )
            ],
            reasoning="r",
        )
```

- [ ] **Step 1.2: Run test, verify it fails**

Run: `pytest tests/test_orchestrator_schemas.py::test_search_plan_query_type_defaults_to_fact -v`
Expected: FAIL — `query_type` not a field on `SearchPlan`.

- [ ] **Step 1.3: Add `query_type` to `SearchPlan` in `src/agent/schemas.py`**

In `src/agent/schemas.py`, locate the existing `SearchPlan` class and add the `query_type` field between `intent` and `resources`:

```python
class SearchPlan(BaseModel):
    """Complete search plan generated by the Planning Agent."""
    intent: Literal["factual", "comparative", "analytical", "temporal", "unknown"] = Field(
        ..., description="Query intent classification"
    )
    query_type: Literal["fact", "summary", "comparison", "reasoning", "policy"] = Field(
        "fact", description="Query type drives allocator budgets"
    )
    resources: list[CollectionSearchPlan] = Field(
        ..., description="Collections to search with query drafts"
    )
    reasoning: str = Field(..., description="Why this plan was chosen")
```

- [ ] **Step 1.4: Run all three new tests, verify they pass**

Run: `pytest tests/test_orchestrator_schemas.py -v`
Expected: 3 passed.

- [ ] **Step 1.5: Commit**

```bash
git add src/agent/schemas.py tests/test_orchestrator_schemas.py
git commit -m "feat(schemas): add query_type literal field to SearchPlan"
```

---

## Task 2: Add `Chunk` and `RetrievalOutput` models

**Files:**
- Modify: `src/agent/schemas.py`
- Test: `tests/test_orchestrator_schemas.py`

- [ ] **Step 2.1: Append failing tests for Chunk + RetrievalOutput**

Append to `tests/test_orchestrator_schemas.py`:

```python
from src.agent.schemas import Chunk, RetrievalOutput


def test_chunk_required_fields():
    c = Chunk(
        chunk_id="c1",
        document_id="d1",
        collection_name="col",
        doc_type="gazete",
        source_title="t",
        text="body",
        score=0.5,
    )
    assert c.rerank_score == 0.0
    assert c.metadata == {}


def test_retrieval_output_defaults_reserve_empty():
    ro = RetrievalOutput(
        collection_name="c",
        chunks=[],
        fetched_count=0,
        returned_count=0,
        latency_ms=0.0,
    )
    assert ro.reserve_chunks == []
    assert ro.filter_applied == {}
```

- [ ] **Step 2.2: Run tests, verify they fail**

Run: `pytest tests/test_orchestrator_schemas.py::test_chunk_required_fields -v`
Expected: FAIL — `Chunk` not importable.

- [ ] **Step 2.3: Add `Chunk` and `RetrievalOutput` to `src/agent/schemas.py`**

Append after the existing `AgentOutput` class:

```python
class Chunk(BaseModel):
    """A retrieved chunk attributed to its source collection."""
    chunk_id: str
    document_id: str
    collection_name: str
    doc_type: str
    source_title: str
    text: str
    score: float
    rerank_score: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievalOutput(BaseModel):
    """Per-collection retrieval result with primary and reserve buffers."""
    collection_name: str
    chunks: list[Chunk] = Field(default_factory=list)
    reserve_chunks: list[Chunk] = Field(default_factory=list)
    fetched_count: int
    returned_count: int
    latency_ms: float
    filter_applied: dict[str, Any] = Field(default_factory=dict)
```

- [ ] **Step 2.4: Run new tests, verify pass**

Run: `pytest tests/test_orchestrator_schemas.py -v`
Expected: 5 passed.

- [ ] **Step 2.5: Commit**

```bash
git add src/agent/schemas.py tests/test_orchestrator_schemas.py
git commit -m "feat(schemas): add Chunk and RetrievalOutput Pydantic models"
```

---

## Task 3: Add `PolicyResult`, `CollectionExecutionPlan`, `ContextAssemblyItem`, `EvidenceDecision`

**Files:**
- Modify: `src/agent/schemas.py`
- Test: `tests/test_orchestrator_schemas.py`

- [ ] **Step 3.1: Append failing tests**

Append to `tests/test_orchestrator_schemas.py`:

```python
from src.agent.schemas import (
    CollectionExecutionPlan,
    ContextAssemblyItem,
    EvidenceDecision,
    PolicyResult,
)


def test_policy_result_defaults():
    p = PolicyResult(allowed_collections=["a", "b"])
    assert p.denied_collections == []
    assert p.reason_by_collection == {}


def test_collection_execution_plan_required():
    cep = CollectionExecutionPlan(
        collection_name="c",
        retrieval_budget=2,
        reserve_budget=2,
        fetch_k=10,
    )
    assert cep.priority == 1
    assert cep.enabled is True
    assert cep.filters == {}


def test_context_assembly_item_slot_type_literal():
    ca = ContextAssemblyItem(
        chunk_id="x",
        collection_name="c",
        document_id="d",
        slot_type="primary",
        assembly_reason="r",
        order_index=0,
    )
    assert ca.slot_type == "primary"


def test_evidence_decision_required_action():
    ed = EvidenceDecision(sufficient=True, confidence=0.9, action="answer")
    assert ed.judge_type == "heuristic"
    assert ed.missing_aspects == []
```

- [ ] **Step 3.2: Run failing tests**

Run: `pytest tests/test_orchestrator_schemas.py -v`
Expected: 4 new failures (`PolicyResult`/`CollectionExecutionPlan`/`ContextAssemblyItem`/`EvidenceDecision` not importable).

- [ ] **Step 3.3: Add the four models in `src/agent/schemas.py`**

Append after `RetrievalOutput`:

```python
class PolicyResult(BaseModel):
    """Result of policy enforcement on planner-suggested collections."""
    allowed_collections: list[str]
    denied_collections: list[str] = Field(default_factory=list)
    reason_by_collection: dict[str, str] = Field(default_factory=dict)


class CollectionExecutionPlan(BaseModel):
    """Per-collection execution plan with primary + reserve budgets."""
    collection_name: str
    priority: int = 1
    retrieval_budget: int
    reserve_budget: int
    fetch_k: int
    filters: dict[str, Any] = Field(default_factory=dict)
    enabled: bool = True
    route_reason: str = ""


class ContextAssemblyItem(BaseModel):
    """One slot in the assembled context with provenance."""
    chunk_id: str
    collection_name: str
    document_id: str
    slot_type: Literal["primary", "supporting", "diversity", "reserve"]
    assembly_reason: str
    order_index: int


class EvidenceDecision(BaseModel):
    """EvidenceJudge output controlling next pipeline step."""
    sufficient: bool
    confidence: float
    missing_aspects: list[str] = Field(default_factory=list)
    action: Literal["answer", "expand", "clarify", "refuse"]
    judge_type: Literal["heuristic", "llm"] = "heuristic"
```

- [ ] **Step 3.4: Run all tests, verify pass**

Run: `pytest tests/test_orchestrator_schemas.py -v`
Expected: 9 passed.

- [ ] **Step 3.5: Commit**

```bash
git add src/agent/schemas.py tests/test_orchestrator_schemas.py
git commit -m "feat(schemas): add Policy/Allocation/Assembly/Evidence Pydantic models"
```

---

## Task 4: Add `OrchestratorState` and augment `AgentOutput`

**Files:**
- Modify: `src/agent/schemas.py`
- Test: `tests/test_orchestrator_schemas.py`

- [ ] **Step 4.1: Append failing tests**

Append to `tests/test_orchestrator_schemas.py`:

```python
from src.agent.schemas import OrchestratorState, AgentOutput


def test_orchestrator_state_minimal():
    st = OrchestratorState(request_id="r1", user_query="q")
    assert st.normalized_query == ""
    assert st.planner_output is None
    assert st.policy_result is None
    assert st.collection_plans == []
    assert st.retrieval_results == {}
    assert st.assembled_chunks == []
    assert st.balanced_context == []
    assert st.evidence_decision is None
    assert st.expanded is False
    assert st.final_answer == ""
    assert st.citations == []
    assert st.trace == []
    assert st.errors == []


def test_agent_output_new_optional_fields():
    out = AgentOutput(answer="a")
    assert out.policy_result is None
    assert out.evidence_decision is None
    assert out.assembly == []
    assert out.expanded is False
    assert out.re_retrieved is False  # deprecated but still present
    assert out.quality_re_retrieved is False
```

- [ ] **Step 4.2: Run failing tests**

Run: `pytest tests/test_orchestrator_schemas.py -v -k "orchestrator_state or new_optional"`
Expected: 2 failures.

- [ ] **Step 4.3: Add `OrchestratorState` and augment `AgentOutput` in `src/agent/schemas.py`**

Append after `EvidenceDecision`:

```python
class OrchestratorState(BaseModel):
    """Full state passed across orchestrator stages."""
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
    citations: list[dict[str, Any]] = Field(default_factory=list)
    trace: list[AgentTraceEvent] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
```

In the existing `AgentOutput` class, add the new fields after `re_retrieved` and `quality_re_retrieved`:

```python
class AgentOutput(BaseModel):
    """Final output from the Planning Agent pipeline."""
    answer: str = Field(..., description="Generated answer text")
    thinking: str = Field(default="", description="LLM thinking/reasoning text")
    plan: Optional[SearchPlan] = Field(None, description="The search plan that was executed")
    validation: Optional[ValidationResult] = Field(None, description="Output validation result")
    trace: list[AgentTraceEvent] = Field(default_factory=list)
    sources: list[dict] = Field(default_factory=list)
    re_retrieved: bool = Field(False, description="Deprecated; always False under orchestrator path")
    quality_re_retrieved: bool = Field(False, description="Deprecated; always False under orchestrator path")

    # New orchestrator fields (all optional for backwards-compat with legacy path)
    policy_result: Optional[PolicyResult] = None
    evidence_decision: Optional[EvidenceDecision] = None
    assembly: list[ContextAssemblyItem] = Field(default_factory=list)
    expanded: bool = False
```

- [ ] **Step 4.4: Run all schema tests, verify pass**

Run: `pytest tests/test_orchestrator_schemas.py -v`
Expected: 11 passed.

- [ ] **Step 4.5: Run full suite, ensure no regressions**

Run: `python -m pytest tests/ -x -q`
Expected: all green (existing tests unaffected by new optional fields).

- [ ] **Step 4.6: Commit**

```bash
git add src/agent/schemas.py tests/test_orchestrator_schemas.py
git commit -m "feat(schemas): add OrchestratorState and augment AgentOutput"
```

---

## Task 5: Add config classes to `pipeline_loader.py`

**Files:**
- Modify: `src/config/pipeline_loader.py`
- Test: `tests/test_pipeline_loader_orchestrator.py`

- [ ] **Step 5.1: Write failing tests**

Create `tests/test_pipeline_loader_orchestrator.py`:

```python
"""Unit tests for orchestrator-related config classes."""
from __future__ import annotations

import pytest

from src.config.pipeline_loader import (
    AllocationConfig,
    JudgeConfig,
    OrchestratorConfig,
    PolicyConfig,
)


def test_orchestrator_config_disabled_by_default():
    oc = OrchestratorConfig({})
    assert oc.enabled is False


def test_orchestrator_config_enabled_when_set():
    oc = OrchestratorConfig({"enabled": True})
    assert oc.enabled is True


def test_policy_config_defaults():
    pc = PolicyConfig({})
    assert pc.mode == "session_intersection"


def test_allocation_config_defaults_and_query_type_lookup():
    ac = AllocationConfig({
        "defaults": {"primary": 2, "reserve": 2, "fetch_k": 10},
        "by_query_type": {
            "comparison": {"primary": 3, "reserve": 2, "fetch_k": 12},
        },
        "max_per_document": 1,
        "max_total_primary": 12,
    })
    assert ac.max_per_document == 1
    assert ac.max_total_primary == 12
    # explicit query_type
    b = ac.budget_for("comparison")
    assert b.primary == 3 and b.reserve == 2 and b.fetch_k == 12
    # missing query_type → defaults
    b2 = ac.budget_for("fact")
    assert b2.primary == 2 and b2.reserve == 2 and b2.fetch_k == 10


def test_judge_config_defaults():
    jc = JudgeConfig({})
    assert jc.heuristic.min_chunks == 4
    assert jc.heuristic.min_collection_coverage == 2
    assert jc.llm.enabled is True
    assert jc.llm.borderline_band == (2, 4)
    assert jc.max_expand_iterations == 1
```

- [ ] **Step 5.2: Run failing tests**

Run: `pytest tests/test_pipeline_loader_orchestrator.py -v`
Expected: 5 ImportErrors (classes not present).

- [ ] **Step 5.3: Add config classes to `src/config/pipeline_loader.py`**

In `src/config/pipeline_loader.py`, before the `class PipelineConfig:` declaration, add:

```python
class OrchestratorConfig:
    """Feature flag for new orchestrator pipeline."""

    def __init__(self, config: dict) -> None:
        self.enabled = bool(config.get("enabled", False))


class PolicyConfig:
    """Collection-access policy configuration."""

    def __init__(self, config: dict) -> None:
        self.mode = config.get("mode", "session_intersection")


class _AllocationBudget:
    """Tuple-like budget triple for one query_type."""

    def __init__(self, primary: int, reserve: int, fetch_k: int) -> None:
        self.primary = primary
        self.reserve = reserve
        self.fetch_k = fetch_k


class AllocationConfig:
    """Per-query-type retrieval budget configuration."""

    def __init__(self, config: dict) -> None:
        defaults = config.get("defaults", {})
        self._defaults = _AllocationBudget(
            primary=int(defaults.get("primary", 2)),
            reserve=int(defaults.get("reserve", 2)),
            fetch_k=int(defaults.get("fetch_k", 10)),
        )
        raw_by_qt = config.get("by_query_type", {})
        self._by_query_type: dict[str, _AllocationBudget] = {}
        for qt, cfg in raw_by_qt.items():
            self._by_query_type[qt] = _AllocationBudget(
                primary=int(cfg.get("primary", self._defaults.primary)),
                reserve=int(cfg.get("reserve", self._defaults.reserve)),
                fetch_k=int(cfg.get("fetch_k", self._defaults.fetch_k)),
            )
        self.max_per_document = int(config.get("max_per_document", 1))
        self.max_total_primary = int(config.get("max_total_primary", 12))

    def budget_for(self, query_type: str) -> _AllocationBudget:
        return self._by_query_type.get(query_type, self._defaults)


class _JudgeHeuristicConfig:
    def __init__(self, config: dict) -> None:
        self.min_chunks = int(config.get("min_chunks", 4))
        self.min_collection_coverage = int(config.get("min_collection_coverage", 2))
        self.min_rerank_score = float(config.get("min_rerank_score", 0.0))


class _JudgeLLMConfig:
    def __init__(self, config: dict) -> None:
        self.enabled = bool(config.get("enabled", True))
        self.block = config.get("block", "fast-01")
        self.model_key = config.get("model_key", "judge")
        band = config.get("borderline_band", [2, 4])
        self.borderline_band: tuple[int, int] = (int(band[0]), int(band[1]))
        self.max_borderline_score_floor = float(config.get("max_borderline_score_floor", 0.35))
        self.timeout_seconds = int(config.get("timeout_seconds", 5))


class JudgeConfig:
    """EvidenceJudge configuration (hybrid heuristic + LLM)."""

    def __init__(self, config: dict) -> None:
        self.mode = config.get("mode", "hybrid")
        self.heuristic = _JudgeHeuristicConfig(config.get("heuristic", {}))
        self.llm = _JudgeLLMConfig(config.get("llm", {}))
        self.max_expand_iterations = int(config.get("max_expand_iterations", 1))
        self.on_low_confidence = config.get("on_low_confidence", "expand")
```

Then update `PipelineConfig.__init__` (around line 102) to load the new sections. Replace the existing body with:

```python
class PipelineConfig:
    """Top-level pipeline configuration loaded from YAML."""

    def __init__(self, config: dict) -> None:
        blocks = config.get("deployment_blocks", {})
        self.blocks: dict[str, DeploymentBlock] = {
            name: DeploymentBlock(name, cfg) for name, cfg in blocks.items()
        }

        agent_cfg = config.get("agent", {})
        self.planner = PlannerConfig(agent_cfg.get("planner", {}))
        self.answering = AgentConfig(agent_cfg.get("answering", {}))
        self.sanitizer = AgentConfig(agent_cfg.get("sanitizer", {}))

        self.retrieval = RetrievalConfig(config.get("retrieval", {}))

        # New orchestrator blocks (optional; safe defaults when missing)
        self.orchestrator = OrchestratorConfig(config.get("orchestrator", {}))
        self.policy = PolicyConfig(config.get("policy", {}))
        self.allocation = AllocationConfig(config.get("allocation", {}))
        self.judge = JudgeConfig(config.get("judge", {}))
```

- [ ] **Step 5.4: Run config tests, verify pass**

Run: `pytest tests/test_pipeline_loader_orchestrator.py -v`
Expected: 5 passed.

- [ ] **Step 5.5: Run full suite, ensure no regressions**

Run: `python -m pytest tests/ -x -q`
Expected: all green (existing `pipeline_loader` tests unaffected).

- [ ] **Step 5.6: Commit**

```bash
git add src/config/pipeline_loader.py tests/test_pipeline_loader_orchestrator.py
git commit -m "feat(config): add Orchestrator/Policy/Allocation/Judge config classes"
```

---

## Task 6: Add `pipeline.yaml` blocks

**Files:**
- Modify: `pipeline.yaml`

- [ ] **Step 6.1: Append new blocks to `pipeline.yaml`**

Open `pipeline.yaml`. Append at the bottom (after the existing `retrieval:` section):

```yaml
# ---------------------------------------------------------------------------
# Orchestrator — yeni agentic pipeline (planner → policy → allocator →
# retrieve → assembler → judge → expander → answer → sanitizer → citations)
# ---------------------------------------------------------------------------
orchestrator:
  enabled: false       # false: legacy PlanningAgent.run() / true: OrchestratorAgent.run()

policy:
  mode: session_intersection   # session-seçimi ∩ planner-önerisi
  # auth_rules: ileride eklenecek; tüm koleksiyonlar şu an public

allocation:
  defaults:    { primary: 2, reserve: 2, fetch_k: 10 }
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
  on_low_confidence: expand
```

- [ ] **Step 6.2: Verify YAML loads w/o error**

Run: `python -c "from src.config.pipeline_loader import load_pipeline_config; c = load_pipeline_config(); print(c.orchestrator.enabled, c.policy.mode, c.allocation.max_total_primary, c.judge.heuristic.min_chunks)"`
Expected: `False session_intersection 12 4`

- [ ] **Step 6.3: Commit**

```bash
git add pipeline.yaml
git commit -m "feat(config): add orchestrator/policy/allocation/judge YAML blocks"
```

---

## Task 7: Build `PolicyEnforcer`

**Files:**
- Create: `src/agent/policy.py`
- Test: `tests/test_policy.py`

- [ ] **Step 7.1: Write failing tests**

Create `tests/test_policy.py`:

```python
"""Unit tests for PolicyEnforcer."""
from __future__ import annotations

import pytest

from src.agent.policy import PolicyEnforcer
from src.agent.schemas import (
    CollectionSearchPlan,
    OrchestratorState,
    SearchPlan,
    SearchQueryDraft,
)
from src.config.pipeline_loader import PolicyConfig


def _state_with_suggestions(*collections: str) -> OrchestratorState:
    plan = SearchPlan(
        intent="factual",
        resources=[
            CollectionSearchPlan(
                collection=c,
                query_drafts=[SearchQueryDraft(text="q")],
            )
            for c in collections
        ],
        reasoning="r",
    )
    return OrchestratorState(request_id="r1", user_query="q", planner_output=plan)


def test_intersection_keeps_only_session_collections():
    enf = PolicyEnforcer(PolicyConfig({}))
    state = _state_with_suggestions("a", "b", "c")
    enf.run(state, session_collections=["a", "c"])
    assert state.policy_result.allowed_collections == ["a", "c"]
    assert state.policy_result.denied_collections == ["b"]
    assert state.policy_result.reason_by_collection == {"b": "not_in_session_selection"}


def test_empty_session_denies_all():
    enf = PolicyEnforcer(PolicyConfig({}))
    state = _state_with_suggestions("a", "b")
    enf.run(state, session_collections=[])
    assert state.policy_result.allowed_collections == []
    assert state.policy_result.denied_collections == ["a", "b"]
    assert "policy_no_allowed_collections" in state.errors


def test_empty_planner_suggestions_empty_allowed():
    enf = PolicyEnforcer(PolicyConfig({}))
    state = _state_with_suggestions()
    enf.run(state, session_collections=["a"])
    assert state.policy_result.allowed_collections == []
    assert "policy_no_allowed_collections" in state.errors
```

- [ ] **Step 7.2: Run failing tests**

Run: `pytest tests/test_policy.py -v`
Expected: ImportError on `src.agent.policy`.

- [ ] **Step 7.3: Create `src/agent/policy.py`**

Create the file:

```python
"""PolicyEnforcer — session-selection intersection over planner-suggested collections."""
from __future__ import annotations

from src.agent.schemas import OrchestratorState, PolicyResult
from src.config.pipeline_loader import PolicyConfig


class PolicyEnforcer:
    """Enforces collection-access policy.

    Current mode: session_intersection. Planner-suggested collections are
    intersected with the user's session-selected collections; denied entries
    carry a reason string for trace and UI.
    """

    def __init__(self, config: PolicyConfig) -> None:
        self._config = config

    def run(
        self,
        state: OrchestratorState,
        session_collections: list[str],
    ) -> OrchestratorState:
        suggested: list[str] = []
        if state.planner_output is not None:
            suggested = [r.collection for r in state.planner_output.resources]

        session_set = set(session_collections)
        allowed = [c for c in suggested if c in session_set]
        denied = [c for c in suggested if c not in session_set]

        state.policy_result = PolicyResult(
            allowed_collections=allowed,
            denied_collections=denied,
            reason_by_collection={c: "not_in_session_selection" for c in denied},
        )
        if not allowed:
            state.errors.append("policy_no_allowed_collections")
        return state
```

- [ ] **Step 7.4: Run tests, verify pass**

Run: `pytest tests/test_policy.py -v`
Expected: 3 passed.

- [ ] **Step 7.5: Commit**

```bash
git add src/agent/policy.py tests/test_policy.py
git commit -m "feat(agent): add PolicyEnforcer for session-intersection collection gating"
```

---

## Task 8: Build `AllocationPlanner`

**Files:**
- Create: `src/agent/allocator.py`
- Test: `tests/test_allocator.py`

- [ ] **Step 8.1: Write failing tests**

Create `tests/test_allocator.py`:

```python
"""Unit tests for AllocationPlanner."""
from __future__ import annotations

import pytest

from src.agent.allocator import AllocationPlanner
from src.agent.schemas import (
    CollectionSearchPlan,
    OrchestratorState,
    PolicyResult,
    SearchPlan,
    SearchQueryDraft,
)
from src.config.pipeline_loader import AllocationConfig


@pytest.fixture
def alloc_config() -> AllocationConfig:
    return AllocationConfig({
        "defaults": {"primary": 2, "reserve": 2, "fetch_k": 10},
        "by_query_type": {
            "comparison": {"primary": 3, "reserve": 2, "fetch_k": 12},
        },
        "max_per_document": 1,
        "max_total_primary": 12,
    })


def _state(query_type: str, allowed: list[str], filters_for: dict[str, dict] | None = None) -> OrchestratorState:
    filters_for = filters_for or {}
    resources = [
        CollectionSearchPlan(
            collection=c,
            query_drafts=[SearchQueryDraft(text="q", filters=filters_for.get(c))],
        )
        for c in allowed
    ]
    plan = SearchPlan(
        intent="factual", query_type=query_type, resources=resources, reasoning="r"
    )
    return OrchestratorState(
        request_id="r1",
        user_query="q",
        planner_output=plan,
        policy_result=PolicyResult(allowed_collections=allowed),
    )


def test_allocation_uses_query_type_budget(alloc_config):
    ap = AllocationPlanner(alloc_config)
    state = _state("comparison", ["c1", "c2"])
    ap.run(state)
    assert len(state.collection_plans) == 2
    p1 = state.collection_plans[0]
    assert p1.retrieval_budget == 3
    assert p1.reserve_budget == 2
    assert p1.fetch_k == 12
    assert p1.priority == 1
    assert p1.route_reason == "planner_suggested_and_session_allowed"
    assert state.collection_plans[1].priority == 2


def test_allocation_falls_back_to_defaults(alloc_config):
    ap = AllocationPlanner(alloc_config)
    state = _state("fact", ["c1"])
    ap.run(state)
    assert state.collection_plans[0].retrieval_budget == 2
    assert state.collection_plans[0].fetch_k == 10


def test_allocation_empty_allowed_records_error(alloc_config):
    ap = AllocationPlanner(alloc_config)
    state = _state("fact", [])
    ap.run(state)
    assert state.collection_plans == []
    assert "allocation_no_allowed_collections" in state.errors


def test_allocation_carries_filters_from_planner(alloc_config):
    ap = AllocationPlanner(alloc_config)
    state = _state(
        "fact",
        ["c1"],
        filters_for={"c1": {"year": 2023}},
    )
    ap.run(state)
    # FilterCriteria validates only known fields; we pass model_dump
    assert "year" in state.collection_plans[0].filters
    assert state.collection_plans[0].filters["year"] == 2023
```

- [ ] **Step 8.2: Run failing tests**

Run: `pytest tests/test_allocator.py -v`
Expected: ImportError on `src.agent.allocator`.

- [ ] **Step 8.3: Create `src/agent/allocator.py`**

Create the file:

```python
"""AllocationPlanner — builds per-collection execution plans with YAML-driven budgets."""
from __future__ import annotations

from src.agent.schemas import (
    CollectionExecutionPlan,
    OrchestratorState,
)
from src.config.pipeline_loader import AllocationConfig


class AllocationPlanner:
    """Maps allowed collections to CollectionExecutionPlan entries.

    Budgets are looked up by `state.planner_output.query_type`. When the
    planner provides a draft with filters for a collection, the first draft's
    filters are propagated to the execution plan.
    """

    def __init__(self, config: AllocationConfig) -> None:
        self._config = config

    def run(self, state: OrchestratorState) -> OrchestratorState:
        if not state.policy_result or not state.policy_result.allowed_collections:
            state.errors.append("allocation_no_allowed_collections")
            return state

        if state.planner_output is None:
            state.errors.append("allocation_no_planner_output")
            return state

        budget = self._config.budget_for(state.planner_output.query_type)
        filters_by_collection = self._collect_first_filters(state.planner_output)

        plans = []
        for idx, name in enumerate(state.policy_result.allowed_collections):
            plans.append(
                CollectionExecutionPlan(
                    collection_name=name,
                    priority=idx + 1,
                    retrieval_budget=budget.primary,
                    reserve_budget=budget.reserve,
                    fetch_k=budget.fetch_k,
                    filters=filters_by_collection.get(name, {}),
                    route_reason="planner_suggested_and_session_allowed",
                )
            )
        state.collection_plans = plans
        return state

    @staticmethod
    def _collect_first_filters(plan) -> dict[str, dict]:
        out: dict[str, dict] = {}
        for resource in plan.resources:
            if not resource.query_drafts:
                continue
            first = resource.query_drafts[0]
            if first.filters is None:
                continue
            data = first.filters.model_dump(exclude_none=True)
            if data:
                out[resource.collection] = data
        return out
```

- [ ] **Step 8.4: Run tests, verify pass**

Run: `pytest tests/test_allocator.py -v`
Expected: 4 passed.

- [ ] **Step 8.5: Commit**

```bash
git add src/agent/allocator.py tests/test_allocator.py
git commit -m "feat(agent): add AllocationPlanner with YAML-driven per-query-type budgets"
```

---

## Task 9: Build `BalancedContextAssembler`

**Files:**
- Create: `src/agent/assembler.py`
- Test: `tests/test_assembler.py`

- [ ] **Step 9.1: Write failing tests**

Create `tests/test_assembler.py`:

```python
"""Unit tests for BalancedContextAssembler."""
from __future__ import annotations

import pytest

from src.agent.assembler import BalancedContextAssembler
from src.agent.schemas import (
    Chunk,
    CollectionExecutionPlan,
    OrchestratorState,
    RetrievalOutput,
)
from src.config.pipeline_loader import AllocationConfig


def _chunk(cid: str, did: str, collection: str, score: float = 0.5) -> Chunk:
    return Chunk(
        chunk_id=cid,
        document_id=did,
        collection_name=collection,
        doc_type="gazete",
        source_title="t",
        text=f"body-{cid}",
        score=score,
        rerank_score=score,
    )


def _config(max_per_doc: int = 1, max_total: int = 12) -> AllocationConfig:
    return AllocationConfig({
        "defaults": {"primary": 2, "reserve": 2, "fetch_k": 10},
        "max_per_document": max_per_doc,
        "max_total_primary": max_total,
    })


def _state(plans: list[CollectionExecutionPlan], results: dict[str, RetrievalOutput]) -> OrchestratorState:
    return OrchestratorState(
        request_id="r1",
        user_query="q",
        collection_plans=plans,
        retrieval_results=results,
    )


def test_assembler_respects_collection_budget():
    plan = CollectionExecutionPlan(
        collection_name="c1", retrieval_budget=2, reserve_budget=2, fetch_k=10,
    )
    ro = RetrievalOutput(
        collection_name="c1",
        chunks=[_chunk("a", "d1", "c1"), _chunk("b", "d2", "c1"), _chunk("c", "d3", "c1")],
        fetched_count=3, returned_count=3, latency_ms=1.0,
    )
    state = _state([plan], {"c1": ro})
    BalancedContextAssembler(_config()).run(state)
    assert [c.chunk_id for c in state.assembled_chunks] == ["a", "b"]
    assert all(item.slot_type == "primary" for item in state.balanced_context)


def test_assembler_dedups_documents_across_collections():
    plans = [
        CollectionExecutionPlan(collection_name="c1", priority=1, retrieval_budget=2, reserve_budget=2, fetch_k=10),
        CollectionExecutionPlan(collection_name="c2", priority=2, retrieval_budget=2, reserve_budget=2, fetch_k=10),
    ]
    results = {
        "c1": RetrievalOutput(
            collection_name="c1",
            chunks=[_chunk("a", "shared_doc", "c1"), _chunk("b", "d2", "c1")],
            fetched_count=2, returned_count=2, latency_ms=1.0,
        ),
        "c2": RetrievalOutput(
            collection_name="c2",
            chunks=[_chunk("x", "shared_doc", "c2"), _chunk("y", "d3", "c2")],
            fetched_count=2, returned_count=2, latency_ms=1.0,
        ),
    }
    state = _state(plans, results)
    BalancedContextAssembler(_config(max_per_doc=1)).run(state)
    doc_ids = [c.document_id for c in state.assembled_chunks]
    # shared_doc must appear at most once
    assert doc_ids.count("shared_doc") == 1


def test_assembler_honors_max_total_primary():
    plans = [
        CollectionExecutionPlan(collection_name=f"c{i}", priority=i, retrieval_budget=2, reserve_budget=0, fetch_k=10)
        for i in range(1, 4)
    ]
    results = {
        f"c{i}": RetrievalOutput(
            collection_name=f"c{i}",
            chunks=[_chunk(f"k{i}a", f"d{i}a", f"c{i}"), _chunk(f"k{i}b", f"d{i}b", f"c{i}")],
            fetched_count=2, returned_count=2, latency_ms=1.0,
        )
        for i in range(1, 4)
    }
    state = _state(plans, results)
    BalancedContextAssembler(_config(max_per_doc=1, max_total=3)).run(state)
    assert len(state.assembled_chunks) == 3


def test_assembler_priority_order():
    plans = [
        CollectionExecutionPlan(collection_name="low", priority=2, retrieval_budget=1, reserve_budget=0, fetch_k=10),
        CollectionExecutionPlan(collection_name="high", priority=1, retrieval_budget=1, reserve_budget=0, fetch_k=10),
    ]
    results = {
        "low": RetrievalOutput(collection_name="low", chunks=[_chunk("L", "dL", "low")], fetched_count=1, returned_count=1, latency_ms=0.0),
        "high": RetrievalOutput(collection_name="high", chunks=[_chunk("H", "dH", "high")], fetched_count=1, returned_count=1, latency_ms=0.0),
    }
    state = _state(plans, results)
    BalancedContextAssembler(_config()).run(state)
    assert state.assembled_chunks[0].chunk_id == "H"
    assert state.assembled_chunks[1].chunk_id == "L"
```

- [ ] **Step 9.2: Run failing tests**

Run: `pytest tests/test_assembler.py -v`
Expected: ImportError on `src.agent.assembler`.

- [ ] **Step 9.3: Create `src/agent/assembler.py`**

Create the file:

```python
"""BalancedContextAssembler — cross-collection doc-deduped primary slot fill."""
from __future__ import annotations

from src.agent.schemas import (
    Chunk,
    ContextAssemblyItem,
    OrchestratorState,
)
from src.config.pipeline_loader import AllocationConfig


class BalancedContextAssembler:
    """Iterates collections in priority order, deduplicates by document_id
    across collections, honors max_per_document and max_total_primary.
    """

    def __init__(self, config: AllocationConfig) -> None:
        self._config = config

    def run(self, state: OrchestratorState) -> OrchestratorState:
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
                per_doc_count[chunk.document_id] = per_doc_count.get(chunk.document_id, 0) + 1
                taken += 1
                total += 1

            if total >= self._config.max_total_primary:
                break

        state.assembled_chunks = assembled
        state.balanced_context = items
        return state
```

- [ ] **Step 9.4: Run tests, verify pass**

Run: `pytest tests/test_assembler.py -v`
Expected: 4 passed.

- [ ] **Step 9.5: Commit**

```bash
git add src/agent/assembler.py tests/test_assembler.py
git commit -m "feat(agent): add BalancedContextAssembler with cross-collection doc-dedup"
```

---

## Task 10: Build `EvidenceJudge` (heuristic only)

**Files:**
- Create: `src/agent/judge.py`
- Test: `tests/test_evidence_judge.py`

- [ ] **Step 10.1: Write failing tests for heuristic path**

Create `tests/test_evidence_judge.py`:

```python
"""Unit tests for EvidenceJudge heuristic decision path."""
from __future__ import annotations

import pytest

from src.agent.judge import EvidenceJudge
from src.agent.schemas import Chunk, OrchestratorState
from src.config.pipeline_loader import JudgeConfig


def _chunk(cid: str, collection: str) -> Chunk:
    return Chunk(
        chunk_id=cid,
        document_id=f"d{cid}",
        collection_name=collection,
        doc_type="gazete",
        source_title="t",
        text="body",
        score=0.5,
        rerank_score=0.5,
    )


def _judge(llm_enabled: bool = False) -> EvidenceJudge:
    cfg = JudgeConfig({
        "heuristic": {"min_chunks": 4, "min_collection_coverage": 2},
        "llm": {"enabled": llm_enabled, "borderline_band": [2, 4]},
        "max_expand_iterations": 1,
    })
    return EvidenceJudge(cfg, client_pool=None)


def test_judge_no_chunks_clarify():
    state = OrchestratorState(request_id="r", user_query="q", assembled_chunks=[])
    _judge().run(state)
    assert state.evidence_decision.action == "clarify"
    assert state.evidence_decision.judge_type == "heuristic"
    assert "no_results" in state.evidence_decision.missing_aspects


def test_judge_heuristic_pass_with_enough_chunks_and_coverage():
    chunks = [_chunk(str(i), "c1" if i < 3 else "c2") for i in range(5)]
    state = OrchestratorState(request_id="r", user_query="q", assembled_chunks=chunks)
    _judge().run(state)
    assert state.evidence_decision.action == "answer"
    assert state.evidence_decision.sufficient is True
    assert state.evidence_decision.judge_type == "heuristic"


def test_judge_heuristic_expand_when_below_threshold_and_llm_disabled():
    # 5 chunks, single collection (coverage = 1 < 2) → fall through to expand
    chunks = [_chunk(str(i), "c1") for i in range(5)]
    state = OrchestratorState(request_id="r", user_query="q", assembled_chunks=chunks)
    _judge(llm_enabled=False).run(state)
    assert state.evidence_decision.action == "expand"
    assert state.evidence_decision.judge_type == "heuristic"
```

- [ ] **Step 10.2: Run failing tests**

Run: `pytest tests/test_evidence_judge.py -v`
Expected: ImportError on `src.agent.judge`.

- [ ] **Step 10.3: Create `src/agent/judge.py` with heuristic only**

Create the file (LLM path is a stub returning heuristic fallback; filled in Task 11):

```python
"""EvidenceJudge — hybrid heuristic + LLM evidence-sufficiency decision."""
from __future__ import annotations

from typing import Optional

from src.agent.schemas import EvidenceDecision, OrchestratorState
from src.config.pipeline_loader import JudgeConfig


class EvidenceJudge:
    """Decides whether assembled chunks suffice to answer.

    Heuristic stage: chunk count + cross-collection coverage. Borderline
    cases fall through to an LLM judge (added in a later task); until then,
    they map to heuristic 'expand'.
    """

    def __init__(self, config: JudgeConfig, client_pool: Optional[object]) -> None:
        self._config = config
        self._pool = client_pool

    def run(self, state: OrchestratorState) -> OrchestratorState:
        chunks = state.assembled_chunks
        h = self._config.heuristic

        if len(chunks) == 0:
            state.evidence_decision = EvidenceDecision(
                sufficient=False,
                confidence=0.0,
                action="clarify",
                missing_aspects=["no_results"],
                judge_type="heuristic",
            )
            return state

        coverage = len({c.collection_name for c in chunks})
        if len(chunks) >= h.min_chunks and coverage >= h.min_collection_coverage:
            state.evidence_decision = EvidenceDecision(
                sufficient=True,
                confidence=0.85,
                action="answer",
                judge_type="heuristic",
            )
            return state

        llm = self._config.llm
        in_band = llm.borderline_band[0] <= len(chunks) <= llm.borderline_band[1]
        if llm.enabled and in_band and self._pool is not None:
            state.evidence_decision = self._llm_judge(state)
            return state

        state.evidence_decision = EvidenceDecision(
            sufficient=False,
            confidence=0.4,
            action="expand",
            missing_aspects=["insufficient_chunks"],
            judge_type="heuristic",
        )
        return state

    def _llm_judge(self, state: OrchestratorState) -> EvidenceDecision:
        # Filled in Task 11. Heuristic fallback for now.
        return EvidenceDecision(
            sufficient=False,
            confidence=0.4,
            action="expand",
            missing_aspects=["insufficient_chunks"],
            judge_type="heuristic",
        )
```

- [ ] **Step 10.4: Run tests, verify pass**

Run: `pytest tests/test_evidence_judge.py -v`
Expected: 3 passed.

- [ ] **Step 10.5: Commit**

```bash
git add src/agent/judge.py tests/test_evidence_judge.py
git commit -m "feat(agent): add EvidenceJudge heuristic decision path"
```

---

## Task 11: Add LLM judge path

**Files:**
- Modify: `src/agent/judge.py`
- Test: `tests/test_evidence_judge.py`

- [ ] **Step 11.1: Append failing tests for LLM path**

Append to `tests/test_evidence_judge.py`:

```python
class _FakeLLMClient:
    """Returns a fixed chat response payload to drive judge decisions."""

    def __init__(self, response_text: str) -> None:
        self.response_text = response_text
        self.calls: list[dict] = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        return {"message": {"content": self.response_text}}


class _FakeLLMPool:
    """Minimal LLMClientPool stand-in for tests."""

    def __init__(self, client: _FakeLLMClient) -> None:
        self._client = client

    def client_for(self, block: str):
        return self._client


def _judge_with_llm(response_text: str) -> tuple[EvidenceJudge, _FakeLLMClient]:
    cfg = JudgeConfig({
        "heuristic": {"min_chunks": 4, "min_collection_coverage": 2},
        "llm": {
            "enabled": True,
            "borderline_band": [2, 4],
            "block": "fast-01",
            "model_key": "judge",
        },
    })
    client = _FakeLLMClient(response_text)
    pool = _FakeLLMPool(client)
    return EvidenceJudge(cfg, client_pool=pool), client


def test_judge_llm_path_returns_answer_action():
    judge, client = _judge_with_llm(
        '{"sufficient": true, "confidence": 0.7, "action": "answer", "missing_aspects": []}'
    )
    # 3 chunks → within borderline_band [2,4]; coverage 1 → not heuristic-pass
    chunks = [_chunk(str(i), "c1") for i in range(3)]
    state = OrchestratorState(request_id="r", user_query="q", assembled_chunks=chunks)
    judge.run(state)
    assert state.evidence_decision.action == "answer"
    assert state.evidence_decision.judge_type == "llm"
    assert len(client.calls) == 1


def test_judge_llm_invalid_json_falls_back_to_heuristic_expand():
    judge, _ = _judge_with_llm("not json")
    chunks = [_chunk(str(i), "c1") for i in range(3)]
    state = OrchestratorState(request_id="r", user_query="q", assembled_chunks=chunks)
    judge.run(state)
    assert state.evidence_decision.action == "expand"
    assert state.evidence_decision.judge_type == "heuristic"
```

- [ ] **Step 11.2: Run failing tests**

Run: `pytest tests/test_evidence_judge.py::test_judge_llm_path_returns_answer_action -v`
Expected: FAIL — `_llm_judge` still returns heuristic stub, `judge_type` ≠ `"llm"`.

- [ ] **Step 11.3: Implement `_llm_judge` in `src/agent/judge.py`**

Replace the `_llm_judge` stub with a real implementation. Add a `_call_llm` helper and a `_parse_decision` helper. Update imports to add `json`:

```python
"""EvidenceJudge — hybrid heuristic + LLM evidence-sufficiency decision."""
from __future__ import annotations

import json
import re
from typing import Optional

from src.agent.schemas import EvidenceDecision, OrchestratorState
from src.config.pipeline_loader import JudgeConfig


_JUDGE_PROMPT = """Sen bir kanıt yeterlilik değerlendirme uzmanısın.
Aşağıdaki soruya verilen bağlam parçaları yeterli mi?

Soru: {query}
Niyet: {intent} / Sorgu tipi: {query_type}

Bağlam parçaları:
{context}

Yanıt JSON formatında ve sadece bu alanlarla:
{{"sufficient": true|false, "confidence": 0.0-1.0,
  "action": "answer"|"expand"|"clarify"|"refuse",
  "missing_aspects": ["..."]}}
"""


class EvidenceJudge:
    def __init__(self, config: JudgeConfig, client_pool: Optional[object]) -> None:
        self._config = config
        self._pool = client_pool

    def run(self, state: OrchestratorState) -> OrchestratorState:
        chunks = state.assembled_chunks
        h = self._config.heuristic

        if len(chunks) == 0:
            state.evidence_decision = EvidenceDecision(
                sufficient=False,
                confidence=0.0,
                action="clarify",
                missing_aspects=["no_results"],
                judge_type="heuristic",
            )
            return state

        coverage = len({c.collection_name for c in chunks})
        if len(chunks) >= h.min_chunks and coverage >= h.min_collection_coverage:
            state.evidence_decision = EvidenceDecision(
                sufficient=True,
                confidence=0.85,
                action="answer",
                judge_type="heuristic",
            )
            return state

        llm = self._config.llm
        in_band = llm.borderline_band[0] <= len(chunks) <= llm.borderline_band[1]
        if llm.enabled and in_band and self._pool is not None:
            decision = self._llm_judge(state)
            state.evidence_decision = decision
            return state

        state.evidence_decision = EvidenceDecision(
            sufficient=False,
            confidence=0.4,
            action="expand",
            missing_aspects=["insufficient_chunks"],
            judge_type="heuristic",
        )
        return state

    def _llm_judge(self, state: OrchestratorState) -> EvidenceDecision:
        llm = self._config.llm
        try:
            client = self._pool.client_for(llm.block)
        except Exception:
            return self._heuristic_expand_fallback()

        intent = state.planner_output.intent if state.planner_output else "unknown"
        query_type = state.planner_output.query_type if state.planner_output else "fact"
        context = "\n".join(
            f"[{i+1}] ({c.collection_name}/{c.document_id}) {c.text[:240]}"
            for i, c in enumerate(state.assembled_chunks)
        )
        prompt = _JUDGE_PROMPT.format(
            query=state.user_query,
            intent=intent,
            query_type=query_type,
            context=context,
        )

        try:
            response = client.chat(
                messages=[{"role": "user", "content": prompt}],
                model=llm.model_key,
                timeout=llm.timeout_seconds,
            )
            raw = response["message"]["content"]
        except Exception:
            return self._heuristic_expand_fallback()

        decision = self._parse_decision(raw)
        if decision is None:
            return self._heuristic_expand_fallback()
        return decision

    @staticmethod
    def _parse_decision(raw: str) -> Optional[EvidenceDecision]:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            return None
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
        try:
            return EvidenceDecision(
                sufficient=bool(data.get("sufficient", False)),
                confidence=float(data.get("confidence", 0.0)),
                action=data.get("action", "expand"),
                missing_aspects=list(data.get("missing_aspects", []) or []),
                judge_type="llm",
            )
        except Exception:
            return None

    @staticmethod
    def _heuristic_expand_fallback() -> EvidenceDecision:
        return EvidenceDecision(
            sufficient=False,
            confidence=0.4,
            action="expand",
            missing_aspects=["insufficient_chunks"],
            judge_type="heuristic",
        )
```

- [ ] **Step 11.4: Run all judge tests, verify pass**

Run: `pytest tests/test_evidence_judge.py -v`
Expected: 5 passed.

- [ ] **Step 11.5: Commit**

```bash
git add src/agent/judge.py tests/test_evidence_judge.py
git commit -m "feat(agent): add LLM borderline path to EvidenceJudge with safe fallback"
```

---

## Task 12: Build `ExpansionPlanner`

**Files:**
- Create: `src/agent/expander.py`
- Test: `tests/test_expander.py`

- [ ] **Step 12.1: Write failing tests**

Create `tests/test_expander.py`:

```python
"""Unit tests for ExpansionPlanner."""
from __future__ import annotations

from src.agent.expander import ExpansionPlanner
from src.agent.schemas import (
    Chunk,
    CollectionExecutionPlan,
    EvidenceDecision,
    OrchestratorState,
    RetrievalOutput,
)


def _chunk(cid: str, did: str, collection: str) -> Chunk:
    return Chunk(
        chunk_id=cid, document_id=did, collection_name=collection,
        doc_type="gazete", source_title="t", text="body",
        score=0.5, rerank_score=0.5,
    )


def _state_with_reserves(action: str = "expand") -> OrchestratorState:
    plan = CollectionExecutionPlan(
        collection_name="c1", retrieval_budget=2, reserve_budget=2, fetch_k=10,
    )
    ro = RetrievalOutput(
        collection_name="c1",
        chunks=[_chunk("p1", "d1", "c1"), _chunk("p2", "d2", "c1")],
        reserve_chunks=[_chunk("r1", "d3", "c1"), _chunk("r2", "d2", "c1")],  # r2 dups d2
        fetched_count=4, returned_count=2, latency_ms=1.0,
    )
    return OrchestratorState(
        request_id="r", user_query="q",
        collection_plans=[plan],
        retrieval_results={"c1": ro},
        assembled_chunks=[_chunk("p1", "d1", "c1"), _chunk("p2", "d2", "c1")],
        evidence_decision=EvidenceDecision(
            sufficient=False, confidence=0.4, action=action, judge_type="heuristic"
        ),
    )


def test_expander_pulls_only_non_duplicate_reserves():
    state = _state_with_reserves()
    ExpansionPlanner().run(state)
    cids = [c.chunk_id for c in state.assembled_chunks]
    assert "r1" in cids
    # r2's document_id d2 already assembled → must be skipped
    assert "r2" not in cids
    assert state.expanded is True


def test_expander_honors_reserve_budget():
    plan = CollectionExecutionPlan(
        collection_name="c1", retrieval_budget=1, reserve_budget=1, fetch_k=10,
    )
    ro = RetrievalOutput(
        collection_name="c1",
        chunks=[_chunk("p1", "d1", "c1")],
        reserve_chunks=[_chunk("r1", "d2", "c1"), _chunk("r2", "d3", "c1")],
        fetched_count=3, returned_count=1, latency_ms=1.0,
    )
    state = OrchestratorState(
        request_id="r", user_query="q",
        collection_plans=[plan],
        retrieval_results={"c1": ro},
        assembled_chunks=[_chunk("p1", "d1", "c1")],
        evidence_decision=EvidenceDecision(
            sufficient=False, confidence=0.4, action="expand", judge_type="heuristic"
        ),
    )
    ExpansionPlanner().run(state)
    cids = [c.chunk_id for c in state.assembled_chunks]
    assert cids == ["p1", "r1"]


def test_expander_noop_if_action_not_expand():
    state = _state_with_reserves(action="answer")
    before = list(state.assembled_chunks)
    ExpansionPlanner().run(state)
    assert state.assembled_chunks == before
    assert state.expanded is False
```

- [ ] **Step 12.2: Run failing tests**

Run: `pytest tests/test_expander.py -v`
Expected: ImportError on `src.agent.expander`.

- [ ] **Step 12.3: Create `src/agent/expander.py`**

Create the file:

```python
"""ExpansionPlanner — consumes per-collection reserves to extend assembled context."""
from __future__ import annotations

from src.agent.schemas import ContextAssemblyItem, OrchestratorState


class ExpansionPlanner:
    """Pulls held-back reserve_chunks into assembled_chunks when the judge
    asks for expansion. No new vector calls; reserves are the next-best
    candidates from the original fetch.
    """

    def run(self, state: OrchestratorState) -> OrchestratorState:
        decision = state.evidence_decision
        if decision is None or decision.action != "expand":
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

- [ ] **Step 12.4: Run tests, verify pass**

Run: `pytest tests/test_expander.py -v`
Expected: 3 passed.

- [ ] **Step 12.5: Commit**

```bash
git add src/agent/expander.py tests/test_expander.py
git commit -m "feat(agent): add ExpansionPlanner consuming per-collection reserves"
```

---

## Task 13: Build `CitationBuilder`

**Files:**
- Create: `src/agent/citations.py`
- Test: `tests/test_citations.py`

- [ ] **Step 13.1: Write failing tests**

Create `tests/test_citations.py`:

```python
"""Unit tests for CitationBuilder."""
from __future__ import annotations

from src.agent.citations import CitationBuilder
from src.agent.schemas import Chunk


def _chunk(i: int) -> Chunk:
    return Chunk(
        chunk_id=f"c{i}",
        document_id=f"d{i}",
        collection_name="col",
        doc_type="gazete",
        source_title=f"title-{i}",
        text=f"body-{i}",
        score=0.5,
        metadata={"year": 2020 + i},
    )


def test_citation_builder_produces_indexed_dicts():
    chunks = [_chunk(1), _chunk(2)]
    cites = CitationBuilder.build(chunks)
    assert len(cites) == 2
    assert cites[0]["index"] == 1
    assert cites[1]["index"] == 2
    assert cites[0]["chunk_id"] == "c1"
    assert cites[0]["collection_name"] == "col"
    assert cites[0]["doc_type"] == "gazete"
    assert cites[0]["source_title"] == "title-1"
    assert cites[0]["metadata"] == {"year": 2021}


def test_citation_builder_empty_returns_empty_list():
    assert CitationBuilder.build([]) == []
```

- [ ] **Step 13.2: Run failing tests**

Run: `pytest tests/test_citations.py -v`
Expected: ImportError on `src.agent.citations`.

- [ ] **Step 13.3: Create `src/agent/citations.py`**

Create the file:

```python
"""CitationBuilder — produces a stable citation list from assembled chunks."""
from __future__ import annotations

from src.agent.schemas import Chunk


class CitationBuilder:
    """Maps assembled chunks to citation dicts in stable order."""

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
                "metadata": dict(c.metadata),
            }
            for i, c in enumerate(chunks)
        ]
```

- [ ] **Step 13.4: Run tests, verify pass**

Run: `pytest tests/test_citations.py -v`
Expected: 2 passed.

- [ ] **Step 13.5: Commit**

```bash
git add src/agent/citations.py tests/test_citations.py
git commit -m "feat(agent): add CitationBuilder producing indexed citation dicts"
```

---

## Task 14: Extract `Planner.plan()` from `PlanningAgent`

**Files:**
- Modify: `src/agent/planner.py`
- Test: `tests/test_agent_planner.py` (existing — keep passing)

This task adds a thin `Planner` class that exposes only plan generation, reusing `PlanningAgent`'s prompt logic. `PlanningAgent` stays for the legacy path; `OrchestratorAgent` (Task 15) will use `Planner` instead.

- [ ] **Step 14.1: Write failing test for new `Planner.plan()`**

Append to `tests/test_agent_planner.py`:

```python
def test_new_planner_class_returns_search_plan(monkeypatch):
    """The Planner class wraps PlanningAgent._generate_plan into a public method."""
    from src.agent.planner import Planner
    from src.agent.tracer import PipelineTracer

    cfg = load_pipeline_config()
    pool = LLMClientPool.from_config(cfg)
    planner = Planner(cfg, pool)

    # Force fallback path: empty LLM response → fallback_plan
    def _no_plan(*a, **kw):
        return None
    monkeypatch.setattr(planner._inner, "_generate_plan", _no_plan)

    plan = planner.plan("test query")
    assert plan is not None
    assert plan.intent == "unknown"  # fallback marker
    assert plan.resources
```

- [ ] **Step 14.2: Run failing test**

Run: `pytest tests/test_agent_planner.py::test_new_planner_class_returns_search_plan -v`
Expected: ImportError on `Planner`.

- [ ] **Step 14.3: Add `Planner` class to `src/agent/planner.py`**

At the bottom of `src/agent/planner.py`, append:

```python
class Planner:
    """Thin facade exposing only plan generation for the OrchestratorAgent.

    Reuses PlanningAgent's prompt and fallback logic; does not run retrieval,
    answering, sanitizer, or any retry loops.
    """

    def __init__(self, config: PipelineConfig, client_pool: LLMClientPool) -> None:
        self._inner = PlanningAgent(config, client_pool)

    def plan(self, query: str, tracer: "PipelineTracer | None" = None) -> SearchPlan:
        from src.agent.tracer import PipelineTracer

        tracer = tracer or PipelineTracer()
        plan = self._inner._generate_plan(query, tracer)
        if plan is None:
            plan = self._inner._fallback_plan(query)
        return plan
```

(The `_generate_plan` / `_fallback_plan` are existing private methods on `PlanningAgent`; calling them from the same module is acceptable.)

- [ ] **Step 14.4: Run test, verify pass**

Run: `pytest tests/test_agent_planner.py::test_new_planner_class_returns_search_plan -v`
Expected: PASS.

- [ ] **Step 14.5: Run full planner test file, ensure no regressions**

Run: `pytest tests/test_agent_planner.py -v`
Expected: all green (existing tests untouched).

- [ ] **Step 14.6: Commit**

```bash
git add src/agent/planner.py tests/test_agent_planner.py
git commit -m "feat(agent): add Planner facade for plan-only invocation"
```

---

## Task 15: Build `OrchestratorAgent` (heuristic-judge path, no streaming)

**Files:**
- Create: `src/agent/orchestrator.py`
- Test: `tests/test_orchestrator.py`

This task lands the orchestrator with synchronous, non-streaming answer generation. Streaming is wired in a later task.

- [ ] **Step 15.1: Write failing tests using mocked tools**

Create `tests/test_orchestrator.py`:

```python
"""End-to-end orchestrator tests with mocked SearchTool and answering."""
from __future__ import annotations

import pytest

from src.agent.orchestrator import OrchestratorAgent
from src.agent.schemas import (
    CollectionSearchPlan,
    SearchPlan,
    SearchQueryDraft,
)
from src.common.llm_client_pool import LLMClientPool
from src.config.pipeline_loader import load_pipeline_config


def _make_plan(*collections: str) -> SearchPlan:
    return SearchPlan(
        intent="factual",
        query_type="fact",
        resources=[
            CollectionSearchPlan(
                collection=c,
                query_drafts=[SearchQueryDraft(text="q", top_k=5)],
            )
            for c in collections
        ],
        reasoning="r",
    )


def _make_search_result(chunk_ids: list[str], doc_ids: list[str], collection: str) -> dict:
    return {
        "documents": [f"body-{i}" for i in chunk_ids],
        "metadatas": [
            {
                "chunk_id": cid,
                "document_id": did,
                "doc_type": "gazete",
                "source_title": f"t-{cid}",
                "collection": collection,
            }
            for cid, did in zip(chunk_ids, doc_ids)
        ],
        "distances": [0.1 for _ in chunk_ids],
    }


def _agent(monkeypatch, plan_collections=("gazete_arsivi",), result_chunks_by_collection=None):
    """Build an OrchestratorAgent with mocked planner/search/answer/sanitizer."""
    cfg = load_pipeline_config()
    pool = LLMClientPool.from_config(cfg)
    agent = OrchestratorAgent(cfg, pool)

    monkeypatch.setattr(agent._planner, "plan", lambda q, tracer=None: _make_plan(*plan_collections))

    def _search(collection_key, query_text, filters=None, top_k=5):
        chunks = (result_chunks_by_collection or {}).get(collection_key, [])
        return _make_search_result(
            chunk_ids=[c["chunk_id"] for c in chunks],
            doc_ids=[c["document_id"] for c in chunks],
            collection=collection_key,
        )
    monkeypatch.setattr(agent._search_tool, "search", _search)

    monkeypatch.setattr(
        agent._answer_tool, "answer",
        lambda query, context: ("thinking", "Cevap metni."),
    )
    monkeypatch.setattr(
        agent._sanitizer, "validate",
        lambda *a, **kw: None,  # treat as pass-through
    )
    return agent


def test_orchestrator_no_allowed_collections_returns_refuse(monkeypatch):
    agent = _agent(monkeypatch, plan_collections=("disallowed_collection",))
    out = agent.run("q", session_collections=["gazete_arsivi"])
    assert out.evidence_decision is None
    assert "Yetkili kaynaklarla" in out.answer or "Seçili koleksiyon" in out.answer
    assert out.policy_result.allowed_collections == []


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


def test_orchestrator_zero_chunks_returns_clarify(monkeypatch):
    agent = _agent(
        monkeypatch,
        plan_collections=("col_a",),
        result_chunks_by_collection={"col_a": []},
    )
    out = agent.run("q", session_collections=["col_a"])
    assert out.evidence_decision.action == "clarify"


def test_orchestrator_expand_path_uses_reserves(monkeypatch):
    # Single collection (coverage=1 < 2) with fetch_k chunks: primary fills, reserve held back
    chunks = [{"chunk_id": f"k{i}", "document_id": f"d{i}"} for i in range(6)]
    agent = _agent(
        monkeypatch,
        plan_collections=("col_a",),
        result_chunks_by_collection={"col_a": chunks},
    )
    out = agent.run("q", session_collections=["col_a"])
    # After expand, more chunks assembled, but action settles to either answer (heuristic-pass after expand) or expand again capped
    assert out.expanded is True
```

- [ ] **Step 15.2: Run failing tests**

Run: `pytest tests/test_orchestrator.py -v`
Expected: ImportError on `src.agent.orchestrator`.

- [ ] **Step 15.3: Create `src/agent/orchestrator.py`**

Create the file:

```python
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
from src.agent.schemas import (
    AgentOutput,
    Chunk,
    EvidenceDecision,
    OrchestratorState,
    RetrievalOutput,
)
from src.agent.tools import AnswerTool, ContextBuilderTool, SearchTool
from src.agent.tracer import PipelineTracer
from src.agent.sanitizer import SanitizerAgent
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

    def __init__(self, config: PipelineConfig, client_pool: LLMClientPool) -> None:
        self._config = config
        self._pool = client_pool
        self._planner = Planner(config, client_pool)
        self._policy = PolicyEnforcer(config.policy)
        self._allocator = AllocationPlanner(config.allocation)
        self._search_tool = SearchTool(config, client_pool)
        self._context_tool = ContextBuilderTool(config)
        self._assembler = BalancedContextAssembler(config.allocation)
        self._judge = EvidenceJudge(config.judge, client_pool)
        self._expander = ExpansionPlanner()
        self._answer_tool = AnswerTool(client_pool, config)
        self._sanitizer = SanitizerAgent(client_pool, config)

    def run(self, query: str, session_collections: list[str]) -> AgentOutput:
        state = OrchestratorState(
            request_id=str(uuid.uuid4()),
            user_query=query,
        )
        tracer = PipelineTracer()

        # 1. Plan
        state.planner_output = self._planner.plan(query, tracer)

        # 2. Policy
        self._policy.run(state, session_collections)
        if not state.policy_result.allowed_collections:
            return self._build_refuse_output(state, "no_allowed_collections", tracer)

        # 3. Allocate
        self._allocator.run(state)
        if not state.collection_plans:
            return self._build_refuse_output(state, "no_allowed_collections", tracer)

        # 4. Retrieve in parallel
        self._retrieve_all(state)

        # 5. Assemble
        self._assembler.run(state)

        # 6. Judge
        self._judge.run(state)

        # 7. Expand once if needed; re-judge
        max_iters = self._config.judge.max_expand_iterations
        if state.evidence_decision.action == "expand" and max_iters > 0:
            self._expander.run(state)
            self._judge.run(state)

        action = state.evidence_decision.action
        if action == "clarify":
            return self._build_refuse_output(state, "clarify", tracer)
        if action == "refuse":
            return self._build_refuse_output(state, "judge_refuse", tracer)

        # 8. Build context and answer
        context = self._build_context(state)
        thinking, answer = self._answer_tool.answer(query=query, context=context)
        state.final_answer = answer

        # 9. Sanitizer (post-answer text validator; existing behavior)
        validation = self._sanitizer.validate(
            query=query, answer=answer, context=context,
        )
        if validation and not validation.passes and validation.corrected_answer:
            state.final_answer = validation.corrected_answer

        # 10. Citations
        state.citations = CitationBuilder.build(state.assembled_chunks)

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

    # --- helpers ------------------------------------------------------------

    def _retrieve_all(self, state: OrchestratorState) -> None:
        def _one(plan):
            t0 = time.perf_counter()
            result = self._search_tool.search(
                collection_key=plan.collection_name,
                query_text=state.user_query,
                filters=plan.filters or None,
                top_k=plan.fetch_k,
            )
            chunks = self._dict_to_chunks(result, plan.collection_name)
            primary = chunks[: plan.retrieval_budget]
            reserve_start = plan.retrieval_budget
            reserve_end = plan.retrieval_budget + plan.reserve_budget
            reserve = chunks[reserve_start:reserve_end]
            return plan.collection_name, RetrievalOutput(
                collection_name=plan.collection_name,
                chunks=primary,
                reserve_chunks=reserve,
                fetched_count=len(chunks),
                returned_count=len(primary),
                latency_ms=(time.perf_counter() - t0) * 1000,
                filter_applied=plan.filters or {},
            )

        with ThreadPoolExecutor(max_workers=max(1, len(state.collection_plans))) as ex:
            for name, ro in ex.map(_one, state.collection_plans):
                state.retrieval_results[name] = ro

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

    def _build_context(self, state: OrchestratorState) -> str:
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
        # Ensure an evidence_decision exists for downstream consumers when refusal
        # came from policy stage.
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
        )
```

- [ ] **Step 15.4: Run tests, verify pass**

Run: `pytest tests/test_orchestrator.py -v`
Expected: 4 passed.

- [ ] **Step 15.5: Run full suite**

Run: `python -m pytest tests/ -x -q`
Expected: all green.

- [ ] **Step 15.6: Commit**

```bash
git add src/agent/orchestrator.py tests/test_orchestrator.py
git commit -m "feat(agent): add OrchestratorAgent state-machine pipeline (non-streaming)"
```

---

## Task 16: Wire `RAGService` to dispatch on flag

**Files:**
- Modify: `src/generator/service.py`
- Test: `tests/test_service_dispatch.py`

- [ ] **Step 16.1: Read `src/generator/service.py` to locate dispatch point**

Run: `grep -n "PlanningAgent\|ask\|def " src/generator/service.py | head -30`
Confirm the entrypoint method (e.g. `ask`, `ask_stream`, or `run_agent`). Adapt the wiring step to that exact method.

- [ ] **Step 16.2: Write failing test**

Create `tests/test_service_dispatch.py`:

```python
"""Tests for RAGService dispatch between legacy PlanningAgent and OrchestratorAgent."""
from __future__ import annotations

import pytest

from src.config.pipeline_loader import load_pipeline_config
from src.generator.service import RAGService


def test_service_uses_orchestrator_when_flag_enabled(monkeypatch):
    cfg = load_pipeline_config()
    cfg.orchestrator.enabled = True   # force flag on
    svc = RAGService(config=cfg)

    captured = {}

    def fake_run(query, session_collections):
        captured["called"] = "orchestrator"
        captured["query"] = query
        captured["session"] = session_collections
        from src.agent.schemas import AgentOutput
        return AgentOutput(answer="ok")

    monkeypatch.setattr(svc._orchestrator, "run", fake_run)
    out = svc.run_agent("hello", session_collections=["c1"])
    assert captured["called"] == "orchestrator"
    assert out.answer == "ok"


def test_service_uses_legacy_planner_when_flag_disabled(monkeypatch):
    cfg = load_pipeline_config()
    cfg.orchestrator.enabled = False
    svc = RAGService(config=cfg)

    captured = {}

    def fake_run(query, *, trace=None):
        captured["called"] = "legacy"
        from src.agent.schemas import AgentOutput
        return AgentOutput(answer="legacy")

    monkeypatch.setattr(svc._planning_agent, "run", fake_run)
    out = svc.run_agent("hello", session_collections=["c1"])
    assert captured["called"] == "legacy"
    assert out.answer == "legacy"
```

If `RAGService` does not yet expose `run_agent` with this signature, this test pins the contract. Implement to match.

- [ ] **Step 16.3: Run failing tests**

Run: `pytest tests/test_service_dispatch.py -v`
Expected: failures — either `run_agent` missing or `_orchestrator` not on service.

- [ ] **Step 16.4: Modify `src/generator/service.py`**

Add the orchestrator to `RAGService.__init__` and a `run_agent` dispatch method. Locate `RAGService.__init__` and add after the existing `PlanningAgent` construction:

```python
from src.agent.orchestrator import OrchestratorAgent
# ... in __init__ after self._planning_agent = PlanningAgent(...)
self._orchestrator = OrchestratorAgent(config, client_pool)
```

Then add a `run_agent` method on `RAGService`:

```python
def run_agent(self, query: str, session_collections: list[str]):
    """Dispatch to OrchestratorAgent or legacy PlanningAgent based on flag."""
    if self._config.orchestrator.enabled:
        return self._orchestrator.run(query, session_collections)
    return self._planning_agent.run(query)
```

If `RAGService` currently exposes a different entrypoint name (e.g. `ask_stream`), keep that name and add `run_agent` as a sibling. Do not modify the existing entrypoint in this task; chat will switch over in a later task.

- [ ] **Step 16.5: Run tests, verify pass**

Run: `pytest tests/test_service_dispatch.py -v`
Expected: 2 passed.

- [ ] **Step 16.6: Run full suite**

Run: `python -m pytest tests/ -x -q`
Expected: all green.

- [ ] **Step 16.7: Commit**

```bash
git add src/generator/service.py tests/test_service_dispatch.py
git commit -m "feat(service): add RAGService.run_agent dispatching on orchestrator flag"
```

---

## Task 17: Add tracer events from orchestrator stages

**Files:**
- Modify: `src/agent/orchestrator.py`
- Test: `tests/test_orchestrator.py`

- [ ] **Step 17.1: Append failing test asserting trace events**

Append to `tests/test_orchestrator.py`:

```python
def test_orchestrator_emits_phase_trace_events(monkeypatch):
    chunks = [{"chunk_id": f"k{i}", "document_id": f"d{i}"} for i in range(4)]
    agent = _agent(
        monkeypatch,
        plan_collections=("col_a", "col_b"),
        result_chunks_by_collection={"col_a": chunks[:2], "col_b": chunks[2:]},
    )
    out = agent.run("q", session_collections=["col_a", "col_b"])
    phases = {e.phase for e in out.trace}
    assert "planning" in phases
    assert "policy" in phases
    assert "allocation" in phases
    assert "retrieval" in phases
    assert "assembly" in phases
    assert "judge" in phases
    assert "answering" in phases
    assert "citation" in phases
```

- [ ] **Step 17.2: Run failing test**

Run: `pytest tests/test_orchestrator.py::test_orchestrator_emits_phase_trace_events -v`
Expected: FAIL — trace empty.

- [ ] **Step 17.3: Wire tracer events into orchestrator stages**

In `src/agent/orchestrator.py`, after each stage in `run()`, append a tracer event. Helper:

```python
def _emit(self, tracer: PipelineTracer, phase: str, t0: float, details: dict) -> None:
    from src.agent.schemas import AgentTraceEvent
    tracer.events.append(AgentTraceEvent(
        trace_id=tracer.events[0].trace_id if tracer.events else phase,
        phase=phase,
        latency_ms=(time.perf_counter() - t0) * 1000,
        details=details,
    ))
```

Insert measured `t0 = time.perf_counter()` before each stage call and `self._emit(tracer, "<phase>", t0, {...})` after. Phases to emit: `planning`, `policy`, `allocation`, `retrieval`, `assembly`, `judge`, `expansion` (when run), `judge_post_expand` (when expanded), `answering`, `validation`, `citation`. Details payloads as enumerated in spec §13.

Example for `policy`:

```python
t0 = time.perf_counter()
self._policy.run(state, session_collections)
self._emit(tracer, "policy", t0, {
    "allowed": state.policy_result.allowed_collections,
    "denied": state.policy_result.denied_collections,
    "reason_by_collection": state.policy_result.reason_by_collection,
})
```

(Repeat per stage.) Use `PipelineTracer` API. If the existing tracer has helper methods like `tracer.record(phase=..., latency_ms=..., details=...)`, prefer those over directly appending to `tracer.events`. Confirm by reading `src/agent/tracer.py` and adjust.

- [ ] **Step 17.4: Run trace test, verify pass**

Run: `pytest tests/test_orchestrator.py::test_orchestrator_emits_phase_trace_events -v`
Expected: PASS.

- [ ] **Step 17.5: Run all orchestrator tests**

Run: `pytest tests/test_orchestrator.py -v`
Expected: 5 passed.

- [ ] **Step 17.6: Commit**

```bash
git add src/agent/orchestrator.py tests/test_orchestrator.py
git commit -m "feat(agent): emit per-stage trace events from OrchestratorAgent"
```

---

## Task 18: Wire streaming through orchestrator

**Files:**
- Modify: `src/agent/orchestrator.py`
- Modify: `src/generator/service.py`

- [ ] **Step 18.1: Inspect existing stream contract**

Run: `grep -n "stream\|yield\|ask_stream" src/agent/tools.py src/generator/service.py | head -30`
Confirm `AnswerTool` exposes a `stream()` method. If only `answer()` exists, defer streaming to a follow-up task and stop here.

- [ ] **Step 18.2: Add `stream_callback` parameter to `OrchestratorAgent.run()`**

In `src/agent/orchestrator.py`, change the `run` signature to:

```python
def run(
    self,
    query: str,
    session_collections: list[str],
    stream_callback: Optional[callable] = None,
) -> AgentOutput:
```

In the answering block, when `stream_callback` is provided, replace the `answer()` call with token-by-token streaming:

```python
if stream_callback is not None and hasattr(self._answer_tool, "stream"):
    thinking_parts: list[str] = []
    answer_parts: list[str] = []
    for event in self._answer_tool.stream(query=query, context=context):
        # event shape mirrors existing chat.py consumer; reuse it.
        if event.get("type") == "token":
            answer_parts.append(event["text"])
            stream_callback(event)
        elif event.get("type") == "thinking":
            thinking_parts.append(event["text"])
    thinking = "".join(thinking_parts)
    answer = "".join(answer_parts)
else:
    thinking, answer = self._answer_tool.answer(query=query, context=context)
```

Run `_search_tool` and `_answer_tool` exactly as the legacy `PlanningAgent` does — copy the call shape if details differ from the above placeholder.

- [ ] **Step 18.3: Add streaming pass-through to `RAGService.run_agent`**

In `src/generator/service.py`, change `run_agent` signature:

```python
def run_agent(
    self,
    query: str,
    session_collections: list[str],
    stream_callback=None,
):
    if self._config.orchestrator.enabled:
        return self._orchestrator.run(query, session_collections, stream_callback=stream_callback)
    return self._planning_agent.run(query)
```

- [ ] **Step 18.4: Run full suite, ensure no regressions**

Run: `python -m pytest tests/ -x -q`
Expected: all green. (No new unit tests in this task — streaming is exercised manually via chat.py in the next task; full pipeline test is in Task 19.)

- [ ] **Step 18.5: Commit**

```bash
git add src/agent/orchestrator.py src/generator/service.py
git commit -m "feat(agent): wire stream_callback through OrchestratorAgent and RAGService"
```

---

## Task 19: Add evaluator smoke test for orchestrator

**Files:**
- Create: `tests/test_orchestrator_smoke.py`

- [ ] **Step 19.1: Write smoke test (skipped unless `RUN_ORCHESTRATOR_SMOKE=1`)**

Create `tests/test_orchestrator_smoke.py`:

```python
"""Integration smoke test for OrchestratorAgent against real ChromaDB + Ollama.

Skipped by default. Run with: RUN_ORCHESTRATOR_SMOKE=1 pytest tests/test_orchestrator_smoke.py -v
"""
from __future__ import annotations

import os

import pytest

from src.common.llm_client_pool import LLMClientPool
from src.agent.orchestrator import OrchestratorAgent
from src.config.pipeline_loader import load_pipeline_config


pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_ORCHESTRATOR_SMOKE") != "1",
    reason="Smoke test requires Ollama + ChromaDB; opt in with RUN_ORCHESTRATOR_SMOKE=1",
)


def test_orchestrator_answers_real_question():
    cfg = load_pipeline_config()
    cfg.orchestrator.enabled = True
    pool = LLMClientPool.from_config(cfg)
    agent = OrchestratorAgent(cfg, pool)

    session = cfg.get_collection_keys()[:2]   # first two registered collections
    out = agent.run("1997 yılında ne oldu?", session_collections=session)

    assert out.answer
    assert out.evidence_decision is not None
    assert out.policy_result is not None
    assert len(out.sources) > 0
    # Trace must include every key phase
    phases = {e.phase for e in out.trace}
    assert {"planning", "policy", "allocation", "retrieval", "assembly", "judge"}.issubset(phases)
```

- [ ] **Step 19.2: Verify the test is collected and skipped under default env**

Run: `pytest tests/test_orchestrator_smoke.py -v`
Expected: 1 skipped.

- [ ] **Step 19.3: (Optional) Run smoke against running Ollama**

Run: `RUN_ORCHESTRATOR_SMOKE=1 pytest tests/test_orchestrator_smoke.py -v`
Expected: PASS if Ollama + ChromaDB are reachable. If not, document in commit message that the smoke was skipped.

- [ ] **Step 19.4: Commit**

```bash
git add tests/test_orchestrator_smoke.py
git commit -m "test(agent): add gated smoke test for orchestrator against real backends"
```

---

## Task 20: Switch chat.py to `RAGService.run_agent`

**Files:**
- Modify: `src/ui/chat.py`

- [ ] **Step 20.1: Locate existing agent invocation**

Run: `grep -n "PlanningAgent\|run_agent\|process(" src/ui/chat.py`
Identify the line where chat invokes the agent.

- [ ] **Step 20.2: Replace with `RAGService.run_agent`**

Substitute the agent invocation with the dispatch method. The exact replacement depends on what `chat.py` currently calls; the contract:

- Pass `query` (user's input)
- Pass `session_collections` (already available in chat from the startup selector — locate the variable name with `grep -n "session_collections\|selected_collections" src/ui/chat.py`)
- Pass `stream_callback=<existing token printer>` if available; else omit

Example shape (adapt names):

```python
output = self._service.run_agent(
    query=user_input,
    session_collections=self._selected_collections,
    stream_callback=self._on_token,
)
```

After the call, render `output.answer`, then iterate `output.sources` for the citations panel, then surface `output.evidence_decision` and `output.expanded` if the chat UI shows debug info.

- [ ] **Step 20.3: Smoke test manually**

Start chat: `python chat.py` and ask a query. With `orchestrator.enabled: false` you get the legacy path. Flip to `true` in `pipeline.yaml`, restart, query again — same path now hits `OrchestratorAgent`.

- [ ] **Step 20.4: Commit**

```bash
git add src/ui/chat.py
git commit -m "feat(ui): switch chat to RAGService.run_agent dispatch"
```

---

## Task 21: Add Turkish UX strings for judge spinners

**Files:**
- Modify: `src/ui/chat.py`

- [ ] **Step 21.1: Locate chat status / spinner pattern**

Run: `grep -n "with.*status\|console.status\|spinner" src/ui/chat.py`
Identify the spinner pattern used during retrieval and answering.

- [ ] **Step 21.2: Add `Kısa değerlendirme...` and `Kanıt genişletiliyor...` indicators**

Wrap the orchestrator call in a status block. If `RAGService.run_agent` is one synchronous call, the spinner cannot reflect intermediate phases. Two options:

- **A (simple)**: Show a single `Düşünüyor...` while `run_agent` runs. Post-stream, if `output.evidence_decision.judge_type == "llm"` or `output.expanded`, log a one-line note: `(Kanıt değerlendirildi · genişletildi)`.
- **B (richer)**: Add `progress_callback` parameter to `OrchestratorAgent.run()` that fires `"judge_start"` / `"judge_end"` / `"expand_start"` / `"expand_end"` events and update spinner text in chat.

Recommended: option A (one task, no schema changes). If product wants live spinners, file follow-up.

Apply option A: in `chat.py` after `run_agent` returns and before printing the answer:

```python
notes = []
if output.evidence_decision and output.evidence_decision.judge_type == "llm":
    notes.append("kanıt değerlendirildi")
if output.expanded:
    notes.append("genişletildi")
if notes:
    self._console.print(f"[dim]({' · '.join(notes)})[/dim]")
```

- [ ] **Step 21.3: Manual smoke**

Run chat with `orchestrator.enabled: true`. Verify the note appears when the heuristic fails and the LLM judge fires, or when expansion happens.

- [ ] **Step 21.4: Commit**

```bash
git add src/ui/chat.py
git commit -m "feat(ui): surface judge_type=llm and expansion in chat output"
```

---

## Task 22: Self-review and migration handoff

- [ ] **Step 22.1: Full test suite green**

Run: `python -m pytest tests/ -q`
Expected: all green.

- [ ] **Step 22.2: Flag flip dry run**

Edit `pipeline.yaml`: set `orchestrator.enabled: true`. Run: `python -m pytest tests/test_service_dispatch.py tests/test_orchestrator.py -v`
Expected: both files green. Revert the YAML change for the merge commit (leave the default `false` for the deprecation window per spec §11).

- [ ] **Step 22.3: Verify nothing in the orchestrator path imports legacy retry helpers**

Run: `grep -n "_needs_reretrieval\|_needs_quality_reretrieval\|_generate_broader_plan\|_generate_gap_fill_plan" src/agent/orchestrator.py`
Expected: no matches.

- [ ] **Step 22.4: Confirm spec §11 step 1 complete**

Spec §11 step 1: "Add infra without flipping flag. Land schemas, new modules, YAML keys with `orchestrator.enabled: false`. All new tests pass; existing tests unchanged."

Run: `git log --oneline feature/agent-pipeline | head -25` and verify all 20+ task commits are present.

- [ ] **Step 22.5: Commit migration handoff note (optional)**

Create `docs/superpowers/plans/2026-05-24-agentic-orchestrator-handoff.md` only if the next step is owned by a different engineer; otherwise skip. The deprecation cleanup (spec §11 steps 6-8) is intentionally outside this plan and tracked separately.

```bash
# Skip the handoff doc unless explicitly needed.
```

---

## Out of Scope (do NOT do in this plan)

- Guardrails (input/output/grounding/toxicity) — `docs/superpowers/specs/2026-05-24-guardrails-design.md`
- Deletion of `PlanningAgent.run()` body and aliasing to `OrchestratorAgent.run()` — spec §11 step 8, a future release
- Removal of deprecated `re_retrieved` / `quality_re_retrieved` `AgentOutput` fields — spec §11 step 8
- `query_type` LLM emission in the planner prompt — defaults to `"fact"` for this plan; planner-prompt update is a follow-up
- New evaluator goldens to compare flag on/off — spec §11 step 5, follow-up
