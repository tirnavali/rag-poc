"""Unit tests for orchestrator-related Pydantic schemas."""
from __future__ import annotations

import pytest

from src.agent.schemas import (
    SearchPlan,
    CollectionSearchPlan,
    SearchQueryDraft,
    Chunk,
    RetrievalOutput,
    PolicyResult,
    CollectionExecutionPlan,
    ContextAssemblyItem,
    EvidenceDecision,
    OrchestratorState,
    AgentOutput,
)


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
