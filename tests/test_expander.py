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
        reserve_chunks=[_chunk("r1", "d3", "c1"), _chunk("r2", "d2", "c1")],
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
