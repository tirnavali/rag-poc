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


def _state(
    plans: list[CollectionExecutionPlan],
    results: dict[str, RetrievalOutput],
) -> OrchestratorState:
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
