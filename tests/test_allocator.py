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
from src.common.schemas import FilterCriteria
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


def _state(
    query_type: str,
    allowed: list[str],
    filters_for: dict[str, FilterCriteria] | None = None,
) -> OrchestratorState:
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
    """Filtreler ChromaFilterTranslator ile geçerli where-dict'e çevrilir."""
    ap = AllocationPlanner(alloc_config)
    state = _state(
        "fact",
        ["c1"],
        filters_for={"c1": FilterCriteria(year=2023)},
    )
    ap.run(state)
    # Ham model_dump ({"year": 2023}) değil; Chroma operatör formatı.
    assert state.collection_plans[0].filters == {"year": {"$eq": 2023}}


def test_allocation_translates_multifield_filter_to_chroma(alloc_config):
    """Çok-alanlı filtre $and ile sarılır; yazar Türkçe büyük-harf $or'u içerir."""
    ap = AllocationPlanner(alloc_config)
    state = _state(
        "fact",
        ["c1"],
        filters_for={"c1": FilterCriteria(year=1996, author="Deniz Baykal")},
    )
    ap.run(state)
    where = state.collection_plans[0].filters
    # Geçerli Chroma where-dict: tek anahtarlı ham dump DEĞİL.
    assert "$and" in where
    conds = where["$and"]
    assert {"year": {"$eq": 1996}} in conds
    author_cond = next(c for c in conds if "$or" in c)
    authors = {o["author"]["$eq"] for o in author_cond["$or"]}
    assert "Deniz Baykal" in authors
    assert "DENİZ BAYKAL" in authors
