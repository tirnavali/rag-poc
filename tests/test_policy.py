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
