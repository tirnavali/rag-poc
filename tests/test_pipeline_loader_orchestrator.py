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
