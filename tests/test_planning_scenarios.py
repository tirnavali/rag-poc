"""Real-LLM golden test for the planning agent gate flow.

Marked @pytest.mark.slow because each scenario issues at least one Ollama
call. CI should run this with `pytest -m slow`. Default `pytest tests/`
skips it.

Run: pytest tests/test_planning_scenarios.py -m slow -v
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from src.agent.planner import PlanningAgent
from src.agent.schemas import SearchPlan
from src.common.llm_client_pool import LLMClientPool
from src.config.pipeline_loader import load_pipeline_config

FIXTURE_PATH = Path(__file__).parent / "golden" / "planning_scenarios.yaml"


def _load_scenarios() -> list[dict[str, Any]]:
    with open(FIXTURE_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def agent() -> PlanningAgent:
    config = load_pipeline_config()
    if config is None:
        pytest.skip("pipeline.yaml missing")
    pool = LLMClientPool(config)
    return PlanningAgent(config, pool)


def _assert_filters_match(plan: SearchPlan, expected: dict[str, Any]) -> None:
    """Each expected key/value must appear in at least one query_draft's filters."""
    all_filters: list[dict[str, Any]] = []
    for resource in plan.resources:
        for draft in resource.query_drafts:
            if draft.filters:
                all_filters.append(dict(draft.filters))

    for key, value in expected.items():
        if key == "author_contains":
            found = any(
                str(f.get("author", "")).lower().find(value.lower()) >= 0
                for f in all_filters
            )
            assert found, f"no draft filter has author containing {value!r}; got {all_filters}"
        else:
            found = any(f.get(key) == value for f in all_filters)
            assert found, f"no draft filter has {key}={value}; got {all_filters}"


@pytest.mark.slow
@pytest.mark.parametrize("scenario", _load_scenarios(), ids=lambda s: s["id"])
def test_planning_scenario(scenario: dict[str, Any], agent: PlanningAgent) -> None:
    output = agent.run(scenario["query"])
    expect = scenario["expect"]

    assert output.scope == expect["scope"], (
        f"{scenario['id']}: expected scope={expect['scope']!r}, got {output.scope!r}"
    )

    if expect["scope"] == "bad_word":
        if "answer_contains" in expect:
            assert expect["answer_contains"] in output.answer, (
                f"{scenario['id']}: answer missing {expect['answer_contains']!r}; got: {output.answer[:200]}"
            )
        assert output.plan is None
        assert output.suggestions == []
        return

    if expect["scope"] == "off_domain":
        if "suggestions_count" in expect:
            assert len(output.suggestions) == expect["suggestions_count"], (
                f"{scenario['id']}: expected {expect['suggestions_count']} suggestions, got {len(output.suggestions)}"
            )
        if "answer_contains" in expect:
            assert expect["answer_contains"] in output.answer, (
                f"{scenario['id']}: answer missing {expect['answer_contains']!r}"
            )
        return

    # in_scope assertions
    plan = output.plan
    assert plan is not None, f"{scenario['id']}: in_scope but no plan produced"

    if "intent" in expect:
        assert plan.intent == expect["intent"], (
            f"{scenario['id']}: expected intent={expect['intent']!r}, got {plan.intent!r}"
        )

    if "collections_any_of" in expect:
        got = {r.collection for r in plan.resources}
        expected_set = set(expect["collections_any_of"])
        assert got & expected_set, (
            f"{scenario['id']}: planner picked {got}, expected at least one of {expected_set}"
        )

    if "collections_min_count" in expect:
        distinct = len({r.collection for r in plan.resources})
        assert distinct >= expect["collections_min_count"], (
            f"{scenario['id']}: only {distinct} distinct collections, want ≥ {expect['collections_min_count']}"
        )

    if "filters" in expect:
        _assert_filters_match(plan, expect["filters"])
