"""Unit tests for Planner strategy selection (research_strategies.md playbook).

The planner's LLM call is mocked (client.chat) so these run offline; they only
verify that (a) the strategy catalog is injected into the planning prompt and
(b) the LLM's "strategy" field round-trips into SearchPlan.strategy.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock

from src.agent.planner import Planner
from src.common.llm_client_pool import LLMClientPool
from src.config.pipeline_loader import load_pipeline_config


def _mock_client(response_dict: dict) -> MagicMock:
    client = MagicMock()
    client.chat.return_value = MagicMock(message=MagicMock(content=json.dumps(response_dict)))
    return client


def _planner() -> Planner:
    cfg = load_pipeline_config()
    pool = LLMClientPool.from_config(cfg)
    return Planner(cfg, pool)


def test_planner_parses_strategy_field_from_llm_response():
    planner = _planner()
    response = {
        "intent": "factual",
        "query_type": "comprehensive",
        "strategy": "enumerate",
        "resources": [
            {"collection": "col_a", "mode": "parallel", "priority": 1,
             "query_drafts": [{"text": "q", "top_k": 5}]},
        ],
        "reasoning": "r",
    }
    planner._pool.get_client = MagicMock(return_value=_mock_client(response))

    plan = planner.plan("tüm konuşmaları listele")
    assert plan.strategy == "enumerate"


def test_planner_missing_strategy_field_defaults_to_none():
    planner = _planner()
    response = {
        "intent": "factual",
        "query_type": "fact",
        "resources": [
            {"collection": "col_a", "query_drafts": [{"text": "q", "top_k": 5}]},
        ],
        "reasoning": "r",
    }
    planner._pool.get_client = MagicMock(return_value=_mock_client(response))

    plan = planner.plan("basit soru")
    assert plan.strategy is None


def test_planner_blank_strategy_field_coerced_to_none():
    planner = _planner()
    response = {
        "intent": "factual",
        "query_type": "fact",
        "strategy": "   ",
        "resources": [
            {"collection": "col_a", "query_drafts": [{"text": "q", "top_k": 5}]},
        ],
        "reasoning": "r",
    }
    planner._pool.get_client = MagicMock(return_value=_mock_client(response))

    plan = planner.plan("basit soru")
    assert plan.strategy is None


def test_planner_injects_strategy_catalog_into_system_prompt():
    """The single planning LLM call carries the playbook catalog — no extra LLM turn."""
    planner = _planner()
    response = {"intent": "factual", "query_type": "fact", "resources": [], "reasoning": "r"}
    client = _mock_client(response)
    planner._pool.get_client = MagicMock(return_value=client)

    planner.plan("basit soru")

    assert client.chat.call_count == 1  # one call total: no separate strategy-selection turn
    sys_prompt = client.chat.call_args.kwargs["messages"][0]["content"]
    assert "enumerate" in sys_prompt  # from the real research_strategies.md catalog
