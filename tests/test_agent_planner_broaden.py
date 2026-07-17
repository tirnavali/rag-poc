"""Unit tests for Planner.broaden() re-query mechanics: real previous-round
context (result_count/missing_aspects) reaching the LLM prompt instead of the
old hardcoded result_count=0 placeholder, negative-constraint injection for
previously-rejected term hypotheses, and term_hypothesis round-tripping from
the LLM's JSON response into SearchPlan.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock

from src.agent.planner import Planner
from src.agent.schemas import CollectionSearchPlan, SearchPlan, SearchQueryDraft
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


def _previous_plan() -> SearchPlan:
    return SearchPlan(
        intent="unknown", query_type="comprehensive",
        resources=[CollectionSearchPlan(collection="col_a", query_drafts=[SearchQueryDraft(text="q", top_k=5)])],
        reasoning="r",
    )


def test_broaden_passes_real_result_count_not_zero():
    """Regression: _generate_broader_plan used to hardcode result_count=0
    regardless of the actual previous round's outcome — the LLM reasoned
    completely blind. Now the real count must reach the prompt."""
    planner = _planner()
    client = _mock_client({"intent": "unknown", "resources": [], "reasoning": "r"})
    planner._pool.get_client = MagicMock(return_value=client)

    planner.broaden("kadük tüm listeyi ver", _previous_plan(), result_count=38)

    sys_prompt = client.chat.call_args.kwargs["messages"][0]["content"]
    assert "38 sonuç" in sys_prompt


def test_broaden_includes_missing_aspects_reason():
    planner = _planner()
    client = _mock_client({"intent": "unknown", "resources": [], "reasoning": "r"})
    planner._pool.get_client = MagicMock(return_value=client)

    planner.broaden(
        "kadük tüm listeyi ver", _previous_plan(),
        result_count=38, missing_aspects=["low_relevance_all_chunks"],
    )

    sys_prompt = client.chat.call_args.kwargs["messages"][0]["content"]
    assert "low_relevance_all_chunks" in sys_prompt


def test_broaden_injects_rejected_hypotheses_as_negative_constraint():
    planner = _planner()
    client = _mock_client({"intent": "unknown", "resources": [], "reasoning": "r"})
    planner._pool.get_client = MagicMock(return_value=client)

    planner.broaden(
        "kadük tüm listeyi ver", _previous_plan(),
        rejected_hypotheses=[{"term": "kadük", "hypothesis": "yanlış karşılık"}],
    )

    sys_prompt = client.chat.call_args.kwargs["messages"][0]["content"]
    assert "yanlış karşılık" in sys_prompt
    assert "TEKRAR ÖNERME" in sys_prompt


def test_broaden_no_context_omits_optional_blocks():
    """No missing_aspects/rejected_hypotheses/tried_queries given -> prompt stays
    clean, no empty artifacts like 'Yetersizlik nedeni: .' or stray negative-
    constraint text."""
    planner = _planner()
    client = _mock_client({"intent": "unknown", "resources": [], "reasoning": "r"})
    planner._pool.get_client = MagicMock(return_value=client)

    planner.broaden("basit soru", _previous_plan())

    sys_prompt = client.chat.call_args.kwargs["messages"][0]["content"]
    assert "Yetersizlik nedeni" not in sys_prompt
    assert "TEKRAR ÖNERME" not in sys_prompt
    assert "DAHA ÖNCE ARANMIŞ SORGULAR" not in sys_prompt


def test_broaden_injects_tried_queries_as_negative_constraint():
    """Regression: at temperature 0 broaden() regenerated the SAME drafts every
    gather round (the repeat-search loop). The already-searched query texts must
    reach the prompt as a don't-repeat block."""
    planner = _planner()
    client = _mock_client({"intent": "unknown", "resources": [], "reasoning": "r"})
    planner._pool.get_client = MagicMock(return_value=client)

    planner.broaden(
        "kadük tüm listeyi ver", _previous_plan(),
        tried_queries=["sağlık personeli özlük hakları", "çalışma şartları iyileştirme"],
    )

    sys_prompt = client.chat.call_args.kwargs["messages"][0]["content"]
    assert "DAHA ÖNCE ARANMIŞ SORGULAR" in sys_prompt
    assert "- sağlık personeli özlük hakları" in sys_prompt
    assert "- çalışma şartları iyileştirme" in sys_prompt


def test_broaden_parses_term_hypothesis_from_response():
    planner = _planner()
    response = {
        "intent": "unknown",
        "resources": [
            {"collection": "col_a", "query_drafts": [{"text": "hükümsüz sayılan kanun teklifleri", "top_k": 10}]},
        ],
        "reasoning": "r",
        "term_hypothesis": {"term": "kadük", "official_phrase": "hükümsüz sayılan kanun teklifleri"},
    }
    planner._pool.get_client = MagicMock(return_value=_mock_client(response))

    plan = planner.broaden("kadük tüm listeyi ver", _previous_plan())

    assert plan.term_hypothesis is not None
    assert plan.term_hypothesis.term == "kadük"
    assert plan.term_hypothesis.official_phrase == "hükümsüz sayılan kanun teklifleri"


def test_broaden_null_term_hypothesis_is_none():
    planner = _planner()
    response = {
        "intent": "unknown",
        "resources": [{"collection": "col_a", "query_drafts": [{"text": "q", "top_k": 10}]}],
        "reasoning": "r",
        "term_hypothesis": None,
    }
    planner._pool.get_client = MagicMock(return_value=_mock_client(response))

    plan = planner.broaden("basit soru", _previous_plan())

    assert plan.term_hypothesis is None


def test_broaden_malformed_term_hypothesis_falls_back_to_none():
    """Tolerant of LLM schema slips (e.g. only 'term' present, no phrase)."""
    planner = _planner()
    response = {
        "intent": "unknown",
        "resources": [{"collection": "col_a", "query_drafts": [{"text": "q", "top_k": 10}]}],
        "reasoning": "r",
        "term_hypothesis": {"term": "kadük"},
    }
    planner._pool.get_client = MagicMock(return_value=_mock_client(response))

    plan = planner.broaden("basit soru", _previous_plan())

    assert plan.term_hypothesis is None
