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


def test_planner_includes_chat_history_hint_in_user_message():
    """chat_history must reach the planner LLM call as a topic hint so query_drafts
    can resolve anaphora ("konuyla ilgili") instead of drafting a topic-blind
    search — see orchestrator.py _node_planning wiring ctx.chat_history through."""
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
    mock_client = _mock_client(response)
    planner._pool.get_client = MagicMock(return_value=mock_client)

    chat_history = [
        {"role": "user", "content": "başkanlık sistemine geçiş ile ilgili chp'nin ilk 5 meclis konuşması"},
        {"role": "assistant", "content": "CHP milletvekillerinin başkanlık sistemi hakkındaki konuşmaları..."},
    ]
    planner.plan(
        "engin özkoç konuyla ilgili başka açıklama yaptı mı?",
        chat_history=chat_history,
    )

    user_msg = mock_client.chat.call_args.kwargs["messages"][1]["content"]
    assert "başkanlık sistemine geçiş" in user_msg
    assert user_msg.endswith("Sorgu: engin özkoç konuyla ilgili başka açıklama yaptı mı?")


def test_planner_without_chat_history_matches_old_user_message():
    """No history → no hint block; user message is exactly 'Sorgu: <query>' as before
    the chat_history param was added (backward compatible, zero behavior change)."""
    planner = _planner()
    response = {
        "intent": "factual", "query_type": "fact",
        "resources": [{"collection": "col_a", "query_drafts": [{"text": "q", "top_k": 5}]}],
        "reasoning": "r",
    }
    mock_client = _mock_client(response)
    planner._pool.get_client = MagicMock(return_value=mock_client)

    planner.plan("basit soru")

    user_msg = mock_client.chat.call_args.kwargs["messages"][1]["content"]
    assert user_msg == "Sorgu: basit soru"


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


# ------------------------------------------------------ law-strategy domain gate
# Deterministic (LLM-independent) safety net in orchestrator._node_planning: the small
# planner model can latch onto bare speech-verb triggers ("ne dedi/ne konuşuldu") and pick
# a kanun_* strategy for a non-law event/person query (the cemal kaşıkçı regression). The
# gate drops such a strategy when the query carries no law signal.

from src.agent.orchestrator import OrchestratorAgent
from src.agent.schemas import CollectionSearchPlan, ReflectionOutput, SearchPlan, SearchQueryDraft

_GATE_COL = "tutanaklar_ctx1024"


def test_is_law_query_truth_table():
    f = OrchestratorAgent._is_law_query
    # kanun/mevzuat sinyali VAR
    assert f("Karayolları Trafik Kanunu kaç oyla kabul edildi")
    assert f("5 sıra sayılı teklifin görüşmeleri")
    assert f("bütçe kanunu 3. madde")
    assert f("KANUN TEKLİFİ hakkında")            # Türkçe büyük harf → normalize_tr güvenli
    assert f("komisyon raporu muhalefet şerhi")
    # olay / kişi / gündem — kanun sinyali YOK
    assert not f("cemal kaşıkçı nın ölümü ile ilgili kim ne konuşmuştur")
    assert not f("15 temmuz hakkında kim ne konuştu")
    assert not f("başkan enflasyon hakkında ne dedi")


def test_is_law_strategy_detects_law_section_types():
    cfg = load_pipeline_config()
    is_law = OrchestratorAgent._is_law_strategy
    assert is_law(cfg.get_strategy("kanun_gorusmeleri"))
    assert is_law(cfg.get_strategy("kanun_kabul_oylama"))
    assert is_law(cfg.get_strategy("kanun_rapor_bolumu"))
    assert not is_law(cfg.get_strategy("enumerate"))
    assert not is_law(cfg.get_strategy("summarize"))
    assert not is_law(None)


def _gate_plan(strategy: str | None, query_type: str = "reasoning") -> SearchPlan:
    return SearchPlan(
        intent="analytical", query_type=query_type, strategy=strategy,
        resources=[CollectionSearchPlan(
            collection=_GATE_COL, query_drafts=[SearchQueryDraft(text="q", top_k=5)])],
        reasoning="r",
    )


def _gate_agent(monkeypatch, *, plan: SearchPlan, reflect=None) -> OrchestratorAgent:
    """Compact offline agent (mirror of test_agent_reflect._agent) — planner.plan is
    mocked to a fixed plan so the ORCHESTRATOR gate is exercised in isolation."""
    cfg = load_pipeline_config()
    cfg.policy.enabled = False
    agent = OrchestratorAgent(cfg, LLMClientPool.from_config(cfg))
    agent._classifier = None
    agent._config.clarification.enabled = False
    agent._config.judge.llm.enabled = False
    agent._search_tool._reranker = None
    monkeypatch.setattr(agent._planner, "plan", lambda q, tracer=None, **kw: plan)
    monkeypatch.setattr(agent._planner, "reflect", reflect or MagicMock())

    def _search(collection_key, query_text, filters=None, top_k=5, apply_reranker=True):
        return {
            "documents": ["body-1"],
            "metadatas": [{"chunk_id": "c1", "document_id": "d1", "doc_type": "tutanak",
                           "source_title": "t-1", "_source_collection": collection_key}],
            "distances": [0.1],
        }
    monkeypatch.setattr(agent._search_tool, "search", _search)
    monkeypatch.setattr(agent._answer_tool, "generate",
                        lambda *a, **kw: ("thinking", "Cevap metni."))
    monkeypatch.setattr(agent._sanitizer, "validate", lambda *a, **kw: None)
    return agent


def test_gate_drops_law_strategy_for_non_law_query(monkeypatch):
    # planner (yanlışlıkla) kanun_gorusmeleri döndürür; kanun-DIŞI olay sorgusu → kapı düşürür.
    reflect = MagicMock()
    agent = _gate_agent(monkeypatch, plan=_gate_plan("kanun_gorusmeleri"), reflect=reflect)
    out = agent.run("cemal kaşıkçı nın ölümü ile ilgili kim ne konuşmuştur",
                    session_collections=[_GATE_COL])
    assert out.plan.strategy is None                 # kanun stratejisi düşürüldü
    assert out.plan.query_type == "reasoning"        # strateji query_type=comprehensive override'ı OLMADI
    assert reflect.call_count == 0                   # adaptive reflect prosedürü hiç girmedi


def test_gate_preserves_law_strategy_for_law_query(monkeypatch):
    # Gerçek kanun sorgusu → kanun stratejisi KORUNUR (aşırı-kapılamıyoruz). reflect done'da
    # anında dursun diye mock'lanır (adaptive döngüyü test etmeye gerek yok — burada amaç kapı).
    reflect = MagicMock(return_value=ReflectionOutput(
        done=True, done_reason="ok", hop_cursor=1, extracted_anchors={}, next_plan=None))
    agent = _gate_agent(monkeypatch, plan=_gate_plan("kanun_gorusmeleri"), reflect=reflect)
    out = agent.run("5 sıra sayılı kanun teklifinin tüm görüşmeleri",
                    session_collections=[_GATE_COL])
    assert out.plan.strategy == "kanun_gorusmeleri"  # korundu
    assert out.plan.query_type == "comprehensive"    # strateji override'ı uygulandı
