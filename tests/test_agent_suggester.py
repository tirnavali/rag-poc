"""Unit tests for Suggester with a mocked LLMClientPool."""
from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.agent.suggester import Suggester
from src.agent.tracer import PipelineTracer


def _mock_pool(content: str):
    pool = MagicMock()
    client = MagicMock()
    client.chat.return_value = SimpleNamespace(
        message=SimpleNamespace(content=content)
    )
    pool.get_client.return_value = client
    pool.get_model_for_block.return_value = "qwen2.5:3b-instruct"
    return pool, client


def _mock_config(fallbacks=None, count=3):
    suggester_cfg = SimpleNamespace(
        block="fast-01",
        model_key="suggester",
        temperature=0.3,
        think=False,
        suggestion_count=count,
        prompt="Sen bir öneri uzmanısın. Mevcut koleksiyonlar:\n{catalog}",
    )
    block_cfg = SimpleNamespace(max_num_predict=512)
    cfg = SimpleNamespace(
        suggester=suggester_cfg,
        off_domain_fallback_suggestions=fallbacks or [
            "Özal döneminde gazete manşetleri",
            "1997 TBMM bütçe görüşmeleri",
            "Susurluk skandalı haberleri",
        ],
        get_block=lambda name: block_cfg,
        get_collection_catalog=lambda: "- press_jina_v3 (Gazete)\n- tutanaklar_nomic_v2 (Tutanak)",
    )
    return cfg


def test_suggester_returns_three_strings():
    cfg = _mock_config()
    pool, _ = _mock_pool(json.dumps({
        "suggestions": [
            "1990'larda bilim haberleri",
            "TBMM bilim politikası tartışmaları",
            "Akademisyen atamaları onergeleri",
        ]
    }))
    s = Suggester(pool, cfg)

    out = s.suggest("Einstein kimdir", PipelineTracer())

    assert len(out) == 3
    assert all(isinstance(x, str) for x in out)


def test_suggester_pads_with_fallback_when_fewer_than_count():
    cfg = _mock_config()
    pool, _ = _mock_pool(json.dumps({"suggestions": ["Tek öneri"]}))
    s = Suggester(pool, cfg)

    out = s.suggest("x", PipelineTracer())

    assert len(out) == 3
    assert "Tek öneri" in out


def test_suggester_trims_when_more_than_count():
    cfg = _mock_config()
    pool, _ = _mock_pool(json.dumps({
        "suggestions": ["a", "b", "c", "d", "e"]
    }))
    s = Suggester(pool, cfg)

    out = s.suggest("x", PipelineTracer())

    assert out == ["a", "b", "c"]


def test_suggester_drops_query_echo():
    cfg = _mock_config()
    pool, _ = _mock_pool(json.dumps({
        "suggestions": ["hava bugün nasıl", "TBMM bütçe", "gazete manşetleri"]
    }))
    s = Suggester(pool, cfg)

    out = s.suggest("hava bugün nasıl", PipelineTracer())

    assert "hava bugün nasıl" not in out
    assert len(out) == 3


def test_suggester_fail_open_uses_fallbacks_on_llm_error():
    cfg = _mock_config()
    pool, client = _mock_pool("")
    client.chat.side_effect = RuntimeError("ollama down")
    s = Suggester(pool, cfg)

    out = s.suggest("x", PipelineTracer())

    assert out == cfg.off_domain_fallback_suggestions[:3]


def test_suggester_records_trace_phase():
    cfg = _mock_config()
    pool, _ = _mock_pool(json.dumps({"suggestions": ["a", "b", "c"]}))
    s = Suggester(pool, cfg)
    tracer = PipelineTracer()

    s.suggest("x", tracer)

    phases = [e.phase for e in tracer.events]
    assert "suggestion" in phases
