"""Unit tests for ScopeClassifier with a mocked LLMClientPool."""
from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.agent.classifier import ScopeClassifier
from src.agent.schemas import ScopeResult
from src.agent.tracer import PipelineTracer


def _mock_pool(response_json: dict):
    pool = MagicMock()
    client = MagicMock()
    response = SimpleNamespace(
        message=SimpleNamespace(content=json.dumps(response_json))
    )
    client.chat.return_value = response
    pool.get_client.return_value = client
    pool.get_model_for_block.return_value = "qwen2.5:3b-instruct"
    return pool, client


def _mock_config(
    enabled: bool = True,
    threshold: float = 0.6,
    prompt: str = "Sen bir kapı bekçisisin.",
):
    classifier_cfg = SimpleNamespace(
        enabled=enabled,
        block="fast-01",
        model_key="classifier",
        temperature=0.0,
        confidence_threshold=threshold,
        think=False,
        prompt=prompt,
    )
    block_cfg = SimpleNamespace(max_num_predict=512)
    cfg = SimpleNamespace(
        classifier=classifier_cfg,
        get_block=lambda name: block_cfg,
    )
    return cfg


def test_classifier_returns_in_scope():
    cfg = _mock_config()
    pool, _ = _mock_pool({"scope": "in_scope", "confidence": 0.95, "reason": "siyasi"})
    classifier = ScopeClassifier(pool, cfg)

    result = classifier.classify("Özal döneminde gazete manşetleri", PipelineTracer())

    assert isinstance(result, ScopeResult)
    assert result.scope == "in_scope"
    assert result.confidence == 0.95
    assert "siyasi" in result.reason


def test_classifier_returns_off_domain():
    cfg = _mock_config()
    pool, _ = _mock_pool({"scope": "off_domain", "confidence": 0.9, "reason": "hava durumu"})
    classifier = ScopeClassifier(pool, cfg)

    result = classifier.classify("hava bugün nasıl", PipelineTracer())

    assert result.scope == "off_domain"
    assert result.confidence == 0.9


def test_classifier_records_trace_phase():
    cfg = _mock_config()
    pool, _ = _mock_pool({"scope": "off_domain", "confidence": 0.8, "reason": "x"})
    classifier = ScopeClassifier(pool, cfg)
    tracer = PipelineTracer()

    classifier.classify("test", tracer)

    phases = [e.phase for e in tracer.events]
    assert "classification" in phases
    cls_event = next(e for e in tracer.events if e.phase == "classification")
    assert cls_event.block == "fast-01"
    assert cls_event.details.get("scope") == "off_domain"


def test_classifier_fail_open_on_llm_exception():
    cfg = _mock_config()
    pool, client = _mock_pool({"scope": "in_scope", "confidence": 0.0, "reason": ""})
    client.chat.side_effect = RuntimeError("ollama down")
    classifier = ScopeClassifier(pool, cfg)

    result = classifier.classify("test", PipelineTracer())

    # Fail open → in_scope with confidence 0 so the caller will not bail to off-domain
    assert result.scope == "in_scope"
    assert result.confidence == 0.0


def test_classifier_fail_open_on_invalid_json():
    cfg = _mock_config()
    pool, client = _mock_pool({})  # placeholder
    client.chat.return_value = SimpleNamespace(message=SimpleNamespace(content="not json"))
    classifier = ScopeClassifier(pool, cfg)

    result = classifier.classify("test", PipelineTracer())

    assert result.scope == "in_scope"
    assert result.confidence == 0.0
