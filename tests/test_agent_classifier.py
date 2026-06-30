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
        get_collection_catalog=lambda: "- col_x (X — örnek): doc_type=tutanak",
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


def test_classifier_parses_selected_collections():
    """IntentAnalyzer surfaces tool/db selection alongside the scope verdict."""
    cfg = _mock_config()
    pool, _ = _mock_pool({
        "scope": "in_scope", "confidence": 0.9,
        "selected_collections": ["tutanaklar_ctx1024", "gazete_arsivi"],
        "reason": "meclis",
    })
    classifier = ScopeClassifier(pool, cfg)

    result = classifier.classify("1997 bütçe görüşmeleri", PipelineTracer())

    assert result.selected_collections == ["tutanaklar_ctx1024", "gazete_arsivi"]


def test_classifier_parses_conversational():
    """Greetings/chitchat get the 'conversational' scope (whitelisted in classify)."""
    cfg = _mock_config()
    pool, _ = _mock_pool({"scope": "conversational", "confidence": 1.0, "reason": "selam"})
    classifier = ScopeClassifier(pool, cfg)

    result = classifier.classify("nasılsın?", PipelineTracer())

    assert result.scope == "conversational"
    assert result.confidence == 1.0


def test_classifier_catalog_prompt_does_not_fail_open():
    """Regression: a prompt with {catalog} AND a literal JSON example ({"scope": ...})
    must not crash. str.format() treated the JSON braces as fields and raised
    KeyError('"scope"') → fail-open in_scope/0.0 for every query. The fix uses
    .replace('{catalog}', ...). Mock returns a non-default scope so a regression
    (fail-open) is unmistakable: fail-open would give in_scope/0.0, not conversational/1.0.
    """
    prompt = (
        "Koleksiyonlar:\n{catalog}\n"
        'JSON çıktısı:\n'
        '{"scope": "in_scope" veya "conversational", "confidence": 0.0-1.0, "reason": "..."}'
    )
    cfg = _mock_config(prompt=prompt)
    pool, client = _mock_pool({"scope": "conversational", "confidence": 1.0, "reason": "selam"})
    classifier = ScopeClassifier(pool, cfg)

    result = classifier.classify("selam", PipelineTracer())

    # Not the fail-open default → .format() did not blow up on the JSON braces.
    assert result.scope == "conversational"
    assert result.confidence == 1.0
    # And the catalog was actually substituted into the system prompt sent to the LLM.
    sent_system = client.chat.call_args.kwargs["messages"][0]["content"]
    assert "{catalog}" not in sent_system
    assert "col_x" in sent_system


def test_classifier_selected_collections_defaults_empty():
    """Missing selected_collections → empty list (planner selects freely)."""
    cfg = _mock_config()
    pool, _ = _mock_pool({"scope": "in_scope", "confidence": 0.9, "reason": "x"})
    classifier = ScopeClassifier(pool, cfg)

    result = classifier.classify("q", PipelineTracer())

    assert result.selected_collections == []
