"""Integration smoke test for OrchestratorAgent against real ChromaDB + Ollama.

Skipped by default. Run with: RUN_ORCHESTRATOR_SMOKE=1 pytest tests/test_orchestrator_smoke.py -v
"""
from __future__ import annotations

import os

import pytest

from src.agent.orchestrator import OrchestratorAgent
from src.common.llm_client_pool import LLMClientPool
from src.config.pipeline_loader import load_pipeline_config


pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_ORCHESTRATOR_SMOKE") != "1",
    reason="Smoke test requires Ollama + ChromaDB; opt in with RUN_ORCHESTRATOR_SMOKE=1",
)


def test_orchestrator_answers_real_question():
    cfg = load_pipeline_config()
    cfg.orchestrator.enabled = True
    pool = LLMClientPool.from_config(cfg)
    agent = OrchestratorAgent(cfg, pool)

    session = cfg.get_collection_keys()[:2]
    out = agent.run("1997 yılında ne oldu?", session_collections=session)

    assert out.answer
    assert out.evidence_decision is not None
    assert out.policy_result is not None
    assert len(out.sources) > 0
    phases = {e.phase for e in out.trace}
    assert {"planning", "policy", "allocation", "retrieval", "assembly", "judge"}.issubset(phases)
