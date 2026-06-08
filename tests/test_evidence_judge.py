"""Unit tests for EvidenceJudge heuristic decision path."""
from __future__ import annotations

import pytest

from src.agent.judge import EvidenceJudge
from src.agent.schemas import Chunk, OrchestratorState
from src.config.pipeline_loader import JudgeConfig


def _chunk(cid: str, collection: str) -> Chunk:
    return Chunk(
        chunk_id=cid,
        document_id=f"d{cid}",
        collection_name=collection,
        doc_type="gazete",
        source_title="t",
        text="body",
        score=0.5,
        rerank_score=0.5,
    )


def _judge(llm_enabled: bool = False) -> EvidenceJudge:
    cfg = JudgeConfig({
        "heuristic": {"min_chunks": 4, "min_collection_coverage": 2},
        "llm": {"enabled": llm_enabled, "borderline_band": [2, 4]},
        "max_expand_iterations": 1,
    })
    return EvidenceJudge(cfg, client_pool=None)


def test_judge_no_chunks_clarify():
    state = OrchestratorState(request_id="r", user_query="q", assembled_chunks=[])
    _judge().run(state)
    assert state.evidence_decision.action == "clarify"
    assert state.evidence_decision.judge_type == "heuristic"
    assert "no_results" in state.evidence_decision.missing_aspects


def test_judge_heuristic_pass_with_enough_chunks_and_coverage():
    chunks = [_chunk(str(i), "c1" if i < 3 else "c2") for i in range(5)]
    state = OrchestratorState(request_id="r", user_query="q", assembled_chunks=chunks)
    _judge().run(state)
    assert state.evidence_decision.action == "answer"
    assert state.evidence_decision.sufficient is True
    assert state.evidence_decision.judge_type == "heuristic"


def test_judge_heuristic_expand_when_below_threshold_and_llm_disabled():
    chunks = [_chunk(str(i), "c1") for i in range(5)]
    state = OrchestratorState(request_id="r", user_query="q", assembled_chunks=chunks)
    _judge(llm_enabled=False).run(state)
    assert state.evidence_decision.action == "expand"
    assert state.evidence_decision.judge_type == "heuristic"


class _FakeLLMClient:
    """Returns a fixed chat response payload to drive judge decisions."""

    def __init__(self, response_text: str) -> None:
        self.response_text = response_text
        self.calls: list[dict] = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        return {"message": {"content": self.response_text}}


class _FakeLLMPool:
    """Minimal LLMClientPool stand-in for tests."""

    def __init__(self, client: _FakeLLMClient) -> None:
        self._client = client

    def get_client(self, block: str):
        return self._client

    def get_model_for_block(self, block: str, model_key: str) -> str:
        return f"fake-{model_key}"


def _judge_with_llm(response_text: str) -> tuple[EvidenceJudge, _FakeLLMClient]:
    cfg = JudgeConfig({
        "heuristic": {"min_chunks": 4, "min_collection_coverage": 2},
        "llm": {
            "enabled": True,
            "borderline_band": [2, 4],
            "block": "fast-01",
            "model_key": "judge",
        },
    })
    client = _FakeLLMClient(response_text)
    pool = _FakeLLMPool(client)
    return EvidenceJudge(cfg, client_pool=pool), client


def test_judge_llm_path_returns_answer_action():
    judge, client = _judge_with_llm(
        '{"sufficient": true, "confidence": 0.7, "action": "answer", "missing_aspects": []}'
    )
    chunks = [_chunk(str(i), "c1") for i in range(3)]
    state = OrchestratorState(request_id="r", user_query="q", assembled_chunks=chunks)
    judge.run(state)
    assert state.evidence_decision.action == "answer"
    assert state.evidence_decision.judge_type == "llm"
    assert len(client.calls) == 1


def test_judge_llm_invalid_json_falls_back_to_heuristic_expand():
    judge, _ = _judge_with_llm("not json")
    chunks = [_chunk(str(i), "c1") for i in range(3)]
    state = OrchestratorState(request_id="r", user_query="q", assembled_chunks=chunks)
    judge.run(state)
    assert state.evidence_decision.action == "expand"
    assert state.evidence_decision.judge_type == "heuristic"
