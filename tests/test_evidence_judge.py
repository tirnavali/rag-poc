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


def _judge(llm_enabled: bool = False, min_rerank_score: float = 0.0) -> EvidenceJudge:
    cfg = JudgeConfig({
        "heuristic": {"min_chunks": 4, "min_collection_coverage": 2,
                      "min_rerank_score": min_rerank_score},
        "llm": {"enabled": llm_enabled, "borderline_band": [2, 4]},
        "max_expand_iterations": 1,
    })
    return EvidenceJudge(cfg, client_pool=None)


def test_judge_relevance_floor_drops_irrelevant_chunks():
    """Chunks below min_rerank_score are dropped (very irrelevant), informative kept."""
    keep = _chunk("a", "col_a")          # rerank_score 0.5
    drop = _chunk("b", "col_b")
    drop.rerank_score = 0.001            # below floor
    state = OrchestratorState(request_id="r", user_query="q", assembled_chunks=[keep, drop])
    _judge(min_rerank_score=0.02).run(state)
    ids = {c.chunk_id for c in state.assembled_chunks}
    assert ids == {"a"}
    assert any(e.startswith("relevance_floor_dropped") for e in state.errors)


def test_judge_relevance_floor_all_below_signals_insufficiency():
    """If EVERY chunk is below the floor, this is the verified signature of a
    vocabulary/term mismatch (structurally similar but off-topic matches) —
    treat it as insufficient evidence (action=expand) instead of silently
    keeping the irrelevant chunk set (the count-based check below can't see
    relevance at all, so this is the only place this failure mode is caught)."""
    c1 = _chunk("a", "col_a"); c1.rerank_score = 0.001
    c2 = _chunk("b", "col_b"); c2.rerank_score = 0.001
    state = OrchestratorState(request_id="r", user_query="q", assembled_chunks=[c1, c2])
    _judge(min_rerank_score=0.02).run(state)
    assert state.evidence_decision.action == "expand"
    assert state.evidence_decision.judge_type == "heuristic"
    assert "low_relevance_all_chunks" in state.evidence_decision.missing_aspects
    # Chunks are left untouched (not silently trimmed) — expand path decides what's next.
    assert len(state.assembled_chunks) == 2


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
    """Returns a fixed chat response payload to drive judge decisions.

    Mirrors the real BlockClient: chat() accepts only the supported kwargs
    (model/messages/options/format/think) — NOT `timeout` — and returns an
    object with attribute access (``.message.content``), like ollama. An earlier
    fake accepted **kwargs and returned a dict, which masked a production
    TypeError (judge passed an unsupported `timeout` kwarg) so the LLM judge
    silently never ran.
    """

    def __init__(self, response_text: str) -> None:
        self.response_text = response_text
        self.calls: list[dict] = []

    def chat(self, *, model, messages, options=None, format=None, think=None):
        import types
        self.calls.append({"model": model, "messages": messages,
                           "options": options, "format": format, "think": think})
        return types.SimpleNamespace(
            message=types.SimpleNamespace(content=self.response_text)
        )


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
    # Lock the bug fix: call must use the supported, capped, JSON-formatted shape.
    call = client.calls[0]
    assert "timeout" not in call
    assert call["format"] == "json"
    assert call["think"] is False
    assert call["options"]["num_predict"] <= 256


def test_judge_llm_invalid_json_falls_back_to_heuristic_expand():
    judge, _ = _judge_with_llm("not json")
    chunks = [_chunk(str(i), "c1") for i in range(3)]
    state = OrchestratorState(request_id="r", user_query="q", assembled_chunks=chunks)
    judge.run(state)
    assert state.evidence_decision.action == "expand"
    assert state.evidence_decision.judge_type == "heuristic"


def _weak_chunks(n: int, score: float = 0.1) -> list[Chunk]:
    """Enough count/coverage to pass the heuristic, but every match is weak —
    the exact shape that motivated llm_escalation_score (see judge.py)."""
    chunks = [_chunk(str(i), "c1" if i < n // 2 else "c2") for i in range(n)]
    for c in chunks:
        c.rerank_score = score
    return chunks


def test_judge_weak_max_score_escalates_to_llm_despite_sufficient_count():
    """Count/coverage look fine (5 chunks, 2 collections) but the best match is
    weak (0.1 < default 0.3 escalation) — must NOT auto-answer; hands off to the
    LLM judge to actually read the content instead of guessing via a number."""
    judge, client = _judge_with_llm(
        '{"sufficient": true, "confidence": 0.6, "action": "answer", "missing_aspects": []}'
    )
    state = OrchestratorState(request_id="r", user_query="q", assembled_chunks=_weak_chunks(5))
    judge.run(state)
    assert state.evidence_decision.judge_type == "llm"
    assert len(client.calls) == 1


def test_judge_weak_max_score_falls_back_to_answer_when_llm_disabled():
    """Same weak-match shape, but LLM judge disabled — no better signal available,
    so it falls back to the pre-existing heuristic 'answer' behavior (not a
    regression; just can't do better without the LLM)."""
    judge = _judge(llm_enabled=False)
    state = OrchestratorState(request_id="r", user_query="q", assembled_chunks=_weak_chunks(5))
    judge.run(state)
    assert state.evidence_decision.action == "answer"
    assert state.evidence_decision.judge_type == "heuristic"


def test_judge_strong_max_score_skips_llm_escalation():
    """A comfortably strong best match (>= escalation threshold) stays on the
    fast heuristic path — no LLM call for the common/clear-cut case."""
    judge, client = _judge_with_llm('{"sufficient": true, "action": "answer"}')
    chunks = [_chunk(str(i), "c1" if i < 3 else "c2") for i in range(5)]  # rerank_score=0.5
    state = OrchestratorState(request_id="r", user_query="q", assembled_chunks=chunks)
    judge.run(state)
    assert state.evidence_decision.action == "answer"
    assert state.evidence_decision.judge_type == "heuristic"
    assert len(client.calls) == 0
