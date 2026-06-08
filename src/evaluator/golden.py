"""Golden-answer regression: compare new answers against stored baselines via LLM judge."""
from __future__ import annotations

from src.evaluator.judge import LLMJudge


def compare_to_golden(
    query: str,
    context: str,
    new_answer: str,
    golden_answer: str,
    judge: LLMJudge | None = None,
) -> dict:
    """Score new_answer against golden_answer using the LLM judge for semantic equivalence.

    Returns a dict with judge scores plus a 'regression' flag when the new answer
    scores 1+ points lower on 'relevance' or 'faithfulness' vs. the golden baseline.
    """
    if judge is None:
        judge = LLMJudge()

    new_scores = judge.score(query, context, new_answer)
    golden_scores = judge.score(query, context, golden_answer)

    regression = (
        new_scores.get("relevance", 0) < golden_scores.get("relevance", 0) - 1
        or new_scores.get("faithfulness", 0) < golden_scores.get("faithfulness", 0) - 1
    )

    return {
        "query": query,
        "new_scores": new_scores,
        "golden_scores": golden_scores,
        "regression": regression,
    }
