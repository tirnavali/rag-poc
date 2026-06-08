"""LLM-as-judge evaluator using Ollama for answer quality scoring."""
from __future__ import annotations

import json
import re

import ollama

from src.config import settings
from src.generator.prompts import JUDGE_PROMPT


class LLMJudge:
    """Score a (query, context, answer) triple using the Ollama LLM as a rubric judge.

    Runs the judge twice per call and flags if the scores diverge by > 1 on any
    dimension, which indicates model instability on this sample.
    """

    def __init__(self, model: str = settings.LLM_MODEL) -> None:
        self.model = model
        self.client = ollama.Client(host=settings.OLLAMA_HOST)

    def _call_judge(self, query: str, context: str, answer: str) -> dict:
        prompt = JUDGE_PROMPT.format(query=query, context=context[:3000], answer=answer)
        try:
            res = self.client.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.0},
            )
            raw = res.message.content.strip()
            # Strip markdown code fences if present
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
            return json.loads(raw)
        except Exception as e:
            return {"error": str(e), "faithfulness": 0, "groundedness": 0, "relevance": 0, "citation_quality": 0}

    def score(self, query: str, context: str, answer: str) -> dict:
        run1 = self._call_judge(query, context, answer)
        run2 = self._call_judge(query, context, answer)
        dimensions = ("faithfulness", "groundedness", "relevance", "citation_quality")
        noisy = any(
            abs(run1.get(d, 0) - run2.get(d, 0)) > 1
            for d in dimensions
        )
        averaged = {
            d: round((run1.get(d, 0) + run2.get(d, 0)) / 2, 2) for d in dimensions
        }
        averaged["rationale"] = run1.get("rationale", "")
        averaged["noisy"] = noisy
        return averaged
