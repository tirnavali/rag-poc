"""EvidenceJudge — hybrid heuristic + LLM evidence-sufficiency decision."""
from __future__ import annotations

import json
import re
from typing import Optional

from src.agent.schemas import EvidenceDecision, OrchestratorState
from src.config.pipeline_loader import JudgeConfig


_JUDGE_PROMPT = """Sen bir kanıt yeterlilik değerlendirme uzmanısın.
Aşağıdaki soruya verilen bağlam parçaları yeterli mi?

Soru: {query}
Niyet: {intent} / Sorgu tipi: {query_type}

Bağlam parçaları:
{context}

Yanıt JSON formatında ve sadece bu alanlarla:
{{"sufficient": true|false, "confidence": 0.0-1.0,
  "action": "answer"|"expand"|"clarify"|"refuse",
  "missing_aspects": ["..."]}}
"""


class EvidenceJudge:
    """Decides whether assembled chunks suffice to answer.

    Heuristic stage: chunk count + cross-collection coverage. Borderline
    cases (chunk count within `llm.borderline_band`) fall through to an LLM
    judge when configured; the LLM result drives the action. On any LLM
    failure (network, parse, schema), falls back to heuristic 'expand'.
    """

    def __init__(self, config: JudgeConfig, client_pool: Optional[object]) -> None:
        self._config = config
        self._pool = client_pool

    def run(self, state: OrchestratorState) -> OrchestratorState:
        chunks = state.assembled_chunks
        h = self._config.heuristic

        if len(chunks) == 0:
            state.evidence_decision = EvidenceDecision(
                sufficient=False,
                confidence=0.0,
                action="clarify",
                missing_aspects=["no_results"],
                judge_type="heuristic",
            )
            return state

        coverage = len({c.collection_name for c in chunks})
        if len(chunks) >= h.min_chunks and coverage >= h.min_collection_coverage:
            state.evidence_decision = EvidenceDecision(
                sufficient=True,
                confidence=0.85,
                action="answer",
                judge_type="heuristic",
            )
            return state

        llm = self._config.llm
        in_band = llm.borderline_band[0] <= len(chunks) <= llm.borderline_band[1]
        if llm.enabled and in_band and self._pool is not None:
            state.evidence_decision = self._llm_judge(state)
            return state

        state.evidence_decision = EvidenceDecision(
            sufficient=False,
            confidence=0.4,
            action="expand",
            missing_aspects=["insufficient_chunks"],
            judge_type="heuristic",
        )
        return state

    def _llm_judge(self, state: OrchestratorState) -> EvidenceDecision:
        llm = self._config.llm
        try:
            client = self._pool.get_client(llm.block)
            model = self._pool.get_model_for_block(llm.block, llm.model_key)
        except Exception:
            return self._heuristic_expand_fallback()

        intent = state.planner_output.intent if state.planner_output else "unknown"
        query_type = state.planner_output.query_type if state.planner_output else "fact"
        context = "\n".join(
            f"[{i+1}] ({c.collection_name}/{c.document_id}) {c.text[:240]}"
            for i, c in enumerate(state.assembled_chunks)
        )
        prompt = _JUDGE_PROMPT.format(
            query=state.user_query,
            intent=intent,
            query_type=query_type,
            context=context,
        )

        try:
            response = client.chat(
                messages=[{"role": "user", "content": prompt}],
                model=model,
                timeout=llm.timeout_seconds,
            )
            raw = response["message"]["content"]
        except Exception:
            return self._heuristic_expand_fallback()

        decision = self._parse_decision(raw)
        if decision is None:
            return self._heuristic_expand_fallback()
        return decision

    @staticmethod
    def _parse_decision(raw: str) -> Optional[EvidenceDecision]:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            return None
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
        try:
            return EvidenceDecision(
                sufficient=bool(data.get("sufficient", False)),
                confidence=float(data.get("confidence", 0.0)),
                action=data.get("action", "expand"),
                missing_aspects=list(data.get("missing_aspects", []) or []),
                judge_type="llm",
            )
        except Exception:
            return None

    @staticmethod
    def _heuristic_expand_fallback() -> EvidenceDecision:
        return EvidenceDecision(
            sufficient=False,
            confidence=0.4,
            action="expand",
            missing_aspects=["insufficient_chunks"],
            judge_type="heuristic",
        )
