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
  "missing_aspects": ["..."], "reason": "kısa Türkçe gerekçe"}}
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
        h = self._config.heuristic

        # Relevance floor: drop very-irrelevant chunks (near-zero rerank score)
        # while keeping merely-informative ones.
        if h.min_rerank_score > 0.0 and state.assembled_chunks:
            kept = [c for c in state.assembled_chunks if c.rerank_score >= h.min_rerank_score]
            if not kept:
                # EVERY chunk is below the relevance floor — verified empirically as
                # the signature of a vocabulary/term mismatch (structurally-similar
                # but off-topic matches, e.g. a colloquial term the corpus phrases
                # differently). Count/coverage-based sufficiency below can't see this
                # at all; treat it as insufficiency explicitly rather than silently
                # proceeding with the full (irrelevant) chunk set.
                state.evidence_decision = EvidenceDecision(
                    sufficient=False,
                    confidence=0.2,
                    action="expand",
                    missing_aspects=["low_relevance_all_chunks"],
                    judge_type="heuristic",
                    reasoning=f"{len(state.assembled_chunks)} chunk bulundu ama hepsi düşük "
                              f"alaka skorlu (<{h.min_rerank_score}) — olası terim uyuşmazlığı.",
                )
                return state
            if len(kept) < len(state.assembled_chunks):
                dropped = len(state.assembled_chunks) - len(kept)
                state.assembled_chunks = kept
                state.balanced_context = [
                    item for item in state.balanced_context
                    if item.chunk_id in {c.chunk_id for c in kept}
                ]
                state.errors.append(f"relevance_floor_dropped:{dropped}")

        chunks = state.assembled_chunks

        if len(chunks) == 0:
            state.evidence_decision = EvidenceDecision(
                sufficient=False,
                confidence=0.0,
                action="clarify",
                missing_aspects=["no_results"],
                judge_type="heuristic",
                reasoning="Hiç sonuç bulunamadı; netleştirme gerekiyor.",
            )
            return state

        coverage = len({c.collection_name for c in chunks})
        if len(chunks) >= h.min_chunks and coverage >= h.min_collection_coverage:
            llm = self._config.llm
            max_score = max((c.rerank_score for c in chunks), default=0.0)
            weak_match = max_score < h.llm_escalation_score
            if weak_match and llm.enabled and self._pool is not None:
                # Count/coverage look fine, but even the BEST match is weak — don't
                # auto-answer blind. Escalate to the LLM judge to actually read the
                # chunk text and decide (see llm_escalation_score for why a numeric
                # reject threshold alone was rejected: it misfires on correct-but-
                # abstractly-phrased queries, not just genuine topic mismatches).
                state.evidence_decision = self._llm_judge(state)
                return state
            state.evidence_decision = EvidenceDecision(
                sufficient=True,
                confidence=0.85,
                action="answer",
                judge_type="heuristic",
                reasoning=f"{len(chunks)} chunk, {coverage} koleksiyon → yeterli (heuristik eşik).",
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
            reasoning=f"{len(chunks)} chunk yetersiz; yeniden arama (expand).",
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
            # NOTE: BlockClient.chat() has no `timeout` kwarg — passing it used to
            # raise TypeError, so this LLM judge silently fell back to heuristic
            # 'expand' on EVERY borderline case (it never actually ran). Call it
            # like the other stages: capped output, JSON format, thinking off.
            response = client.chat(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.0, "num_predict": 200},
                format="json",
                think=False,
            )
            raw = response.message.content
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
                reasoning=str(data.get("reason", "")),
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
            reasoning="LLM yargısı alınamadı; güvenli varsayılan: yeniden arama (expand).",
        )
