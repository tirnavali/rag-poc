"""Grounded clarification (did-you-mean) — the Turn-Based narrowing layer.

Flow: a cheap probe retrieval surfaces ~20 hits; ``FacetMiner`` extracts real
facets (years/topics/authors/collections) from their metadata; ``AmbiguityGate``
decides whether the query is too broad; ``QueryRefiner`` phrases up to N grounded
questions whose options are the mined facet values only (never hallucinated).

The orchestrator owns the probe retrieval (it has the SearchTool); this module is
pure given the probe results, so it is trivially unit-testable offline.
"""
from __future__ import annotations

import json
import logging
from collections import Counter

from src.agent.schemas import (
    ClarificationQuestion,
    ClarificationResult,
    FacetSet,
    FacetValue,
)
from src.agent.tracer import PipelineTracer
from src.common.llm_client_pool import LLMClientPool
from src.common.llm_utils import extract_json_from_text
from src.config.pipeline_loader import PipelineConfig

logger = logging.getLogger(__name__)


_DEFAULT_QUESTION_TEXT = {
    "year": "Hangi döneme / yıla odaklanmamı istersiniz?",
    "scope": "Hangi kaynak türünde arayayım?",
    "topic": "Hangi konuya odaklanayım?",
}


def _topics_of(meta: dict) -> list[str]:
    """Normalize the `topics` metadata field to a list of strings."""
    raw = meta.get("topics")
    if not raw:
        return []
    if isinstance(raw, str):
        return [t.strip() for t in raw.replace(";", ",").split(",") if t.strip()]
    if isinstance(raw, (list, tuple)):
        return [str(t).strip() for t in raw if str(t).strip()]
    return []


class FacetMiner:
    """Deterministically mine facets from probe-retrieval metadata (no LLM)."""

    def mine(self, results: list[dict]) -> FacetSet:
        years: Counter = Counter()
        topics: Counter = Counter()
        authors: Counter = Counter()
        collections: Counter = Counter()
        total = 0

        for result in results:
            for meta in result.get("metadatas", []):
                total += 1
                year = meta.get("year")
                if isinstance(year, int):
                    years[str(year)] += 1
                elif isinstance(year, str) and year[:4].isdigit():
                    years[year[:4]] += 1
                for t in _topics_of(meta):
                    topics[t] += 1
                author = meta.get("author")
                if author:
                    authors[str(author)] += 1
                col = meta.get("_source_collection")
                if col:
                    collections[str(col)] += 1

        def _top(counter: Counter) -> list[FacetValue]:
            return [FacetValue(value=v, count=c) for v, c in counter.most_common()]

        return FacetSet(
            years=_top(years),
            topics=_top(topics),
            authors=_top(authors),
            collections=_top(collections),
            total=total,
        )


class AmbiguityGate:
    """Decides whether the probe results are spread out enough to clarify."""

    def __init__(self, config) -> None:
        self._min_distinct_years = config.min_distinct_years
        self._dominance_ratio = config.dominance_ratio

    def is_ambiguous(self, facets: FacetSet) -> bool:
        if facets.total == 0:
            return False  # nothing to narrow against; let retrieval proceed
        # Many distinct years → broad temporal scope.
        if len(facets.years) >= self._min_distinct_years:
            return True
        # No single dominant year/collection → spread across the corpus.
        for axis in (facets.years, facets.collections):
            if axis:
                top = axis[0].count
                if top / facets.total < self._dominance_ratio:
                    return True
        return False


class QueryRefiner:
    """Builds grounded questions and resolves answers into search constraints.

    Options are always the deterministically mined facet values; the LLM is used
    only to phrase the question text. On any LLM failure the default text is used.
    """

    def __init__(self, pool: LLMClientPool, config: PipelineConfig) -> None:
        self._pool = pool
        self._config = config

    def build_questions(
        self,
        query: str,
        facets: FacetSet,
        tracer: PipelineTracer,
    ) -> list[ClarificationQuestion]:
        cfg = self._config.clarification
        limit = cfg.question_count

        # Deterministic (axis, options) — only axes with more than one value.
        candidates: list[tuple[str, list[str]]] = []
        if len(facets.years) > 1:
            candidates.append(("year", [f.value for f in facets.years[:3]]))
        if len(facets.collections) > 1:
            candidates.append(("scope", [f.value for f in facets.collections[:3]]))
        if len(facets.topics) > 1:
            candidates.append(("topic", [f.value for f in facets.topics[:3]]))
        candidates = candidates[:limit]
        if not candidates:
            return []

        texts = self._phrase(query, candidates, facets, tracer)
        return [
            ClarificationQuestion(
                axis=axis,
                text=texts.get(axis, _DEFAULT_QUESTION_TEXT[axis]),
                options=options,
            )
            for axis, options in candidates
        ]

    def _phrase(
        self,
        query: str,
        candidates: list[tuple[str, list[str]]],
        facets: FacetSet,
        tracer: PipelineTracer,
    ) -> dict[str, str]:
        """One LLM call to phrase question texts; falls back to defaults."""
        cfg = self._config.clarification
        if not cfg.prompt:
            return {}
        facets_blob = "\n".join(f"- {axis}: {', '.join(opts)}" for axis, opts in candidates)
        try:
            client = self._pool.get_client(cfg.block)
            model = self._pool.get_model_for_block(cfg.block, cfg.model_key)
            block = self._config.get_block(cfg.block)
            res = client.chat(
                model=model,
                messages=[{
                    "role": "user",
                    "content": cfg.prompt.format(query=query, facets=facets_blob),
                }],
                options={"temperature": cfg.temperature, "num_predict": min(512, block.max_num_predict)},
                format="json",
                think=bool(cfg.think) if cfg.think is not None else False,
            )
            data = json.loads(extract_json_from_text(res.message.content))
            out: dict[str, str] = {}
            for q in data.get("questions", []) or []:
                axis = q.get("axis")
                text = q.get("text")
                if axis in _DEFAULT_QUESTION_TEXT and isinstance(text, str) and text.strip():
                    out[axis] = text.strip()
            return out
        except Exception as e:
            logger.warning("QueryRefiner phrasing failed (%s); using default texts", e)
            return {}

    def resolve(
        self,
        questions: list[ClarificationQuestion],
        answers: dict[str, str] | None,
    ) -> dict:
        """Map the user's per-axis answers to retrieval constraints.

        ``answers`` maps axis → chosen option string. Missing/blank = skip that
        axis ("hepsini ara"). Returns ``{"year": int, "topic": str, "collections": [..]}``.
        """
        answers = answers or {}
        constraints: dict = {}
        valid_axes = {q.axis for q in questions}
        for axis, value in answers.items():
            if axis not in valid_axes or not value:
                continue
            value = str(value).strip()
            if axis == "year" and value[:4].isdigit():
                constraints["year"] = int(value[:4])
            elif axis == "scope":
                constraints["collections"] = [value]
            elif axis == "topic":
                constraints["topic"] = value
        return constraints

    def auto_constraints(self, facets: FacetSet) -> tuple[dict, str]:
        """Non-interactive narrowing: apply the single strongest facet + a note.

        Prefers the most frequent year (clearest narrowing for this corpus); if no
        years are present, falls back to the most frequent topic.
        """
        if facets.years:
            top = facets.years[0]
            note = (
                f"Sonuçlar birden çok döneme yayılıyordu; en sık geçen {top.value} "
                f"yılına göre daraltıldı (etkileşimsiz mod)."
            )
            return {"year": int(top.value[:4])}, note
        if facets.topics:
            top = facets.topics[0]
            note = (
                f"Sonuçlar geniş bir konu yelpazesine yayılıyordu; en sık geçen "
                f"'{top.value}' konusuna göre daraltıldı (etkileşimsiz mod)."
            )
            return {"topic": top.value}, note
        return {}, ""
