"""Facet-grounded "rabbit hole" suggestions for broad/ambiguous queries.

Flow: the orchestrator's main retrieval already surfaces the hits; ``FacetMiner``
extracts real facets (years/topics/authors/collections) from their metadata;
``AmbiguityGate`` decides whether the query is too broad; ``QueryRefiner.rabbit_holes``
builds drill-down suggestions whose values are the mined facets only (never
hallucinated). The query is NOT narrowed — suggestions are advisory.

These classes are pure given the retrieval metadata, so they are trivially
unit-testable offline. (Earlier this layer ran a separate probe retrieval and
narrowed the query with a hard year filter; both were removed.)
"""
from __future__ import annotations

import json
import logging
import re
from collections import Counter

from src.agent.schemas import (
    ClarificationQuestion,
    ClarificationResult,
    FacetSet,
    FacetValue,
)
from src.common.text import normalize_tr
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

# Explicit year in the query → the user already scoped it; don't treat as vague.
_YEAR_RE = re.compile(r"\b(18|19|20)\d{2}\b")

# Generic "X hakkında bilgi / X nedir" phrasings that signal a broad, unscoped
# information request even when the probe results happen to cluster narrowly.
_DEFAULT_VAGUE_MARKERS = (
    "bilgi", "hakkında", "hakkinda", "konusunda", "nedir", "ne demek",
    "anlat", "açıkla", "acikla", "genel", "özet", "ozet", "bahset",
)


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
        self._vague_clarify = getattr(config, "vague_query_clarify", True)
        markers = getattr(config, "vague_markers", None)
        self._vague_markers = tuple(markers) if markers else _DEFAULT_VAGUE_MARKERS

    def is_ambiguous(self, facets: FacetSet, query: str = "") -> bool:
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
        # Facet-diversity didn't fire, but the *query itself* is broad/unscoped
        # (e.g. "ohal hakkında bilgi") — clarify as long as there is at least one
        # facet axis to narrow on, so the user can pick a year/topic/source.
        if (
            self._vague_clarify
            and self._is_vague_query(query)
            and self._has_askable_facet(facets)
        ):
            return True
        return False

    def _has_askable_facet(self, facets: FacetSet) -> bool:
        return (
            len(facets.years) > 1
            or len(facets.topics) > 1
            or len(facets.collections) > 1
        )

    def _is_vague_query(self, query: str) -> bool:
        if not query:
            return False
        q = query.lower()
        if _YEAR_RE.search(q):
            return False  # explicit year → already scoped, not vague
        return any(m in q for m in self._vague_markers)


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

    def rabbit_holes(self, query: str, facets: FacetSet, count: int = 3) -> list[str]:
        """Build facet-grounded drill-down ("rabbit hole") suggestions.

        Deterministic (no LLM): combines the original query with the most frequent
        mined facet values, diversified across axes (year / topic / author) so the
        N suggestions aren't all years. Suggestions that merely echo the query or
        duplicate another are dropped. Fail-open: no usable facets → empty list
        (the orchestrator then simply shows no chips).
        """
        q = (query or "").strip()
        q_low = normalize_tr(q)

        # Per-axis candidate pools, most frequent first. Skip facet values already
        # present in the query (e.g. query "1980 ohal" shouldn't suggest "... 1980").
        # normalize_tr (not bare .lower()) so Turkish "İ" doesn't defeat this check —
        # str.lower() maps "İ" to "i̇" (combining dot), which never substring-matches
        # a plain-typed query, letting an already-named author leak back as a "new"
        # suggestion (e.g. "... ENGİN ÖZKOÇ" when the query already said "engin özkoç").
        years = [f.value for f in facets.years[:2] if f.value and normalize_tr(f.value) not in q_low]
        topics = [f.value for f in facets.topics[:2] if f.value and normalize_tr(f.value) not in q_low]
        authors = [f.value for f in facets.authors[:1] if f.value and normalize_tr(f.value) not in q_low]

        # Round-robin across axes so the result set spans dimensions, not just years.
        axes = [years, topics, authors]
        ordered: list[str] = []
        idx = 0
        while any(idx < len(axis) for axis in axes):
            for axis in axes:
                if idx < len(axis):
                    ordered.append(axis[idx])
                if len(ordered) >= count:
                    break
            if len(ordered) >= count:
                break
            idx += 1

        out: list[str] = []
        seen: set[str] = set()
        for value in ordered:
            suggestion = f"{q} {value}".strip() if q else str(value).strip()
            key = normalize_tr(suggestion)
            if not suggestion or key == q_low or key in seen:
                continue
            seen.add(key)
            out.append(suggestion)
            if len(out) >= count:
                break
        return out

    def auto_constraints(self, facets: FacetSet) -> tuple[dict, str]:
        """Non-interactive narrowing: apply the single strongest facet + a note.

        DEPRECATED — no longer called from the orchestrator path. Broad/ambiguous
        queries now surface ``rabbit_holes`` suggestions instead of being narrowed
        by a hard year filter. Kept for backward compatibility / tests.

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
