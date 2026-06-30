"""Planner — intent-aware search-plan generation for the OrchestratorAgent.

Owns plan generation only: turns a user query (plus optional tool/db selection
from the IntentAnalyzer and constraints from the clarification stage) into a
SearchPlan with diversified query drafts. It does NOT run retrieval, answering,
sanitizer, or any retry loops — those live in the orchestrator and its stages.
"""
from __future__ import annotations

import json
import logging

from src.agent.schemas import (
    CollectionSearchPlan,
    SearchPlan,
    SearchQueryDraft,
)
from src.agent.tracer import PipelineTracer
from src.common.filter_translators import mask_filters
from src.common.llm_client_pool import LLMClientPool
from src.common.llm_utils import extract_json_from_text
from src.common.schemas import FilterCriteria
from src.config.collections import COLLECTIONS
from src.config.pipeline_loader import PipelineConfig

logger = logging.getLogger(__name__)


PLAN_SYSTEM_PROMPT = """Sen bir RAG araştırma planlama uzmanısın. Kullanıcı sorgusunu analiz et ve
arama planı oluştur.

Mevcut koleksiyonlar:
{catalog}

Kurallar:
1. Önce sorgunun amacını belirle: factual (basit bilgi), comparative (karşılaştırma),
   analytical (derin analiz), temporal (zaman bazlı), unknown
2. Hangi koleksiyonların ilgili olduğunu belirle. Doc-type yönlendirme:
   - Gazete/basın/köşe yazısı/manşet/muhabir/gazeteci soruları → doc_type=gazete koleksiyonları
   - Meclis/oturum/birleşim/milletvekili/konuşma/tutanak soruları → doc_type=tutanak koleksiyonları
   - Kanun teklifi/önerge/yasa taslağı soruları → doc_type=onerge koleksiyonları
   - Konu hangi türü ima ediyorsa o doc_type'tan en az bir koleksiyon seç; birden fazla tür
     ilgiliyse her birinden bir koleksiyon kullan.
3. Her koleksiyon için alternatif arama sorguları üret (farklı kelime seçimleri)
4. Arama stratejisini seç: parallel (hızlı) veya sequential (önceki sonuçlar
   sonraki aramayı etkilesin)
5. Kısa bir gerekçe yaz

JSON çıktısı:
{{
  "intent": "factual|comparative|analytical|temporal|unknown",
  "resources": [
    {{
      "collection": "koleksiyon_adi",
      "mode": "parallel|sequential",
      "priority": 1,
      "query_drafts": [
        {{"text": "arama_sorgusu", "top_k": 10}}
      ]
    }}
  ],
  "reasoning": "neden bu plan"
}}
"""

RE_RETRIEVAL_PROMPT = """Önceki arama yetersiz sonuç döndürdü ({result_count} sonuç).
Filtreleri gevşeterek yeni arama sorguları üret.

Mevcut koleksiyonlar:
{catalog}

Orijinal sorgu: {query}
Önceki plan: {previous_plan}

Daha geniş tarih aralığı, yazar filtresi kaldır, alternatif kelimeler kullan.
top_k değerini artır.

Doc-type yönlendirme (önceki plan yanlış doc_type seçmiş olabilir):
- Gazete/basın/köşe yazısı/muhabir → doc_type=gazete koleksiyonları
- Meclis/oturum/birleşim/milletvekili → doc_type=tutanak
- Kanun teklifi/önerge → doc_type=onerge
İlgili görünen başka doc_type varsa, ona ait koleksiyon ekleyerek aramayı genişlet.

JSON çıktısı (aynı format):
{{
  "intent": "...",
  "resources": [...],
  "reasoning": "..."
}}
"""


class Planner:
    """Generates and refines SearchPlans for the orchestrator.

    Public API:
      * ``plan()``  — base plan from the query + intent selection + clarification constraints.
      * ``broaden()`` — a broader plan for bounded re-query expansion.
    """

    # Tolerant coercion of LLM output: qwen occasionally emits an invalid token in
    # these fields, which would otherwise raise a Pydantic ValidationError and crash
    # the whole plan into the fallback path.
    _VALID_INTENTS = {"factual", "comparative", "analytical", "temporal", "unknown"}
    _VALID_QUERY_TYPES = {"fact", "summary", "comparison", "reasoning", "policy"}

    def __init__(
        self,
        config: PipelineConfig,
        client_pool: LLMClientPool,
        filter_extractor=None,
    ) -> None:
        self._config = config
        self._pool = client_pool
        self._filter_extractor = filter_extractor
        self._last_planner_error: str | None = None

    # ------------------------------------------------------------------ public

    def plan(
        self,
        query: str,
        tracer: "PipelineTracer | None" = None,
        *,
        selected_collections: list[str] | None = None,
        constraints: dict | None = None,
        max_variants: int | None = None,
        depth: int | None = None,
    ) -> SearchPlan:
        """Build an executable SearchPlan.

        Args:
            selected_collections: tool/db selection from the IntentAnalyzer. When
                given, the plan is restricted to these (catalog filtered upfront +
                resource intersection). Empty/None = planner selects freely.
            constraints: clarification narrowing — ``{"year": int, "topic": str,
                "collections": list[str]}`` — applied after FilterExtractor.
            max_variants: cap on total query drafts (breadth). Defaults to
                ``planner.normal_max_query_variants``.
            depth: minimum per-draft top_k (depth). Optional.
        """
        tracer = tracer or PipelineTracer()
        allowed = set(selected_collections) if selected_collections else None

        plan = self._generate_plan(query, tracer, allowed_keys=allowed)
        if plan is None:
            plan = self._fallback_plan(query, allowed_keys=allowed)
        if allowed:
            plan = self._restrict_to(plan, allowed, query)

        plan = self._apply_filter_extractor(query, plan, tracer)
        if constraints:
            self._apply_constraints(plan, constraints)

        cap = max_variants if max_variants is not None else self._config.planner.normal_max_query_variants
        self._cap_variants(plan, cap)
        if depth:
            self._apply_depth(plan, depth)
        return plan

    def broaden(
        self,
        query: str,
        previous_plan: SearchPlan,
        tracer: "PipelineTracer | None" = None,
        *,
        selected_collections: list[str] | None = None,
        max_variants: int | None = None,
    ) -> SearchPlan | None:
        """Generate a broader plan for bounded re-query expansion.

        Returns None when the LLM fails (caller keeps the original results).
        """
        tracer = tracer or PipelineTracer()
        allowed = set(selected_collections) if selected_collections else None

        plan = self._generate_broader_plan(query, previous_plan, tracer, allowed_keys=allowed)
        if plan is None:
            return None
        if allowed:
            plan = self._restrict_to(plan, allowed, query)
        plan = self._apply_filter_extractor(query, plan, tracer)
        cap = max_variants if max_variants is not None else self._config.planner.normal_max_query_variants
        self._cap_variants(plan, cap)
        return plan

    # --------------------------------------------------------- plan generation

    def _parse_plan(self, plan_data: dict) -> SearchPlan:
        """Build a SearchPlan from a parsed JSON dict, tolerant of LLM schema slips.

        Out-of-enum ``intent``/``query_type`` are coerced to safe defaults; a
        resource missing/blank ``collection`` or with no usable drafts is skipped.
        ``filters`` emitted by the planner LLM are intentionally DROPPED here —
        FilterExtractor is the single source of truth (populated later).
        """
        intent = plan_data.get("intent", "unknown")
        if intent not in self._VALID_INTENTS:
            intent = "unknown"
        query_type = plan_data.get("query_type", "fact")
        if query_type not in self._VALID_QUERY_TYPES:
            query_type = "fact"

        resources: list[CollectionSearchPlan] = []
        for r in plan_data.get("resources", []) or []:
            if not isinstance(r, dict):
                continue
            collection = r.get("collection")
            if not isinstance(collection, str) or not collection.strip():
                continue
            drafts: list[SearchQueryDraft] = []
            for d in r.get("query_drafts", []) or []:
                if not isinstance(d, dict):
                    continue
                text = d.get("text")
                if not isinstance(text, str) or not text.strip():
                    continue
                top_k = d.get("top_k", 10)
                drafts.append(SearchQueryDraft(
                    text=text.strip(),
                    filters=None,  # FilterExtractor doldurur; LLM filtreleri yok sayılır
                    top_k=top_k if isinstance(top_k, int) and top_k > 0 else 10,
                ))
            if not drafts:
                continue
            mode = r.get("mode")
            priority = r.get("priority", 1)
            resources.append(CollectionSearchPlan(
                collection=collection.strip(),
                mode=mode if mode in ("parallel", "sequential") else "parallel",
                priority=priority if isinstance(priority, int) else 1,
                query_drafts=drafts,
            ))

        return SearchPlan(
            intent=intent,
            query_type=query_type,
            resources=resources,
            reasoning=plan_data.get("reasoning", ""),
        )

    def _call_planner_llm(self, user_msg: str, system_prompt: str) -> SearchPlan | None:
        """Call the planner LLM and parse the response into a SearchPlan.

        Returns None on any failure so callers can apply fallback logic.
        """
        planner_cfg = self._config.planner
        block_name = planner_cfg.block
        model_key = planner_cfg.model_key
        client = self._pool.get_client(block_name)
        model = self._pool.get_model_for_block(block_name, model_key)
        try:
            think_val = planner_cfg.think if planner_cfg.think is not None else False
            res = client.chat(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_msg},
                ],
                options={"temperature": 0.0, "num_predict": self._config.get_block(block_name).max_num_predict},
                format="json",
                think=think_val,
            )
            # qwen3.5:9b wraps JSON in markdown fences even with format="json",
            # so strip them before parsing instead of feeding json.loads raw text.
            return self._parse_plan(json.loads(extract_json_from_text(res.message.content)))
        except Exception as e:
            self._last_planner_error = f"{type(e).__name__}: {e}"
            logger.warning("Planner LLM call failed: %s", self._last_planner_error)
            return None

    def _generate_plan(
        self,
        query: str,
        tracer: PipelineTracer,
        allowed_keys: set[str] | None = None,
    ) -> SearchPlan | None:
        """Generate a search plan using the planning agent LLM.

        When ``allowed_keys`` is given, the catalog shown to the planner is
        restricted to those collections so it can only route within the selection.
        The orchestrator owns the "planning" trace phase; this method does the
        LLM work and returns None on failure (the caller applies fallback logic).
        """
        catalog = self._config.get_collection_catalog(allowed_keys=allowed_keys)
        system_prompt = PLAN_SYSTEM_PROMPT.format(catalog=catalog)
        self._last_planner_error = None
        return self._call_planner_llm(f"Sorgu: {query}", system_prompt)

    def _apply_filter_extractor(
        self,
        query: str,
        plan: SearchPlan,
        tracer: PipelineTracer,
    ) -> SearchPlan:
        """Populate every query_draft.filters from FilterExtractor (single source of truth).

        Extraction runs ONCE on the original query (filters are a property of user
        intent, not phrasing). The resulting FilterCriteria is applied to each
        collection MASKED to the fields that type indexes (avoids cross-type
        over-filtering). ``refined_query`` is carried onto the plan for retrieval.
        No-op when filter_extractor is not injected (offline tests).
        """
        if self._filter_extractor is None:
            return plan

        with tracer.phase(
            "filter_extraction",
            model=getattr(self._filter_extractor, "model", None),
            details={"query": query[:100]},
        ) as ctx:
            result = self._filter_extractor.extract(query)
            criteria = result.filters
            applied = criteria.model_dump(exclude_none=True) if criteria else {}
            plan.refined_query = result.refined_query or None
            per_collection_applied: dict[str, dict] = {}
            for resource in plan.resources:
                spec = COLLECTIONS.get(resource.collection)
                if not applied:
                    masked = None
                elif spec is not None:
                    masked = mask_filters(criteria, spec.doc_type)
                else:
                    masked = criteria  # bilinmeyen koleksiyon: maskeleme yok
                masked_applied = masked.model_dump(exclude_none=True) if masked else {}
                per_collection_applied[resource.collection] = masked_applied
                for draft in resource.query_drafts:
                    draft.filters = masked.model_copy() if masked_applied else None
            if ctx:
                ctx.update_details(
                    filters=per_collection_applied,
                    refined_query=result.refined_query,
                )
        return plan

    def _fallback_plan(self, query: str, allowed_keys: set[str] | None = None) -> SearchPlan:
        """Generate a fallback plan when the planner LLM fails.

        When ``allowed_keys`` is given, search exactly those collections instead of
        the configured fallback set, so the selection is honored on the fallback path.
        """
        fb = self._config.planner
        if allowed_keys:
            collections = list(allowed_keys)
        else:
            collections = fb.fallback_collections or ["tutanaklar_ctx1024"]

        drafts = []
        for fq in fb.fallback_queries:
            text = fq.get("text", "{original_query}").format(original_query=query)
            drafts.append(SearchQueryDraft(
                text=text,
                filters=fq.get("filters"),
                top_k=fq.get("top_k", 10),
            ))
        if not drafts:
            drafts = [SearchQueryDraft(text=query, filters=None, top_k=10)]

        resources = [
            CollectionSearchPlan(
                collection=c,
                mode="parallel",
                priority=i + 1,
                query_drafts=drafts,
            )
            for i, c in enumerate(collections)
        ]

        return SearchPlan(
            intent="unknown",
            resources=resources,
            reasoning="Fallback plan (planner LLM failed)",
        )

    def _generate_broader_plan(
        self,
        query: str,
        previous_plan: SearchPlan,
        tracer: PipelineTracer,
        allowed_keys: set[str] | None = None,
    ) -> SearchPlan | None:
        """Generate a broader plan for re-query expansion.

        When ``allowed_keys`` is given, the catalog is restricted to the selection
        so re-query broadens the QUERY, not the collection set.
        """
        catalog = self._config.get_collection_catalog(allowed_keys=allowed_keys)
        system_prompt = RE_RETRIEVAL_PROMPT.format(
            catalog=catalog,
            query=query,
            previous_plan=previous_plan.model_dump_json(indent=2),
            result_count=0,
        )
        return self._call_planner_llm(f"Sorgu: {query}", system_prompt)

    # --------------------------------------------------------------- shaping

    def _restrict_to(self, plan: SearchPlan, allowed: set[str], query: str) -> SearchPlan:
        """Intersect the plan's collections with ``allowed`` (intent selection).

        If nothing survives (planner misrouted entirely), rebuild a fallback plan
        scoped to the selection so the chosen collections ARE searched.
        """
        kept = [r for r in plan.resources if r.collection in allowed]
        if kept:
            plan.resources = kept
            return plan
        return self._fallback_plan(query, allowed_keys=allowed)

    def _apply_constraints(self, plan: SearchPlan, constraints: dict) -> None:
        """Apply clarification narrowing (year / topic / collections) in place."""
        year = constraints.get("year")
        topic = constraints.get("topic")
        cols = constraints.get("collections")

        if cols:
            kept = [r for r in plan.resources if r.collection in set(cols)]
            if kept:
                plan.resources = kept

        for resource in plan.resources:
            for draft in resource.query_drafts:
                if year is not None:
                    f = draft.filters.model_copy() if draft.filters else FilterCriteria()
                    f.year = int(year)
                    draft.filters = f
                if topic and topic.lower() not in draft.text.lower():
                    draft.text = f"{draft.text} {topic}".strip()

    def _cap_variants(self, plan: SearchPlan, cap: int) -> None:
        """Trim total query drafts down to ``cap`` (breadth), never below 1/resource."""
        if cap <= 0:
            return
        total = sum(len(r.query_drafts) for r in plan.resources)
        if total <= cap:
            return
        ordered = sorted(plan.resources, key=lambda r: r.priority)
        while total > cap:
            trimmed = False
            for r in reversed(ordered):
                if len(r.query_drafts) > 1:
                    r.query_drafts.pop()
                    total -= 1
                    trimmed = True
                    if total <= cap:
                        break
            if not trimmed:
                break  # every resource at 1 draft; don't drop whole collections

    def _apply_depth(self, plan: SearchPlan, depth: int) -> None:
        """Raise each draft's top_k to at least ``depth`` (retrieval depth)."""
        for resource in plan.resources:
            for draft in resource.query_drafts:
                draft.top_k = max(draft.top_k, int(depth))
