"""Planning Agent — orchestrates retrieval, answering, and validation.

Receives a user query, generates a search plan, executes searches across
collections, routes results to the answering agent, and validates output.
"""
from __future__ import annotations

import json
import logging
import re as _re
from concurrent.futures import ThreadPoolExecutor

from src.agent.bad_words_filter import BadWordsFilter
from src.agent.classifier import ScopeClassifier
from src.agent.sanitizer import SanitizerAgent
from src.agent.schemas import (
    AgentOutput,
    CollectionSearchPlan,
    SearchPlan,
    SearchQueryDraft,
    ValidationResult,
)
from src.agent.suggester import Suggester
from src.agent.tools import AnswerTool, ContextBuilderTool, SearchTool
from src.agent.tracer import PipelineTracer
from src.common.filter_translators import ChromaFilterTranslator
from src.common.llm_client_pool import LLMClientPool
from src.common.llm_utils import extract_json_from_text
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
4. Filtreleri çıkar (yıl, yazar, kaynak, dönem, birleşim)
5. Arama stratejisini seç: parallel (hızlı) veya sequential (önceki sonuçlar
   sonraki aramayı etkilesin)
6. Kısa bir gerekçe yaz

JSON çıktısı:
{{
  "intent": "factual|comparative|analytical|temporal|unknown",
  "resources": [
    {{
      "collection": "koleksiyon_adi",
      "mode": "parallel|sequential",
      "priority": 1,
      "query_drafts": [
        {{"text": "arama_sorgusu", "filters": {{"year": 1997, "author": null}}, "top_k": 5}}
      ]
    }}
  ],
  "reasoning": "neden bu plan"
}}

Filtre alanları: year (int), year_lte (int), year_gte (int), author (string),
author_role (string), source_name (string), period (int), session (int).
Kullanılmayan filtreler null olmalı.
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

NOTHING_FOUND_PATTERNS = _re.compile(
    r"bulunamadı|bilgi\s+yok|kaynaklarda\s+yer\s+almıyor|tespit\s+edilemedi|"
    r"bilgiye\s+ulaşılamadı|mevcut\s+değil|yer\s+almamaktadır|"
    r"bilgi\s+bulunmamaktadır|yanıt\s+veremiyorum",
    _re.IGNORECASE,
)

GAP_FILL_PROMPT = """Önceki arama soruyu yanıtlayacak bilgiyi bulamadı.
Kullanıcının sorusunda aradığı spesifik bilgiyi bulmak için yeni hedefli arama sorguları üret.

Mevcut koleksiyonlar:
{catalog}

Orijinal soru: {query}
Yetersiz yanıt: {answer}
Sorunlar: {issues}

Özellikle şunlara odaklan:
- Soruda geçen spesifik kavramları farklı kelimelerle ifade et
- Eş anlamlı terimler, alternatif yazımlar dene
- Daha geniş veya daha dar kapsam alternatifleri ekle
- top_k değerini artır (en az 8)

JSON çıktısı (aynı format):
{{
  "intent": "...",
  "resources": [...],
  "reasoning": "..."
}}
"""


class PlanningAgent:
    """Planning Agent that orchestrates the full RAG pipeline."""

    def __init__(
        self,
        config: PipelineConfig,
        client_pool: LLMClientPool,
    ) -> None:
        self._config = config
        self._pool = client_pool
        self._last_planner_error: str | None = None
        self._search_tool = SearchTool(config, client_pool)
        self._context_tool = ContextBuilderTool(config)
        self._answer_tool = AnswerTool(client_pool, config)
        self._sanitizer = SanitizerAgent(client_pool, config)
        self._bad_words = (
            BadWordsFilter(config.bad_words_filter)
            if getattr(config, "bad_words_filter", None) and config.bad_words_filter.enabled
            else None
        )
        self._classifier = (
            ScopeClassifier(client_pool, config)
            if getattr(config, "classifier", None) and config.classifier.enabled
            else None
        )
        self._suggester = Suggester(client_pool, config)

    def run(
        self,
        query: str,
        *,
        trace: PipelineTracer | None = None,
        session_collections: list[str] | None = None,
    ) -> AgentOutput:
        """Execute the full agent pipeline.

        Args:
            query: user query
            trace: optional existing tracer (creates new one if None)
            session_collections: collections the user selected at session start.
                When given, the planner is restricted to these (catalog filtered
                upfront + plan resources intersected as a safety net), mirroring
                the OrchestratorAgent's PolicyEnforcer. None = no restriction.

        Returns:
            AgentOutput with answer, trace, plan, and validation.
        """
        tracer = trace or PipelineTracer()

        # Stage 1: bad-words filter (cheapest, no LLM, fail-closed on match)
        if self._bad_words is not None:
            bw = self._bad_words.check(query)
            with tracer.phase(
                "bad_words_filter",
                details={"matched": bw.matched, "matched_terms": bw.matched_terms},
            ):
                pass
            if bw.matched:
                return AgentOutput(
                    answer=self._config.bad_words_filter.response_message,
                    scope="bad_word",
                    suggestions=[],
                    plan=None,
                    validation=None,
                    sources=[],
                    trace=tracer.events,
                )

        # Stage 2: scope classifier (LLM, fail-open)
        if self._classifier is not None:
            scope_result = self._classifier.classify(query, tracer)
            if (
                scope_result.scope == "off_domain"
                and scope_result.confidence >= self._config.classifier.confidence_threshold
            ):
                suggestions = self._suggester.suggest(query, tracer)
                answer = self._format_off_domain_answer(suggestions)
                return AgentOutput(
                    answer=answer,
                    scope="off_domain",
                    suggestions=suggestions,
                    plan=None,
                    validation=None,
                    sources=[],
                    trace=tracer.events,
                )

        # Phase 1: Generate search plan
        allowed = set(session_collections) if session_collections else None
        plan = self._generate_plan(query, tracer, allowed_keys=allowed)
        if plan is None:
            plan = self._fallback_plan(query, allowed_keys=allowed)

        # Phase 1b: Enforce session collection selection (safety net)
        if allowed:
            plan = self._enforce_session_collections(query, plan, allowed, tracer)

        # Phase 2: Execute searches
        all_results = self._execute_plan(plan, tracer)

        # Phase 2b: Re-retrieval if needed — loop up to the configured retry count,
        # broadening the plan each pass until enough results are found.
        re_retrieved = False
        current_plan = plan
        attempts = 0
        while attempts < self._config.planner.re_retrieval_max_retries and self._needs_reretrieval(all_results):
            broader_plan = self._generate_broader_plan(query, current_plan, all_results, tracer)
            if broader_plan is None:
                break
            new_results = self._execute_plan(broader_plan, tracer, phase="re_retrieval")
            all_results = self._merge_results(all_results, new_results)
            current_plan = broader_plan
            re_retrieved = True
            attempts += 1

        # Phase 3: Build context and answer
        context, sources = self._context_tool.build(all_results)
        thinking, answer = self._call_answering(query, context, tracer)

        # Phase 4: Validate
        validation = self._validate_output(query, answer, sources, tracer, context)

        # If validation failed, try sanitization. Prefer the corrected_answer the
        # sanitizer already produced in the validation call (no extra LLM round-trip);
        # fall back to a dedicated sanitize() call only when none was returned.
        if validation and not validation.passes:
            sanitizer_cfg = self._config.sanitizer
            for attempt in range(sanitizer_cfg.max_retries):
                if validation.corrected_answer:
                    answer = validation.corrected_answer
                elif validation.retry_hint:
                    answer = self._sanitizer.sanitize(query, answer, context)
                else:
                    break
                validation = self._validate_output(query, answer, sources, tracer, context)
                if validation and validation.passes:
                    break

        # Phase 4b: Quality-based re-retrieval
        quality_re_retrieved = False
        if self._needs_quality_reretrieval(answer, validation):
            gap_plan = self._generate_gap_fill_plan(query, answer, validation, tracer)
            if gap_plan:
                gap_results = self._execute_plan(
                    gap_plan, tracer, phase="quality_reretrieval"
                )
                all_results = self._merge_results(all_results, gap_results)
                context, sources = self._context_tool.build(all_results)
                thinking, answer = self._call_answering(query, context, tracer)
                validation = self._validate_output(query, answer, sources, tracer, context)
                quality_re_retrieved = True

        # Collect source metadata for output (dedup by chunk_id)
        seen_chunk_ids: set[str] = set()
        source_metas = []
        for result in all_results:
            for meta in result.get("metadatas", []):
                cid = meta.get("chunk_id") or str(id(meta))
                if cid not in seen_chunk_ids:
                    seen_chunk_ids.add(cid)
                    source_metas.append(meta)

        return AgentOutput(
            answer=answer,
            thinking=thinking,
            plan=plan,
            validation=validation,
            trace=tracer.events,
            sources=source_metas,
            re_retrieved=re_retrieved,
            quality_re_retrieved=quality_re_retrieved,
        )

    def _format_off_domain_answer(self, suggestions: list[str]) -> str:
        """Render the off-domain response template with up to 3 suggestions."""
        template = (
            self._config.off_domain_response_template
            or "Bu sistem alan dışında.\n1. {suggestion_0}\n2. {suggestion_1}\n3. {suggestion_2}"
        )
        padded = (suggestions + ["", "", ""])[:3]
        return template.format(
            suggestion_0=padded[0],
            suggestion_1=padded[1],
            suggestion_2=padded[2],
        )

    def _parse_plan(self, plan_data: dict) -> SearchPlan:
        """Build a SearchPlan from a parsed JSON dict."""
        return SearchPlan(
            intent=plan_data.get("intent", "unknown"),
            resources=[
                CollectionSearchPlan(
                    collection=r["collection"],
                    mode=r.get("mode", "parallel"),
                    priority=r.get("priority", 1),
                    query_drafts=[
                        SearchQueryDraft(
                            text=d["text"],
                            filters=d.get("filters"),
                            top_k=d.get("top_k", 5),
                        )
                        for d in r.get("query_drafts", [])
                    ],
                )
                for r in plan_data.get("resources", [])
            ],
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
            # Record the real reason; the bare-except fallback used to hide it,
            # making every parse failure look like a generic "planner failed".
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
        restricted to those collections so it can only route within the user's
        session selection.
        """
        planner_cfg = self._config.planner
        block_name = planner_cfg.block
        model_key = planner_cfg.model_key
        model = self._pool.get_model_for_block(block_name, model_key)
        catalog = self._config.get_collection_catalog(allowed_keys=allowed_keys)

        system_prompt = PLAN_SYSTEM_PROMPT.format(catalog=catalog)

        with tracer.phase(
            "planning",
            block=block_name,
            model=model,
            details={"query": query[:100]},
        ) as phase_ctx:
            self._last_planner_error = None
            plan = self._call_planner_llm(f"Sorgu: {query}", system_prompt)
            if plan is None:
                phase_ctx.update_details(
                    error=self._last_planner_error or "planner LLM returned None"
                )
                return None

            query_drafts_summary = {
                r.collection: [d.text for d in r.query_drafts]
                for r in plan.resources
            }
            phase_ctx.update_details(
                intent=plan.intent,
                resources=", ".join(r.collection for r in plan.resources),
                query_drafts=query_drafts_summary,
            )
            return plan

    def _fallback_plan(self, query: str, allowed_keys: set[str] | None = None) -> SearchPlan:
        """Generate a fallback plan when the planner LLM fails.

        When ``allowed_keys`` is given, search exactly the user-selected
        collections (instead of the configured fallback set), so the session
        selection is honored even on the fallback path.
        """
        fb = self._config.planner
        if allowed_keys:
            collections = list(allowed_keys)
        else:
            collections = fb.fallback_collections or ["tbmm_tutanaklar_nomic_v2"]

        drafts = []
        for fq in fb.fallback_queries:
            text = fq.get("text", "{original_query}").format(original_query=query)
            drafts.append(SearchQueryDraft(
                text=text,
                filters=fq.get("filters"),
                top_k=fq.get("top_k", 10),
            ))

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

    def _enforce_session_collections(
        self,
        query: str,
        plan: SearchPlan,
        allowed: set[str],
        tracer: PipelineTracer,
    ) -> SearchPlan:
        """Intersect the plan's collections with the user's session selection.

        Mirrors the OrchestratorAgent's PolicyEnforcer for the legacy path: the
        catalog is already filtered upfront, but the planner LLM may still
        hallucinate an out-of-scope collection. Drop those here. If nothing
        survives (planner misrouted entirely), rebuild a fallback plan scoped to
        the selected collections so they ARE searched.
        """
        with tracer.phase("policy") as ctx:
            kept = [r for r in plan.resources if r.collection in allowed]
            dropped = [r.collection for r in plan.resources if r.collection not in allowed]
            if kept:
                plan.resources = kept
            else:
                plan = self._fallback_plan(query, allowed_keys=allowed)
                kept = plan.resources
            if ctx:
                ctx.update_details(
                    allowed=sorted(allowed),
                    kept=[r.collection for r in kept],
                    dropped=dropped,
                )
        return plan

    def _execute_plan(
        self,
        plan: SearchPlan,
        tracer: PipelineTracer,
        *,
        phase: str = "retrieval",
    ) -> list[dict]:
        """Execute the search plan and return results per collection.

        Resources run in priority order. Within a resource, ``mode="parallel"``
        runs its query drafts concurrently on a thread pool (search is I/O-bound:
        Chroma query + Ollama rerank); ``mode="sequential"`` runs them in order.
        """
        all_results: list[dict] = []
        sorted_resources = sorted(plan.resources, key=lambda r: r.priority)

        for resource in sorted_resources:
            if resource.mode == "parallel" and len(resource.query_drafts) > 1:
                with ThreadPoolExecutor(max_workers=len(resource.query_drafts)) as pool:
                    futures = [
                        pool.submit(self._execute_single, resource.collection, draft, tracer, phase)
                        for draft in resource.query_drafts
                    ]
                    all_results.extend(f.result() for f in futures)
            else:
                for draft in resource.query_drafts:
                    all_results.append(
                        self._execute_single(resource.collection, draft, tracer, phase)
                    )

        return all_results

    def _execute_single(
        self,
        collection: str,
        draft: SearchQueryDraft,
        tracer: PipelineTracer,
        phase: str = "retrieval",
    ) -> dict:
        """Execute a single search query draft."""
        where_filter = None
        if draft.filters:
            where_filter = ChromaFilterTranslator().translate(draft.filters)

        with tracer.phase(
            phase,
            details={
                "collection": collection,
                "query": draft.text[:80],
            },
        ) as phase_ctx:
            result = self._search_tool.search(
                collection_key=collection,
                query_text=draft.text,
                filters=where_filter,
                top_k=draft.top_k,
            )
            count = len(result.get("documents", []))
            phase_ctx.update_details(result_count=count)
            return result

    def _needs_reretrieval(self, all_results: list[dict]) -> bool:
        """Check if re-retrieval should be triggered."""
        rr = self._config.planner
        if not rr.re_retrieval_enabled:
            return False

        total = sum(len(r.get("documents", [])) for r in all_results)
        return total < rr.re_retrieval_min_results

    def _needs_quality_reretrieval(
        self,
        answer: str,
        validation: ValidationResult,
    ) -> bool:
        """Check if quality-based re-retrieval should be triggered."""
        if not self._config.planner.re_retrieval_on_quality_failure:
            return False
        if not validation.passes:
            return True
        return bool(NOTHING_FOUND_PATTERNS.search(answer))

    def _generate_broader_plan(
        self,
        query: str,
        previous_plan: SearchPlan,
        all_results: list[dict],
        tracer: PipelineTracer,
    ) -> SearchPlan | None:
        """Generate a broader plan for re-retrieval."""
        catalog = self._config.get_collection_catalog()
        system_prompt = RE_RETRIEVAL_PROMPT.format(
            catalog=catalog,
            query=query,
            previous_plan=previous_plan.model_dump_json(indent=2),
            result_count=sum(len(r.get("documents", [])) for r in all_results),
        )
        return self._call_planner_llm(f"Sorgu: {query}", system_prompt)

    def _generate_gap_fill_plan(
        self,
        query: str,
        answer: str,
        validation: ValidationResult,
        tracer: PipelineTracer,
    ) -> SearchPlan | None:
        """Generate a targeted plan to fill the information gap in a failing answer."""
        catalog = self._config.get_collection_catalog()
        issues_text = "; ".join(validation.issues) if validation.issues else "Yanıt soruyu karşılamıyor"
        system_prompt = GAP_FILL_PROMPT.format(
            catalog=catalog,
            query=query,
            answer=answer[:500],
            issues=issues_text,
        )
        return self._call_planner_llm(f"Sorgu: {query}", system_prompt)

    def _merge_results(
        self,
        original: list[dict],
        new: list[dict],
    ) -> list[dict]:
        """Merge re-retrieval results with original results, deduplicating."""
        seen_ids: set[str] = set()
        merged = []

        for result in original:
            merged.append(result)
            for meta in result.get("metadatas", []):
                cid = meta.get("chunk_id", "")
                if cid:
                    seen_ids.add(cid)

        for result in new:
            new_docs = []
            new_metas = []
            new_dists = []
            for doc, meta, dist in zip(
                result.get("documents", []),
                result.get("metadatas", []),
                result.get("distances", []),
            ):
                cid = meta.get("chunk_id", "")
                if cid not in seen_ids:
                    seen_ids.add(cid)
                    new_docs.append(doc)
                    new_metas.append(meta)
                    new_dists.append(dist)

            if new_docs:
                merged.append({
                    "documents": new_docs,
                    "metadatas": new_metas,
                    "distances": new_dists,
                })

        return merged

    def _call_answering(
        self,
        query: str,
        context: str,
        tracer: PipelineTracer,
    ) -> tuple[str, str]:
        """Call the answering agent LLM."""
        ans_cfg = self._config.answering
        block_name = ans_cfg.block
        model_key = ans_cfg.model_key
        model = self._pool.get_model_for_block(block_name, model_key)

        with tracer.phase(
            "answering",
            block=block_name,
            model=model,
            details={"context_chars": len(context)},
        ):
            return self._answer_tool.generate(query, context)

    def _validate_output(
        self,
        query: str,
        answer: str,
        sources: list[dict],
        tracer: PipelineTracer,
        context: str = "",
    ) -> ValidationResult:
        """Validate the output using the sanitizer agent."""
        sanitizer_cfg = self._config.sanitizer
        block_name = sanitizer_cfg.block
        model_key = sanitizer_cfg.model_key
        model = self._pool.get_model_for_block(block_name, model_key)

        with tracer.phase(
            "validation",
            block=block_name,
            model=model,
        ) as phase_ctx:
            validation = self._sanitizer.validate(query, answer, sources, context)
            phase_ctx.update_details(
                passes=validation.passes,
                checks=validation.checks,
                issues=validation.issues,
            )
            return validation


class Planner:
    """Thin facade exposing only plan generation for the OrchestratorAgent.

    Reuses PlanningAgent's prompt and fallback logic; does not run retrieval,
    answering, sanitizer, or any retry loops.
    """

    def __init__(self, config: PipelineConfig, client_pool: LLMClientPool) -> None:
        self._inner = PlanningAgent(config, client_pool)

    def plan(
        self,
        query: str,
        tracer: "PipelineTracer | None" = None,
    ) -> SearchPlan:
        tracer = tracer or PipelineTracer()
        plan = self._inner._generate_plan(query, tracer)
        if plan is None:
            plan = self._inner._fallback_plan(query)
        return plan
