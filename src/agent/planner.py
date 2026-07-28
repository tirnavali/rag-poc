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
    ReflectionOutput,
    SearchPlan,
    SearchQueryDraft,
    TermHypothesis,
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

Mevcut araştırma stratejileri (uygun olanın adını "strategy" alanına yaz;
hiçbiri uymuyorsa null bırak):
{strategy_catalog}

Kurallar:
1. Önce sorgunun amacını belirle: factual (basit bilgi), comparative (karşılaştırma),
   analytical (derin analiz), temporal (zaman bazlı), unknown
1b. Sorgu tipini (query_type) belirle: fact (tekil bilgi), summary (özet),
   comparison (karşılaştırma), reasoning (analiz), policy (mevzuat/karar),
   comprehensive (kapsamlı/sayım: "tüm", "bütün", "hepsi", "listele", "kaç tane",
   "hangileri", "her ..." gibi çok sayıda kayıt gerektiren toplu sorgular).
1c. Yukarıdaki strateji listesinden sorguya en uygun olanı seç (tetikleyici
   kelimeler ipucudur, zorunlu değildir); seçilen stratejinin query_type'ı ile
   1b'de belirlediğin query_type tutarlı olmalı.
1d. KANUN stratejileri (kanun_kabul_oylama / kanun_gorusmeleri / kanun_rapor_bolumu)
   YALNIZCA sorgu belirli bir KANUN / TEKLİF / YASA / madde / sıra sayısı / komisyon
   raporu bağlamına atıfta bulunuyorsa seçilebilir. Sorgu genel bir OLAY, KİŞİ, gündem
   ya da haber konusu hakkındaysa (bir kanunun Genel Kurul görüşmesi / oylaması / raporu
   DEĞİLSE) bu stratejileri SEÇME — uygun genel stratejiyi (analytical / summarize /
   comparative / enumerate / factual) ya da null kullan. "ne konuşuldu", "ne dedi",
   "eleştiri", "kim ne dedi" gibi ifadeler TEK BAŞINA kanun stratejisi TETİKLEMEZ;
   somut bir kanun bağlamı şarttır.
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
  "query_type": "fact|summary|comparison|reasoning|policy|comprehensive",
  "strategy": "<strateji_adi>|null",
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
{missing_aspects_block}Filtreleri gevşeterek yeni arama sorguları üret.

Mevcut koleksiyonlar:
{catalog}

Orijinal sorgu: {query}
Önceki plan: {previous_plan}

Daha geniş tarih aralığı, yazar filtresi kaldır, alternatif kelimeler kullan.
top_k değerini artır.
{tried_queries_block}

Doc-type yönlendirme (önceki plan yanlış doc_type seçmiş olabilir):
- Gazete/basın/köşe yazısı/muhabir → doc_type=gazete koleksiyonları
- Meclis/oturum/birleşim/milletvekili → doc_type=tutanak
- Kanun teklifi/önerge → doc_type=onerge
İlgili görünen başka doc_type varsa, ona ait koleksiyon ekleyerek aramayı genişlet.

TERİM HİPOTEZİ (opsiyonel ama önemli): Önceki tur "low_relevance_all_chunks" (bulunan
sonuçların hepsi alakasız) nedeniyle başarısız olduysa, sorgudaki bir kelime arşivin resmi/
hukuki terminolojisinden FARKLI, konuşma diline ait bir terim olabilir (örn. "kadük" arşivde
"hükümsüz sayılan" olarak geçebilir). Böyle bir terim seziyorsan, kendi bilgine dayanarak en
olası RESMİ/FORMEL karşılığını tahmin et; bunu hem yeni bir query_draft'a hem de aşağıdaki
"term_hypothesis" alanına yaz. Emin değilsen alanı null bırak — yanlış bir tahmin zararsızdır
(sadece kullanılmayan bir draft daha olur), o yüzden çekinme, en olası tahminini paylaş.
{rejected_hypotheses_block}
JSON çıktısı (aynı format):
{{
  "intent": "...",
  "resources": [...],
  "reasoning": "...",
  "term_hypothesis": {{"term": "...", "official_phrase": "..."}} veya null
}}
"""


REFLECT_PROMPT = """Sen çok-adımlı (multi-hop) bir TBMM arşiv araştırmasını yürüten bir ajan-planlayıcısın.
Aşağıdaki PROSEDÜRÜ hop hop uygula. Her turda: (1) şimdiye dek toplanan KANITTAN varlıkları
çıkar, (2) prosedürün DURMA koşulu sağlandıysa dur, (3) sağlanmadıysa BİR SONRAKİ hop için
arama sorguları üret.

PROSEDÜR (hop reçetesi):
{procedure}

NİHAİ HEDEF (cevabın alacağı biçim / answer_directive):
{answer_directive}

ÇIPA (önce çözülecek kimlik): {anchor}
HEDEF (aranan nihai kaydın biçimi): {target}
{aliases_block}ŞİMDİYE DEK ÇIKARILAN VARLIKLAR (extracted_anchors): {extracted_anchors}
HOP İMLECİ (kaçıncı hoptayız): {hop_cursor}

TOPLANAN KANIT (kompakt özet — chunk'ların tamamı değil):
{evidence_block}

Mevcut koleksiyonlar:
{catalog}
{tried_queries_block}
KURALLAR (TBMM domain — kritik):
- Esas no biçimi {{tür}}/{{no}} (1=tasarı, 2=teklif); metinde "Kanun Teklifi (2/773)" /
  "Kanun Tasarısı (1/N)" ya da yalnızca "(2/773)" biçiminde parantez içinde de geçebilir.
  Esas no'yu / sıra sayısını KANITIN METNİNDEN oku; ham id'yi ("2/773") sorgu olarak ARAMA
  (embedding kesin id'de zayıftır).
- Bir kanunun sıra sayısını / esas no'sunu KANITTAN okuduğunda MUTLAKA "extracted_anchors"a
  yaz (ör. {{"sira_sayisi": 5, "esas_no": "2/773"}}). Sistem bunu KESİN metadata filtresine
  çevirir — sıra_sayısı/esas_no + bölüm (section_type) omurgası ARTIK VAR; kanun adı hiç
  geçmese bile o kanunun roll-call/oylama/görüşme bölgesini tam getirir. Kendi "filters"
  alanına YAZMA (o yok sayılır) — yalnız extracted_anchors kullanılır.
- Güvence için numarayı SORGU METNİNE de göm ("5 sıra sayılı … açık oylama sonucu"); ama asıl
  daraltma extracted_anchors ile olur.
- Bir kaydı (tablo/oylama/gerekçe) kullanmadan önce SORULAN kanuna ait olduğunu adı ve/veya
  esas no'su ile DOĞRULA. Aynı oturumda birden çok kanun işlenir — başka kanunun kaydını bu
  kanuna ATFETME.
- Oy sorusu ise İKİ OLASILIK: (a) açık oylama → Kabul/Ret/Çekimser SAYILARI aranır; (b) işaretle
  (el kaldırarak) → SAYI YOKTUR. Sayı yoksa var sanıp boşuna genişletme.

DURMA: Prosedürün DURMA koşulu sağlandıysa "done": true ver ve "resources"ı boş bırak. Aksi
halde "done": false ver ve SADECE bir sonraki hop için "resources" üret (önceki sorguları
tekrar etme, belirgin FARKLI ve numarayı içeren sorgular kur).

JSON çıktısı:
{{
  "done": true|false,
  "done_reason": "..." veya null,
  "hop_cursor": <bir sonraki hop numarası, int>,
  "extracted_anchors": {{"esas_no": "...", "sira_sayisi": ..., "madde_no": ..., "section_type": "oylama|kanun_gorusmeleri|kanun_raporu|yazili_soru|... (belge bölümü, biliyorsan)", "granularite": "..."}},
  "resources": [
    {{"collection": "<koleksiyon adı>", "query_drafts": [{{"text": "<sonraki hop sorgusu>", "top_k": 10}}]}}
  ],
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
    _VALID_QUERY_TYPES = {"fact", "summary", "comparison", "reasoning", "policy", "comprehensive"}

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
        chat_history: list | None = None,
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
            chat_history: prior turns (``[{"role", "content"}, ...]``), most
                recent last. Used ONLY as a hint so query_drafts resolve
                anaphora ("konuyla ilgili", "bu konuda") into concrete terms —
                it does not change collection routing or filters.
        """
        tracer = tracer or PipelineTracer()
        allowed = set(selected_collections) if selected_collections else None

        plan = self._generate_plan(query, tracer, allowed_keys=allowed, chat_history=chat_history)
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
        result_count: int = 0,
        missing_aspects: list[str] | None = None,
        rejected_hypotheses: list[dict] | None = None,
        tried_queries: list[str] | None = None,
    ) -> SearchPlan | None:
        """Generate a broader plan for bounded re-query expansion.

        Args:
            result_count: actual chunk count from the round being broadened —
                the LLM used to always be told "0 sonuç" regardless of the real
                previous outcome; this carries the truth through.
            missing_aspects: the judge's reason codes for insufficiency (e.g.
                'low_relevance_all_chunks') — lets the prompt reason about WHY,
                not just broaden blindly.
            rejected_hypotheses: prior human-rejected {term, hypothesis} guesses
                (from the term_candidates review table) — fed back as a negative
                constraint so the LLM doesn't re-propose a debunked guess.
            tried_queries: query texts already searched in earlier rounds — fed
                back as a negative constraint; at temperature 0 the LLM would
                otherwise regenerate the same drafts every round (the orchestrator
                also hard-prunes exact repeats via its tried-search ledger, so a
                repeated draft is doubly wasted).

        Returns None when the LLM fails (caller keeps the original results).
        """
        tracer = tracer or PipelineTracer()
        allowed = set(selected_collections) if selected_collections else None

        plan = self._generate_broader_plan(
            query, previous_plan, tracer, allowed_keys=allowed,
            result_count=result_count, missing_aspects=missing_aspects,
            rejected_hypotheses=rejected_hypotheses, tried_queries=tried_queries,
        )
        if plan is None:
            return None
        if allowed:
            plan = self._restrict_to(plan, allowed, query)
        plan = self._apply_filter_extractor(query, plan, tracer)
        cap = max_variants if max_variants is not None else self._config.planner.normal_max_query_variants
        self._cap_variants(plan, cap)
        return plan

    def reflect(
        self,
        query: str,
        previous_plan: SearchPlan,
        tracer: "PipelineTracer | None" = None,
        *,
        procedure: str,
        answer_directive: str = "",
        anchor: str | None = None,
        target: str | None = None,
        aliases: list[dict] | None = None,
        extracted_anchors: dict | None = None,
        hop_cursor: int = 0,
        evidence_summary: str = "",
        tried_queries: list[str] | None = None,
    ) -> ReflectionOutput | None:
        """One round of adaptive multi-hop reflection / re-planning.

        Feeds the reflect LLM the strategy ``procedure`` recipe, the ``answer_directive``
        (final-answer shape), the entities resolved so far (``extracted_anchors``), the
        hop cursor, a compact evidence summary, and the queries already tried, then asks
        for the NEXT hop's search plan plus whether the procedure is DONE.

        Deliberately leaner than ``broaden()``: no FilterExtractor pass and no variant cap
        (the reflect LLM emits one focused hop). The reflect LLM's own ``filters`` field is
        dropped by ``_parse_plan``; instead it writes the law identity into
        ``extracted_anchors`` (sira_sayisi/esas_no), which the orchestrator turns into the
        EXACT metadata where-filter next hop (``_anchor_where_for`` — the sira_sayisi +
        section_type backbone). Numbers also ride in the query TEXT as a belt-and-suspenders.

        Returns None when the LLM fails, so the caller can fail-open to ``broaden()``.
        """
        tracer = tracer or PipelineTracer()
        system_prompt = self._build_reflect_prompt(
            query,
            previous_plan,
            procedure=procedure,
            answer_directive=answer_directive,
            anchor=anchor,
            target=target,
            aliases=aliases,
            extracted_anchors=extracted_anchors,
            hop_cursor=hop_cursor,
            evidence_summary=evidence_summary,
            tried_queries=tried_queries,
        )
        return self._call_reflect_llm(f"Sorgu: {query}", system_prompt)

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
        strategy = plan_data.get("strategy")
        if not isinstance(strategy, str) or not strategy.strip():
            strategy = None
        else:
            strategy = strategy.strip()

        term_hypothesis = None
        th = plan_data.get("term_hypothesis")
        if isinstance(th, dict):
            term = str(th.get("term") or "").strip()
            official_phrase = str(th.get("official_phrase") or "").strip()
            if term and official_phrase:
                term_hypothesis = TermHypothesis(term=term, official_phrase=official_phrase)

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
            strategy=strategy,
            resources=resources,
            reasoning=plan_data.get("reasoning", ""),
            term_hypothesis=term_hypothesis,
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
        chat_history: list | None = None,
    ) -> SearchPlan | None:
        """Generate a search plan using the planning agent LLM.

        When ``allowed_keys`` is given, the catalog shown to the planner is
        restricted to those collections so it can only route within the selection.
        The orchestrator owns the "planning" trace phase; this method does the
        LLM work and returns None on failure (the caller applies fallback logic).
        """
        catalog = self._config.get_collection_catalog(allowed_keys=allowed_keys)
        strategy_catalog = self._config.get_strategy_catalog() or "(tanımlı strateji yok — bu alanı null bırak)"
        system_prompt = PLAN_SYSTEM_PROMPT.format(catalog=catalog, strategy_catalog=strategy_catalog)
        self._last_planner_error = None
        history_hint = self._format_history_hint(chat_history)
        return self._call_planner_llm(f"{history_hint}Sorgu: {query}", system_prompt)

    @staticmethod
    def _format_history_hint(chat_history: list | None) -> str:
        """Compact "last turn" block so query_drafts pick up a carried-over topic.

        Only the last user+assistant turn is used (older turns are noise for
        drafting search terms) and the assistant side is hard-capped — this is
        a topic hint for the planner LLM, not a full transcript. Empty history
        → empty string (no-op, matches today's behavior exactly).
        """
        if not chat_history:
            return ""
        last_user = next((m for m in reversed(chat_history) if m.get("role") == "user" and m.get("content")), None)
        last_assistant = next((m for m in reversed(chat_history) if m.get("role") == "assistant" and m.get("content")), None)
        if not last_user and not last_assistant:
            return ""
        lines = [
            "Önceki konuşma (yalnız BAĞLAM için — sorgudaki \"konuyla ilgili\", \"bu konuda\", "
            "\"peki ya\" gibi atıfları bu geçmişe göre somutlaştır ve arama sorgularına konu "
            "adını/anahtar terimlerini ekle; bu geçmiş koleksiyon seçimini DEĞİŞTİRMEZ):",
        ]
        if last_user:
            lines.append(f"Önceki kullanıcı sorusu: {last_user['content'][:300]}")
        if last_assistant:
            lines.append(f"Önceki yanıt (özet): {last_assistant['content'][:400]}")
        lines.append("")
        return "\n".join(lines) + "\n"

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
        result_count: int = 0,
        missing_aspects: list[str] | None = None,
        rejected_hypotheses: list[dict] | None = None,
        tried_queries: list[str] | None = None,
    ) -> SearchPlan | None:
        """Generate a broader plan for re-query expansion.

        When ``allowed_keys`` is given, the catalog is restricted to the selection
        so re-query broadens the QUERY, not the collection set.
        """
        catalog = self._config.get_collection_catalog(allowed_keys=allowed_keys)
        missing_aspects_block = ""
        if missing_aspects:
            missing_aspects_block = f"Yetersizlik nedeni: {', '.join(missing_aspects)}.\n"
        rejected_hypotheses_block = ""
        if rejected_hypotheses:
            pairs = "; ".join(
                f"'{h['term']}' ≠ '{h['hypothesis']}'" for h in rejected_hypotheses
            )
            rejected_hypotheses_block = (
                f"\nDAHA ÖNCE DENENMİŞ VE YANLIŞ OLDUĞU DOĞRULANMIŞ KARŞILIKLAR "
                f"(BUNLARI TEKRAR ÖNERME): {pairs}\n"
            )
        tried_queries_block = ""
        if tried_queries:
            lines = "\n".join(f"- {q}" for q in tried_queries)
            tried_queries_block = (
                "\nDAHA ÖNCE ARANMIŞ SORGULAR (bunları ve çok benzer varyasyonlarını "
                "TEKRAR ÜRETME — aynı arama aynı sonucu döndürür; belirgin FARKLI "
                f"kelimeler, eş anlamlılar ve yeni açılar dene):\n{lines}\n"
            )
        system_prompt = RE_RETRIEVAL_PROMPT.format(
            catalog=catalog,
            query=query,
            previous_plan=previous_plan.model_dump_json(indent=2),
            result_count=result_count,
            missing_aspects_block=missing_aspects_block,
            rejected_hypotheses_block=rejected_hypotheses_block,
            tried_queries_block=tried_queries_block,
        )
        return self._call_planner_llm(f"Sorgu: {query}", system_prompt)

    # ------------------------------------------------------------- reflection

    def _build_reflect_prompt(
        self,
        query: str,
        previous_plan: SearchPlan,
        *,
        procedure: str,
        answer_directive: str,
        anchor: str | None,
        target: str | None,
        aliases: list[dict] | None,
        extracted_anchors: dict | None,
        hop_cursor: int,
        evidence_summary: str,
        tried_queries: list[str] | None,
    ) -> str:
        """Assemble the REFLECT_PROMPT for one adaptive hop."""
        allowed = {r.collection for r in previous_plan.resources} if previous_plan and previous_plan.resources else None
        catalog = self._config.get_collection_catalog(allowed_keys=allowed)

        aliases_block = ""
        if aliases:
            pairs = "\n".join(
                f"- '{a['term']}' → '{a['official_phrase']}'"
                for a in aliases
                if isinstance(a, dict) and a.get("term") and a.get("official_phrase")
            )
            if pairs:
                aliases_block = (
                    "ALIAS KÖPRÜLERİ (halk dili → arşivin resmi ifadesi; sorgularda RESMİ "
                    f"ifadeyi kullan):\n{pairs}\n"
                )

        tried_queries_block = ""
        if tried_queries:
            lines = "\n".join(f"- {q}" for q in tried_queries)
            tried_queries_block = (
                "\nDAHA ÖNCE ARANMIŞ SORGULAR (bunları ve çok benzer varyasyonlarını TEKRAR "
                f"ÜRETME — aynı arama aynı sonucu döndürür):\n{lines}\n"
            )

        return REFLECT_PROMPT.format(
            procedure=procedure or "(prosedür tanımsız)",
            answer_directive=answer_directive or "(yok)",
            anchor=anchor or "(belirtilmemiş)",
            target=target or "(belirtilmemiş)",
            aliases_block=aliases_block,
            extracted_anchors=json.dumps(extracted_anchors or {}, ensure_ascii=False),
            hop_cursor=hop_cursor,
            evidence_block=evidence_summary or "(henüz kanıt yok)",
            catalog=catalog,
            tried_queries_block=tried_queries_block,
        )

    def _call_reflect_llm(self, user_msg: str, system_prompt: str) -> ReflectionOutput | None:
        """Call the reflect LLM (planner block) and parse into a ReflectionOutput.

        Returns None on any failure so the caller can fail-open to broaden().
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
            return self._parse_reflection(json.loads(extract_json_from_text(res.message.content)))
        except Exception as e:
            self._last_planner_error = f"{type(e).__name__}: {e}"
            logger.warning("Reflect LLM call failed: %s", self._last_planner_error)
            return None

    def _parse_reflection(self, data: dict) -> ReflectionOutput:
        """Build a ReflectionOutput from a parsed JSON dict, tolerant of schema slips.

        The ``resources`` portion is parsed by the existing ``_parse_plan`` into a
        ``next_plan`` (so drafts/filters get the same coercion + filter-dropping as any
        plan); ``next_plan`` is None when no usable resources were emitted (done / stop).
        ``query_type`` on the next_plan defaults to "fact" here — the orchestrator carries
        the in-flight query_type forward before retrieval (mirrors the broaden path).
        """
        done = bool(data.get("done", False))
        done_reason = data.get("done_reason")
        if done_reason is not None and not isinstance(done_reason, str):
            done_reason = None
        hop_cursor = data.get("hop_cursor", 0)
        if not isinstance(hop_cursor, int):
            try:
                hop_cursor = int(hop_cursor)
            except (TypeError, ValueError):
                hop_cursor = 0
        anchors = data.get("extracted_anchors")
        if not isinstance(anchors, dict):
            anchors = {}

        next_plan: SearchPlan | None = self._parse_plan(data)
        if not next_plan.resources:
            next_plan = None

        return ReflectionOutput(
            done=done,
            done_reason=done_reason,
            hop_cursor=hop_cursor,
            extracted_anchors=anchors,
            next_plan=next_plan,
            reasoning=str(data.get("reasoning") or ""),
        )

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
