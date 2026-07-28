"""Türkçe doğal dil sorgularından metadata filtrelerini çıkaran ve sorguyu sadeleştiren motor.

Bu modül, kullanıcı sorgularını analiz ederek meclis tutanakları, gazete küpürleri,
kanun teklifleri vb. için metadata filtreleri üretir.
"""
from __future__ import annotations

import re
from typing import Optional

import ollama
import openai
from src.config import settings
from src.common.schemas import ExtractedFilterResponse, FilterCriteria
from src.common.llm_utils import parse_llm_response
from src.common.filter_translators import ChromaFilterTranslator
from src.common.text import normalize_tr
from src.generator.prompts import FILTER_SYSTEM_PROMPT

# Check 4 (has_filter_hints): iki ardışık, kısa dur-kelime listesinde olmayan
# içerik kelimesi bir Türkçe "Ad Soyad" örüntüsü gibi okunur — büyük harfle
# başlamayan (günlük sohbette çok yaygın) özel isimleri yakalamak için
# case-insensitive tamamlayıcı sinyal. Kapsamlı bir Türkçe stopword listesi
# değil — yalnız bu sistemde sorgularda sık geçen jenerik kelimeleri eleyip
# yanlış-pozitifi (iki jenerik kelimeyi isim sanma) azaltan bir alt küme.
_BARE_NAME_STOPWORDS = {
    "bu", "şu", "o", "ben", "sen", "biz", "siz", "onlar", "bir", "iki", "üç",
    "ile", "için", "gibi", "ama", "veya", "ve", "de", "da", "ki", "mı", "mi",
    "mu", "mü", "mısın", "misin", "musun", "müsün", "ne", "nasıl", "neden",
    "niçin", "niye", "kim", "kimler", "nerede", "kaç", "hangi", "hangileri",
    "neler", "nelerdir", "var", "yok", "oldu", "olan", "olur", "olmuş",
    "yaptı", "yapmış", "yapan", "yapılan", "dedi", "demiş", "diyen", "etti",
    "eden", "konusunda", "konusu", "konuyla", "konuyu", "konu", "ilgili",
    "ilgisi", "hakkında", "başka", "başkaca", "açıklama", "açıklamaları",
    "açıklamalar", "açıklamasında", "söyledi", "söylemiş", "belirtti",
    "ifade", "tekrar", "yine", "daha", "çok", "en", "her", "tüm", "bütün",
    "sadece", "yalnız", "ilk", "son", "yani",
    # zaman/genel-jenerik kelimeler — bunlar tek başına özel isim sinyali
    # DEĞİL, ardışık geldiklerinde yanlış-pozitif üretmesinler diye eklendi
    # (ör. "geçen hafta", "bugün hava").
    "bugün", "dün", "yarın", "hafta", "gün", "ay", "yıl", "sabah", "akşam",
    "gece", "önce", "sonra", "geçen", "gelecek", "şimdi", "şu an", "hava",
    "durum", "durumu", "olay", "olayı", "olaylar", "tartışma", "tartışmalar",
    "gündem", "haber", "haberler",
    # sohbet dolguları — sorgu değil, nezaket/kapanış ifadeleri.
    "teşekkür", "teşekkürler", "ederim", "ediyorum", "rica", "selam",
    "selamlar", "lütfen", "sağol", "sağolun", "görüşürüz",
}


def _looks_like_bare_name(query: str) -> bool:
    """Case-insensitive stand-in for the capitalization check below.

    Catches lowercase-typed "Ad Soyad"/"Ad Göbek Soyad" style names (e.g.
    "engin özkoç") that the capitalization heuristic misses entirely — chat
    input is routinely all-lowercase, so relying only on capitalization
    silently drops the author-filter opportunity for exactly this common case.

    Deliberately strict (this gate is precision-critical — see
    TestFilterHintsKnownLimitations in tests/test_filter_extractor.py): a
    2-3 word span only counts as a name if EVERY OTHER word in the query is
    a recognized stopword. A single stray content word elsewhere ("kardak
    krizi tarihi", "baykal konuşmaları") kills the match. That makes this a
    false-negative-prone heuristic (it will miss real names in less-templated
    sentences) rather than a false-positive-prone one — the correct trade-off
    for a cost gate whose only job is deciding whether to spend one cheap LLM
    call, not to correctly extract the filter itself (the LLM still does that).
    """
    tokens = normalize_tr(query).split()
    n = len(tokens)
    if n < 3:
        return False
    for span_len in (2, 3):
        for i in range(n - span_len + 1):
            span = tokens[i:i + span_len]
            rest = tokens[:i] + tokens[i + span_len:]
            if not rest:
                continue
            if all(len(t) >= 3 and t not in _BARE_NAME_STOPWORDS for t in span) and all(
                t in _BARE_NAME_STOPWORDS for t in rest
            ):
                return True
    return False


class FilterExtractor:
    """Doğal dil sorgularından metadata filtrelerini çıkaran ve temizleyen motor sınıfı."""

    def __init__(self, model: str = settings.FILTER_LLM_MODEL) -> None:
        self.model = model
        self.api_base = settings.FILTER_LLM_API_BASE
        if self.api_base:
            self.client = openai.OpenAI(base_url=self.api_base, api_key=settings.FILTER_LLM_API_KEY)
        else:
            self.client = ollama.Client(host=settings.OLLAMA_HOST)

    def has_filter_hints(self, query: str) -> bool:
        """Sorguda olası filtre ipuçları olup olmadığını kontrol eder.

        Eğer sorguda yıl, meclis terimleri veya kaynak isimleri gibi
        herhangi bir filtre ipucu yoksa LLM çağrısını atlamak için kullanılır.
        """
        # 1. Yıl ipuçları (örn. 1996, 2023)
        if re.search(r"\b(19\d{2}|20\d{2})\b", query):
            return True

        # 2. TBMM, Gazete ve Belge anahtar kelimeleri + bölüm (section_type) niyeti.
        # Bölüm kelimeleri ("oylama"/"görüşme") yaygındır → LLM'i daha sık tetikler;
        # zararsız (uygunsuzsa section_type:null döner) ama maliyeti biraz artırır.
        keywords = {
            "dönem", "birleşim", "yasama", "tutanak", "önerge", "teklif",
            "gazete", "haber", "köşe", "kose", "makale", "yazar", "muhabir", "press",
            "hürriyet", "hurriyet", "milliyet", "sabah", "cumhuriyet", "tbmm", "meclis",
            # bölüm niyeti (tag_sections → section_type)
            "oylama", "görüşme", "görüşül", "müzakere", "gündem dışı", "yazılı soru",
            "sözlü soru", "genel görüşme", "araştırma", "yemin", "bölüm",
        }
        normalized = query.lower()
        for kw in keywords:
            if kw in normalized:
                return True

        # 3. Cümle başı dışındaki kelimelerin büyük harfle başlaması (Özel İsim / Yazar vb. ipucu)
        words = query.strip().split()
        if len(words) > 1:
            for w in words[1:]:
                # Kelimenin ilk harfi büyükse ve kelimenin tamamı büyük harf değilse
                if w and w[0].isupper() and not w.isupper():
                    return True

        # 4. Büyük harf kullanılmasa bile "Ad Soyad" örüntüsü (bkz. _looks_like_bare_name) —
        # check 3'ün büyük/küçük harfe bağımlılığı yüzünden kaçırdığı, günlük sohbette
        # yaygın küçük-harfli özel isimleri (ör. "engin özkoç") yakalar.
        if _looks_like_bare_name(query):
            return True

        return False

    def extract(self, query: str) -> ExtractedFilterResponse:
        """Kullanıcı sorgusundan metadata filtrelerini ve sadeleştirilmiş sorguyu çıkarır.

        Eğer sorguda filtre ipucu yoksa LLM çağrısı yapılmadan doğrudan boş filtre döner.
        """
        if not self.has_filter_hints(query):
            return ExtractedFilterResponse(refined_query=query, filters=FilterCriteria())

        try:
            messages = [
                {"role": "system", "content": FILTER_SYSTEM_PROMPT},
                {"role": "user", "content": f"Sorgu: \"{query}\""}
            ]
            if self.api_base:
                # Uzak OpenAI-uyumlu proxy (LiteLLM/vLLM) yolu. chat_template_kwargs
                # ile enable_thinking=False: Ollama'daki think=False'un eşdeğeri —
                # vLLM/Qwen3 reasoning izini kapatır (yoksa ~40x fazla token, çok
                # daha yavaş). num_ctx eşdeğeri yok: vLLM context'i sunucu tarafında
                # sabit (Ollama'daki reload/OOM riski burada geçerli değil).
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.0,
                    response_format={"type": "json_object"},
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                )
                raw_response = resp.choices[0].message.content.strip()
            else:
                res = self.client.chat(
                    model=self.model,
                    messages=messages,
                    # num_ctx PİNLİ: pinlenmezse ollama 256K default context'le yüklenmeye
                    # çalışıp KV OOM (500) verir → tüm filtre çıkarımı sessizce fail-open'a
                    # düşerdi. Filtre çıkarımı küçük iş → küçük context yeter (settings).
                    # think=False: filtre çıkarımı yapısal JSON, muhakeme gerektirmez —
                    # thinking'li modellerde (qwen3.5) reasoning izini kapatır → ~3x hızlı,
                    # daha temiz JSON. Thinking'siz modellerde (gemma) güvenle yok sayılır.
                    options={"temperature": 0.0, "num_ctx": settings.FILTER_LLM_NUM_CTX},
                    think=False,
                    format="json",
                )
                raw_response = res.message.content.strip()
            parsed = parse_llm_response(raw_response, ExtractedFilterResponse)
            # SORGU-KIRPMA YOK (deterministik): precision tamamen FİLTRELERDEN gelir;
            # refined_query'i orijinal sorguya sabitle. LLM'in içerik silmesi (kanun/kurum
            # adı gibi hiçbir filtreye dönüşmeyen ana içeriği atması) böylece fiziksel
            # olarak imkânsız; fallback filtreleri gevşetince de elde tam sorgu kalır.
            # LLM'den yalnız `filters` alınır; refined_query/removed_words yok sayılır.
            return ExtractedFilterResponse(
                refined_query=query,
                filters=parsed.filters,
                removed_words=[],
            )
        except Exception as e:
            print(f"[FilterExtractor] Filtre çıkarma hatası: {e}")
            return ExtractedFilterResponse(refined_query=query, filters=FilterCriteria())

    @staticmethod
    def to_chroma_filter(filters: FilterCriteria) -> Optional[dict]:
        """FilterCriteria nesnesini ChromaDB uyumlu where filtre sözlüğüne dönüştürür."""
        return ChromaFilterTranslator().translate(filters)

    @staticmethod
    def fallback_chain(criteria: FilterCriteria) -> list[tuple[Optional[str], Optional[dict]]]:
        """Ordered relaxation candidates for zero-result fallback.

        Returns (level_name, where_filter) tuples to try in order; caller stops at
        first non-empty result.
        Tiers: (None, full_filter) → ("author_dropped", relaxed) → ("semantic_only", None)
        """
        full = FilterExtractor.to_chroma_filter(criteria)
        chain = [(None, full)]

        relaxed_criteria = criteria.model_copy(
            update={"author": None, "author_role": None}
        )
        relaxed = FilterExtractor.to_chroma_filter(relaxed_criteria)
        if relaxed is not None and relaxed != full:
            chain.append(("author_dropped", relaxed))

        chain.append(("semantic_only", None))
        return chain
