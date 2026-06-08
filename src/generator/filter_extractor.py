"""Türkçe doğal dil sorgularından metadata filtrelerini çıkaran ve sorguyu sadeleştiren motor.

Bu modül, kullanıcı sorgularını analiz ederek meclis tutanakları, gazete küpürleri,
kanun teklifleri vb. için metadata filtreleri üretir.
"""
from __future__ import annotations

import re
from typing import Optional

import ollama
from src.config import settings
from src.common.schemas import ExtractedFilterResponse, FilterCriteria
from src.common.llm_utils import parse_llm_response
from src.common.filter_translators import ChromaFilterTranslator
from src.generator.prompts import FILTER_SYSTEM_PROMPT


class FilterExtractor:
    """Doğal dil sorgularından metadata filtrelerini çıkaran ve temizleyen motor sınıfı."""

    def __init__(self, model: str = settings.FILTER_LLM_MODEL) -> None:
        self.model = model
        self.client = ollama.Client(host=settings.OLLAMA_HOST)

    def has_filter_hints(self, query: str) -> bool:
        """Sorguda olası filtre ipuçları olup olmadığını kontrol eder.

        Eğer sorguda yıl, meclis terimleri veya kaynak isimleri gibi
        herhangi bir filtre ipucu yoksa LLM çağrısını atlamak için kullanılır.
        """
        # 1. Yıl ipuçları (örn. 1996, 2023)
        if re.search(r"\b(19\d{2}|20\d{2})\b", query):
            return True

        # 2. TBMM, Gazete ve Belge anahtar kelimeleri
        keywords = {
            "dönem", "birleşim", "yasama", "tutanak", "önerge", "teklif",
            "gazete", "haber", "köşe", "kose", "makale", "yazar", "muhabir", "press",
            "hürriyet", "hurriyet", "milliyet", "sabah", "cumhuriyet", "tbmm", "meclis"
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

        return False

    def extract(self, query: str) -> ExtractedFilterResponse:
        """Kullanıcı sorgusundan metadata filtrelerini ve sadeleştirilmiş sorguyu çıkarır.

        Eğer sorguda filtre ipucu yoksa LLM çağrısı yapılmadan doğrudan boş filtre döner.
        """
        if not self.has_filter_hints(query):
            return ExtractedFilterResponse(refined_query=query, filters=FilterCriteria())

        try:
            res = self.client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": FILTER_SYSTEM_PROMPT},
                    {"role": "user", "content": f"Sorgu: \"{query}\""}
                ],
                options={"temperature": 0.0},
                format="json",
            )
            raw_response = res.message.content.strip()
            return parse_llm_response(raw_response, ExtractedFilterResponse)
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
