"""Sanitizer Agent — validates and fixes output quality."""
from __future__ import annotations

import json

from src.agent.schemas import ValidationResult
from src.common.llm_client_pool import LLMClientPool
from src.common.llm_utils import extract_json_from_text
from src.config.pipeline_loader import PipelineConfig


SANITIZER_PROMPT = """Sen bir RAG yanıt doğrulama ve düzeltme uzmanısın.

Görevin: Verilen yanıtı kontrol etmek ve gerekirse düzeltmek.

Kontrol kriterleri:
{criteria}

DEĞERLENDİRME KURALLARI (çok önemli — varsayılan tutum GEÇER yönündedir):
- "İddialar kaynaklarla destekleniyor mu?" kontrolünü SADECE BAĞLAM (alınan
  kaynak metinleri) bölümüne bakarak değerlendir. KAYNAK METAVERİSİ bölümü
  yalnızca atıf bilgisidir; eksik veya "?" olması yanıtın yanlış olduğu anlamına
  GELMEZ. Yanıttaki iddialar BAĞLAM metniyle örtüşüyorsa "backed_by_sources": true.
- "passes": false SADECE şu durumlarda olur: (a) yanıt Türkçe değil, (b) yanıt
  boş/anlamsız, (c) yanıt soruyu tamamen görmezden geliyor, ya da (d) yanıt
  BAĞLAM ile AÇIKÇA ÇELİŞEN bir iddia içeriyor.
- Şunlar başarısızlık nedeni DEĞİLDİR: eksik metaveri, "?" alanlar, yanıtın kısa
  olması, üslup, ya da BAĞLAM'da olup yanıtta yer almayan ek ayrıntılar. Yanıt
  doğru ve BAĞLAM'a dayanıyorsa "passes": true döndür.
- Emin değilsen "passes": true döndür.

JSON çıktısı:
{{
  "passes": true/false,
  "checks": {{
    "addresses_query": true/false,
    "backed_by_sources": true/false,
    "no_hallucination": true/false,
    "is_turkish": true/false
  }},
  "issues": ["sorun 1", "sorun 2"],
  "corrected_answer": "düzeltiysen burada, yoksa orijinal yanıt"
}}

Eğer yanıt tüm kriterleri karşılıyorsa "passes": true döndür ve
"corrected_answer" alanını orijinal yanıtla aynı bırak.

Eğer yanıt başarısız olursa (yukarıdaki a-d durumları), kaynaklara dayanarak
düzeltilmiş versiyonunu "corrected_answer" alanında döndür.
"""


class SanitizerAgent:
    """Validates and optionally corrects the answering agent's output."""

    def __init__(self, client_pool: LLMClientPool, config: PipelineConfig) -> None:
        self._pool = client_pool
        self._config = config

    def validate(
        self,
        query: str,
        answer: str,
        sources: list[dict],
        context: str = "",
    ) -> ValidationResult:
        """Validate the answer against configured criteria.

        Args:
            query: original user query
            answer: generated answer text
            sources: list of source metadata dicts

        Returns:
            ValidationResult with pass/fail status and individual checks.
        """
        sanitizer_cfg = self._config.sanitizer
        block_name = sanitizer_cfg.block
        model_key = sanitizer_cfg.model_key

        client = self._pool.get_client(block_name)
        model = self._pool.get_model_for_block(block_name, model_key)

        criteria_text = "\n".join(
            f"  - {i+1}. {c}" for i, c in enumerate(sanitizer_cfg.validation_criteria)
        )

        source_summary = self._format_source_summary(sources)

        prompt = SANITIZER_PROMPT.format(criteria=criteria_text)
        ctx_excerpt = (
            context[:5000] + "\n[...devamı kısaltıldı]"
            if len(context) > 5000
            else context
        )
        user_content = (
            f"SORU: {query}\n\n"
            f"YANIT:\n{answer}\n\n"
            f"BAĞLAM (alınan kaynaklar):\n{ctx_excerpt}\n\n"
            f"KAYNAK METAVERİSİ:\n{source_summary}"
        )

        try:
            think_val = sanitizer_cfg.think if sanitizer_cfg.think is not None else False
            res = client.chat(
                model=model,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": user_content},
                ],
                options={"temperature": sanitizer_cfg.temperature},
                format="json",
                think=think_val,
            )
            parsed = json.loads(extract_json_from_text(res.message.content))

            checks = parsed.get("checks", {})
            issues = parsed.get("issues", [])
            passes = parsed.get("passes", True)
            corrected = parsed.get("corrected_answer")
            # Only surface a correction when it actually differs from the input.
            if corrected is not None and corrected.strip() == answer.strip():
                corrected = None

            return ValidationResult(
                passes=passes,
                checks=checks,
                issues=issues,
                retry_hint=sanitizer_cfg.retry_prompt if not passes else None,
                corrected_answer=corrected if not passes else None,
            )
        except Exception as e:
            # Fail-open: a broken validator must not block the answer or trigger
            # pointless retries. Mark validation as "did not run" so the trace is
            # honest rather than reporting a clean PASS.
            return ValidationResult(
                passes=True,
                checks={"validation_ran": False},
                issues=[f"Validation skipped (sanitizer error): {e}"],
                retry_hint=None,
            )

    @staticmethod
    def _format_source_summary(sources: list[dict]) -> str:
        """Render up to 5 sources as citation lines, omitting absent fields.

        Earlier this emitted ``Kaynak i: ? | ? | ?`` whenever a chunk's metadata
        lacked ``source_name``/``date``/``author`` (common for tutanak/onerge),
        which nudged the validator into a false-negative ``backed_by_sources:
        false``. We now pull each field through a fallback chain and drop fields
        that are still missing — a sparse-but-real source reads as a real source,
        not a string of question marks.
        """
        lines: list[str] = []
        for i, src in enumerate(sources[:5], 1):
            src = src or {}
            pub = src.get("source_name") or src.get("source_title") or src.get("title")
            date = src.get("date")
            author = src.get("author")
            parts = [str(p) for p in (pub, date, author) if p]
            label = " | ".join(parts) if parts else "(metaveri yok, içerik BAĞLAM'da)"
            lines.append(f"  Kaynak {i}: {label}")
        return "\n".join(lines) + ("\n" if lines else "")

    def sanitize(
        self,
        query: str,
        answer: str,
        context: str,
    ) -> str:
        """Attempt to fix a failing answer.

        Args:
            query: original user query
            answer: current (failing) answer
            context: retrieved context text

        Returns:
            Corrected answer text.
        """
        sanitizer_cfg = self._config.sanitizer
        block_name = sanitizer_cfg.block
        model_key = sanitizer_cfg.model_key

        client = self._pool.get_client(block_name)
        model = self._pool.get_model_for_block(block_name, model_key)

        prompt = (
            f"Önceki yanıt yetersiz bulundu. {sanitizer_cfg.retry_prompt}\n\n"
            f"SORU: {query}\n\n"
            f"BAĞLAM:\n{context}\n\n"
            f"Önceki YANIT:\n{answer}\n\n"
            f"Düzeltilmiş yanıtı ver. Sadece yanıtı döndür, açıklama ekleme."
        )

        try:
            res = client.chat(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": sanitizer_cfg.temperature},
                think=sanitizer_cfg.think if sanitizer_cfg.think is not None else False,
            )
            corrected = res.message.content.strip()
            return corrected if corrected else answer
        except Exception:
            return answer
