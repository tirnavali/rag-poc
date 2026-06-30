"""Sanitizer Agent — validates and fixes output quality."""
from __future__ import annotations

import json

from src.agent.schemas import ValidationResult
from src.common.llm_client_pool import LLMClientPool
from src.common.llm_utils import extract_json_from_text
from src.config.pipeline_loader import PipelineConfig


SANITIZER_PROMPT = """Sen bir RAG yanıt doğrulama uzmanısın.

Görevin: Verilen yanıtı kontrol etmek ve KISA bir karar JSON'u döndürmek.
Yanıtı YENİDEN YAZMA veya kopyalama — sadece değerlendir.

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

SADECE şu kısa JSON'u döndür (yanıt metnini ASLA tekrar etme):
{{
  "passes": true/false,
  "checks": {{
    "addresses_query": true/false,
    "backed_by_sources": true/false,
    "no_hallucination": true/false,
    "is_turkish": true/false
  }},
  "issues": ["kısa sorun açıklaması", ...]
}}

"issues" yalnızca "passes": false olduğunda dolu olmalı; aksi halde boş liste.
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
            block = self._config.get_block(block_name)
            res = client.chat(
                model=model,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": user_content},
                ],
                # Validation output is a tiny decision JSON; cap num_predict so the
                # model can't spend seconds regenerating long text. (Previously it
                # echoed the entire answer into a discarded corrected_answer field,
                # which dominated latency — ~24s on long answers.)
                options={
                    "temperature": sanitizer_cfg.temperature,
                    "num_predict": min(256, block.max_num_predict),
                },
                format="json",
                think=think_val,
            )
            parsed = json.loads(extract_json_from_text(res.message.content))

            checks = parsed.get("checks", {})
            issues = parsed.get("issues", [])
            passes = parsed.get("passes", True)

            # Validation is advisory/non-destructive: the orchestrator never applies
            # a correction, so we don't ask the model to produce one.
            return ValidationResult(
                passes=passes,
                checks=checks,
                issues=issues,
                retry_hint=sanitizer_cfg.retry_prompt if not passes else None,
                corrected_answer=None,
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
