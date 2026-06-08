"""LLM-based cleaning of dirty AuthorTransition results at atom level.

Called when regex matched but confidence heuristics flag result as dirty
(e.g. compound title captured as name, OCR intra-word spaces).
Controlled by settings.AUTHOR_VALIDATOR_ENABLED — same flag as chunk-level validator.
"""
from __future__ import annotations

import json
import re
from typing import Optional

from src.config import settings
from src.common.parsing.author_extractor import AuthorTransition

_JSON_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)

# Turkish-aware uppercase mapping: only 'i' → 'İ' is special; others use standard case
_TR_UPPER = str.maketrans("abcçdefgğhıijklmnoöprsştuüvyz",
                           "ABCÇDEFGĞHIİJKLMNOÖPRSŞTUÜVYZ")


def _tr_upper(s: str) -> str:
    """Uppercase with Turkish-specific İ/i handling."""
    return s.translate(_TR_UPPER)


def clean_author_transition(
    transition: AuthorTransition,
    raw_text: str,
    document_type: str,
    ollama_client=None,
    model: Optional[str] = None,
) -> AuthorTransition:
    """Clean dirty AuthorTransition (e.g. OCR errors, title/name conflation).

    Args:
        transition: Detected AuthorTransition from regex (may be noisy).
        raw_text: Original atom/paragraph text.
        document_type: DocumentInput.document_type → prompt selector.
        ollama_client: ollama.Client; lazily created if None.
        model: LLM model; defaults to settings.AUTHOR_TRANSITION_CLEAN_MODEL.

    Returns:
        Corrected AuthorTransition, or original on any error/disabled.
    """
    if not settings.AUTHOR_VALIDATOR_ENABLED:
        return transition

    prompt_template = settings.AUTHOR_TRANSITION_CLEAN_PROMPTS.get(document_type)
    if not prompt_template:
        return transition

    if ollama_client is None:
        import ollama

        ollama_client = ollama.Client(host=settings.OLLAMA_HOST)
    model = model or settings.AUTHOR_TRANSITION_CLEAN_MODEL

    head = raw_text[:200]
    prompt = prompt_template.format(
        raw_head=head,
        detected_name=transition.author,
        detected_role=transition.author_role or "(none)",
    )

    try:
        resp = ollama_client.generate(
            model=model,
            prompt=prompt,
            options={"temperature": 0.0, "num_predict": 200},
            stream=False,
        )
        raw = resp.get("response", "")
        parsed = _parse_json_response(raw)
        if parsed and parsed.get("author"):
            author = parsed["author"].strip()
            if author.upper() != "BİLİNMİYOR":
                return AuthorTransition(
                    author=_tr_upper(author),
                    author_role=parsed.get("author_role") or transition.author_role,
                    extra=transition.extra,
                    confidence=parsed.get("confidence", 0.8),
                )
    except Exception:
        pass

    return transition


def _parse_json_response(raw: str) -> Optional[dict]:
    m = _JSON_RE.search(raw)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
