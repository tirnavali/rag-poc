"""LLM-based author backstop for chunks where regex extraction failed.

Runs ONLY on chunks with metadata["author"] is None.
Uses Ollama LLM with per-type prompt from settings.AUTHOR_VALIDATOR_PROMPTS.
Adds `author_source` to chunk metadata: "regex" | "llm" | "inherited" | None.
"""
from __future__ import annotations

import json
import re
from typing import Optional

from src.config import settings


_JSON_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


def validate_unknown_authors(
    chunks: list[dict],
    document_type: str,
    ollama_client=None,
    model: Optional[str] = None,
    prev_context_chars: Optional[int] = None,
) -> list[dict]:
    """Fill chunk metadata.author for chunks missing it, using local LLM.

    Args:
        chunks: List of {"text": ..., "metadata": {...}} chunks.
        document_type: DocumentInput.document_type → prompt selector.
        ollama_client: ollama.Client; lazily created if None.
        model: LLM model; defaults to settings.LLM_MODEL.
        prev_context_chars: Context window from previous chunk's tail.

    Returns:
        Same chunks list, mutated in-place with author_source set.
    """
    if not settings.AUTHOR_VALIDATOR_ENABLED:
        return chunks

    prompt_template = settings.AUTHOR_VALIDATOR_PROMPTS.get(document_type)
    if not prompt_template:
        return chunks

    if ollama_client is None:
        import ollama
        ollama_client = ollama.Client(host=settings.OLLAMA_HOST)
    model = model or settings.LLM_MODEL
    prev_chars = prev_context_chars or settings.AUTHOR_VALIDATOR_PREV_CHARS

    for i, chunk in enumerate(chunks):
        meta = chunk.setdefault("metadata", {})
        if meta.get("author"):
            meta.setdefault("author_source", "regex")
            continue

        prev = chunks[i - 1]["text"][-prev_chars:] if i > 0 else ""
        prompt = prompt_template.format(prev=prev, chunk=chunk["text"][:2000])

        try:
            resp = ollama_client.generate(
                model=model,
                prompt=prompt,
                options={"temperature": 0.0, "num_predict": 200},
                stream=False,
            )
            raw = resp.get("response", "")
            parsed = _parse_json_response(raw)
            if parsed and parsed.get("author") and parsed["author"].upper() != "BİLİNMİYOR":
                meta["author"] = parsed["author"]
                meta["author_role"] = parsed.get("author_role")
                meta["author_source"] = "llm"
                meta["author_confidence"] = parsed.get("confidence")
            else:
                meta["author_source"] = "llm_unknown"
        except Exception as e:
            meta["author_source"] = "llm_failed"
            meta["author_error"] = str(e)

    return chunks


def _parse_json_response(raw: str) -> Optional[dict]:
    m = _JSON_RE.search(raw)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
