"""Turkish-aware text utilities."""
from __future__ import annotations

import string
import time

from src.config import settings


def normalize_tr(text: str) -> str:
    """Lowercase with Turkish-specific İ/I handling and stripped punctuation."""
    trans = str.maketrans("", "", string.punctuation)
    return text.replace("İ", "i").replace("I", "ı").lower().translate(trans)


# Short-TTL cache of human-approved term_candidates (src.api.db) — the dynamic,
# self-populating counterpart to the static PARLIAMENTARY_TERM_SYNONYMS dict.
# A 60s staleness window means an approval in the "Öğrenilen Terimler" web UI
# panel takes effect in live search without a code change or process restart,
# while avoiding a DB round-trip on every single retrieval call.
_APPROVED_CACHE_TTL_SECONDS = 60
_approved_cache: dict[str, list[str]] = {}
_approved_cache_at: float = 0.0


def _get_approved_synonyms() -> dict[str, list[str]]:
    global _approved_cache, _approved_cache_at
    now = time.monotonic()
    if now - _approved_cache_at > _APPROVED_CACHE_TTL_SECONDS:
        try:
            from src.api import db
            _approved_cache = db.list_approved_term_synonyms()
        except Exception:
            pass  # DB unavailable (e.g. non-web callers) — keep serving the last-known cache
        _approved_cache_at = now
    return _approved_cache


def expand_parliamentary_synonyms(query_text: str) -> str:
    """Prepend official-term synonyms for any known parliamentary jargon word.

    Deterministic vocabulary-gap fix: colloquial terms like 'kadük' rarely
    match the corpus's own phrasing ("hükümsüz sayılan kanun teklifleri") in
    embedding space — see settings.PARLIAMENTARY_TERM_SYNONYMS for the
    empirical evidence. Synonyms are PREPENDED, not appended: empirically this
    clearly outperforms appending for this embedding model (the leading terms
    carry more weight in the pooled query vector). A no-op when no known term
    is present.

    Merges two sources: the static PARLIAMENTARY_TERM_SYNONYMS dict (curated,
    zero-DB-access, always the fastest path) and human-approved discoveries
    from the term_candidates review table (dynamic, short-TTL cached) — see
    src.agent.orchestrator._record_term_hypothesis for how candidates get there.
    """
    merged: dict[str, list[str]] = {k: list(v) for k, v in settings.PARLIAMENTARY_TERM_SYNONYMS.items()}
    for term, synonyms in _get_approved_synonyms().items():
        existing = merged.setdefault(term, [])
        existing.extend(s for s in synonyms if s not in existing)

    lower = normalize_tr(query_text)
    extras: list[str] = []
    for term, synonyms in merged.items():
        if term in lower:
            extras.extend(s for s in synonyms if normalize_tr(s) not in lower)
    if not extras:
        return query_text
    return f"{' '.join(extras)} {query_text}"


def extract_relevant_windows(
    text: str,
    query: str,
    window_size: int = settings.WINDOW_SIZE,
    max_total: int = settings.WINDOW_MAX_TOTAL,
) -> str:
    """Return a concatenation of context windows around query-term matches.

    Windows that overlap are merged. If no query term is found, the leading
    ``max_total`` characters are returned so the caller still gets context.
    """
    if not text:
        return ""
    query_words = [w for w in normalize_tr(query).split() if len(w) > 2]
    text_lower = normalize_tr(text)
    positions: list[int] = []
    for qw in query_words:
        start = 0
        while True:
            idx = text_lower.find(qw, start)
            if idx == -1:
                break
            positions.append(idx)
            start = idx + 1
    if not positions:
        return text[:max_total]
    positions.sort()
    windows: list[tuple[int, int]] = []
    for pos in positions:
        win_start = max(0, pos - window_size)
        win_end = min(len(text), pos + window_size)
        if windows and win_start <= windows[-1][1]:
            windows[-1] = (windows[-1][0], max(windows[-1][1], win_end))
        else:
            windows.append((win_start, win_end))
    parts: list[str] = []
    total_len = 0
    for ws, we in windows:
        part = text[ws:we]
        if total_len + len(part) > max_total:
            remaining = max_total - total_len
            if remaining > 200:
                parts.append(part[:remaining])
            break
        parts.append(part)
        total_len += len(part)
    return "\n[...]\n".join(parts)
