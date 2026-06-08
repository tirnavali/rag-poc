"""Turkish-aware text utilities."""
from __future__ import annotations

import string

from src.config import settings


def normalize_tr(text: str) -> str:
    """Lowercase with Turkish-specific İ/I handling and stripped punctuation."""
    trans = str.maketrans("", "", string.punctuation)
    return text.replace("İ", "i").replace("I", "ı").lower().translate(trans)


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
