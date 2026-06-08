"""Pure-Python pre-classifier bad-words filter.

Runs before any LLM call. Fail-closed on match (returns matched=True);
fail-open on configuration / regex compile errors (returns matched=False
after logging a warning). No external dependencies beyond the standard
library.
"""
from __future__ import annotations

import logging
import re
from typing import Protocol

from src.agent.schemas import BadWordsResult

logger = logging.getLogger(__name__)

_TURKISH_FOLD = str.maketrans({
    "ı": "i", "İ": "i", "I": "i",
    "ş": "s", "Ş": "s",
    "ç": "c", "Ç": "c",
    "ğ": "g", "Ğ": "g",
    "ü": "u", "Ü": "u",
    "ö": "o", "Ö": "o",
})

_TOKEN_RE = re.compile(r"\b[\wçğıöşüÇĞİÖŞÜ]+\b", flags=re.UNICODE)


def _fold(text: str) -> str:
    """Lowercase + Turkish-accent fold."""
    return text.translate(_TURKISH_FOLD).lower()


class _BadWordsConfigLike(Protocol):
    bad_words_enabled: bool
    bad_words: list[str]
    bad_word_patterns: list[str]
    bad_words_response_message: str


class BadWordsFilter:
    """Word-boundary + accent-folded match against a YAML-curated list.

    Tokens are matched whole (no substring matches) to avoid false positives
    on legitimate words that share a substring with a bad term. Multi-word
    patterns use IGNORECASE regex against the accent-folded query.
    """

    def __init__(self, config: _BadWordsConfigLike) -> None:
        self._enabled = bool(config.bad_words_enabled)
        # Pre-fold the word set once; subsequent checks are O(tokens).
        self._words: set[str] = {_fold(w) for w in config.bad_words if w}
        # Pre-compile patterns against folded text; warn + skip bad ones.
        self._patterns: list[re.Pattern[str]] = []
        for raw in config.bad_word_patterns:
            if not raw:
                continue
            try:
                self._patterns.append(re.compile(_fold(raw), flags=re.IGNORECASE | re.UNICODE))
            except re.error as e:
                logger.warning("BadWordsFilter: skipping invalid pattern %r (%s)", raw, e)

    def check(self, query: str) -> BadWordsResult:
        if not self._enabled or not query:
            return BadWordsResult(matched=False)

        folded = _fold(query)
        matched: list[str] = []

        # Token-level match
        for token in _TOKEN_RE.findall(folded):
            if token in self._words:
                matched.append(token)

        # Multi-word patterns
        for pat in self._patterns:
            m = pat.search(folded)
            if m:
                matched.append(m.group(0))

        if matched:
            # Preserve first-seen order while dropping duplicates
            seen: set[str] = set()
            deduped: list[str] = []
            for term in matched:
                if term not in seen:
                    seen.add(term)
                    deduped.append(term)
            return BadWordsResult(matched=True, matched_terms=deduped)
        return BadWordsResult(matched=False)
