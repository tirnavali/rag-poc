"""Author name → indexed-label resolver for ChromaDB metadata filtering.

ChromaDB metadata filtering is exact-match only (`$eq`), but the indexed
`author` for parliament/onerge is the FULL titled label as it appears in the
record — e.g. "BAŞBAKAN RECEP TAYYİP ERDOĞAN". So an exact filter for a plain
person name ("Recep Tayyip Erdoğan") matches nothing.

This resolver reads the collection's actual author vocabulary (cached), then
returns every label whose tokens are a CASE-INSENSITIVE (Turkish-aware)
superset of the queried name's tokens. The caller turns those labels into an
`author $in [...]` filter — so "geçen kelime" eşleşen tüm konuşmacı etiketleri
doğrudan gelir.

resolve() return contract:
  - None  → vocabulary unavailable (collection missing / read error): caller
            should fall back to the default $eq/$or translation.
  - []    → vocabulary available but no label matches: caller should DROP the
            author filter and rely on semantic search (avoid zeroing out).
  - [..]  → matched labels for an `$in` filter.
"""
from __future__ import annotations

import re

# İ/I/Ş/Ğ/Ç/Ö/Ü → Turkish-correct lowercase before str.lower() (which would
# turn "İ" into "i̇" with a combining dot and mangle dotless-ı handling).
_TR_LOWER = str.maketrans({
    "İ": "i", "I": "ı", "Ş": "ş", "Ğ": "ğ", "Ç": "ç", "Ö": "ö", "Ü": "ü",
})


def _tokens(text: str) -> set[str]:
    """Normalized word tokens of a name/label (Turkish-aware, case-insensitive)."""
    if not text:
        return set()
    return set(re.findall(r"\w+", text.translate(_TR_LOWER).lower(), flags=re.UNICODE))


class AuthorResolver:
    """Resolves a person name to the collection's matching author labels."""

    def __init__(self) -> None:
        # collection_key → list[label] | None (None = vocab unavailable, cached)
        self._cache: dict[str, list[str] | None] = {}

    def _vocabulary(self, collection_key: str) -> list[str] | None:
        if collection_key in self._cache:
            return self._cache[collection_key]

        vocab: list[str] | None = None
        try:
            from src.common.chroma import open_collection
            from src.config.collections import COLLECTIONS
            from src.config.document_types import normalize_metadata

            spec = COLLECTIONS.get(collection_key)
            if spec is not None:
                _, col = open_collection(spec.db_path, spec.name)
                res = col.get(include=["metadatas"])
                seen: dict[str, None] = {}
                for meta in (res.get("metadatas") or []):
                    author = normalize_metadata(meta or {}).get("author")
                    if author:
                        seen[author] = None
                vocab = list(seen.keys())
        except Exception:
            vocab = None

        self._cache[collection_key] = vocab
        return vocab

    def resolve(self, collection_key: str, name: str) -> list[str] | None:
        """Labels in ``collection_key`` whose tokens cover every token of ``name``."""
        vocab = self._vocabulary(collection_key)
        if vocab is None:
            return None
        query_tokens = _tokens(name)
        if not query_tokens:
            return []
        return [label for label in vocab if query_tokens <= _tokens(label)]

    def clear_cache(self) -> None:
        """Drop cached vocabularies (e.g. after a reindex)."""
        self._cache.clear()


# Process-wide singleton (vocabulary cached for the process lifetime).
AUTHOR_RESOLVER = AuthorResolver()
