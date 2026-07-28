"""CitationBuilder — produces a stable citation list from assembled chunks."""
from __future__ import annotations

import re

from src.agent.schemas import Chunk
from src.common.text import normalize_tr

# Matches any "(...)" group and reads its LAST TWO comma-separated segments as
# (tarih, yazar) — covers both prompt variants in generator/prompts.py:
# "(Kaynak: Gazete adı/TBMM Tutanak, Tarih, Yazar/Konuşmacı)" and the
# müfettiş-mode "(Gazete/Tutanak, Tarih, Yazar/Konuşmacı)" (no "Kaynak:" prefix).
_PAREN_RE = re.compile(r"\(([^()]{3,150})\)")


class CitationBuilder:
    """Maps assembled chunks to citation dicts in stable order."""

    @staticmethod
    def _cited_pairs(final_answer: str | None) -> set[tuple[str, str]]:
        """Best-effort (tarih, yazar) pairs parsed from inline "(Kaynak: ...)" markers.

        Used only to flag which retrieved chunks the prose actually cites — never
        to drop a chunk. A miss here (unusual LLM phrasing) just leaves that
        chunk's ``cited`` as False, not removed from the source list.
        """
        pairs: set[tuple[str, str]] = set()
        for group in _PAREN_RE.findall(final_answer or ""):
            parts = [p.strip() for p in group.split(",")]
            if len(parts) < 2:
                continue
            date_part, author_part = parts[-2], parts[-1]
            if date_part and author_part:
                pairs.add((normalize_tr(date_part), normalize_tr(author_part)))
        return pairs

    @staticmethod
    def build(chunks: list[Chunk], final_answer: str | None = None) -> list[dict]:
        """Build the citation list. When ``final_answer`` is given, each entry
        also gets ``cited: bool`` — whether its (date, author) appears in an
        inline citation marker in the prose (fail-open: unmatched → False, the
        chunk still stays in the list). Omitting ``final_answer`` preserves the
        old behavior exactly (no ``cited`` key)."""
        cited_pairs = CitationBuilder._cited_pairs(final_answer) if final_answer is not None else None
        out = []
        for i, c in enumerate(chunks):
            date = c.metadata.get("date") or c.metadata.get("document_date") or str(c.metadata.get("year", ""))
            author = c.metadata.get("author") or c.metadata.get("speaker") or "Belirtilmemiş"
            entry = {
                "index": i + 1,
                "collection_name": c.collection_name,
                "document_id": c.document_id,
                "chunk_id": c.chunk_id,
                "source_title": c.source_title,
                "doc_type": c.doc_type,
                "document_type": c.metadata.get("document_type") or c.doc_type,
                "source_name": c.metadata.get("source_name") or c.metadata.get("publication") or c.collection_name,
                "date": date,
                "title": c.metadata.get("source_title") or c.metadata.get("title") or c.source_title,
                "author": author,
                "text": c.text,
                "metadata": dict(c.metadata),
            }
            if cited_pairs is not None:
                entry["cited"] = (normalize_tr(str(date)), normalize_tr(str(author))) in cited_pairs
            out.append(entry)
        return out
