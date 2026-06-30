"""CitationBuilder — produces a stable citation list from assembled chunks."""
from __future__ import annotations

from src.agent.schemas import Chunk


class CitationBuilder:
    """Maps assembled chunks to citation dicts in stable order."""

    @staticmethod
    def build(chunks: list[Chunk]) -> list[dict]:
        return [
            {
                "index": i + 1,
                "collection_name": c.collection_name,
                "document_id": c.document_id,
                "chunk_id": c.chunk_id,
                "source_title": c.source_title,
                "doc_type": c.doc_type,
                "document_type": c.metadata.get("document_type") or c.doc_type,
                "source_name": c.metadata.get("source_name") or c.metadata.get("publication") or c.collection_name,
                "date": c.metadata.get("date") or c.metadata.get("document_date") or str(c.metadata.get("year", "")),
                "title": c.metadata.get("source_title") or c.metadata.get("title") or c.source_title,
                "author": c.metadata.get("author") or c.metadata.get("speaker") or "Belirtilmemiş",
                "metadata": dict(c.metadata),
            }
            for i, c in enumerate(chunks)
        ]
