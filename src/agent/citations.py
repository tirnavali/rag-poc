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
                "metadata": dict(c.metadata),
            }
            for i, c in enumerate(chunks)
        ]
