"""Press clip adapter — inline text from JSON metadata.

No PDF parsing needed. Text comes from metadata.dokuman_metni.
Uses split_with_offsets for chunking with late chunking support.

Each press clip is a standalone article. Chunk metadata includes
publication info for display during retrieval.
"""
from __future__ import annotations

import hashlib
from typing import Optional

from src.common.chunking import split_with_offsets
from src.config import settings
from src.trainer.ingestion.adapters.base import DocumentAdapter, DocumentInput


class PressClipAdapter(DocumentAdapter):
    """Press clip from inline text in metadata.

    document_source is IGNORED — content comes from metadata.dokuman_metni.
    Chunklama **token-tabanlıdır**: boyut doc.max_chunk_tokens (token), tokenizer
    doc.tokenizer_name ile ölçülür.
    """

    def __init__(self, splitter=None):
        # splitter yalnızca testlerde enjekte edilir; üretimde parse() içinde
        # doc.tokenizer_name + doc.max_chunk_tokens ile token-tabanlı kurulur.
        self.splitter = splitter

    def parse(self, doc: DocumentInput) -> tuple[str, list[dict]]:
        text = (doc.metadata or {}).get("dokuman_metni", "")
        if not text:
            raise ValueError(
                f"press_clip adapter requires metadata.dokuman_metni: {doc.document_id}"
            )

        # Build prefix header (same format as existing pipeline)
        gazete = doc.source_name or "Bilinmiyor"
        baslik = doc.title or ""
        yazar = doc.author or "Bilinmiyor"
        tarih = doc.document_date or "?"
        konular = doc.topics or ""

        prefix = (
            f"Gazete: {gazete} | Tarih: {tarih} | "
            f"Yazar: {yazar} | Başlık: {baslik}\n"
        )

        # Late chunking: compute spans relative to full_text = prefix + text
        full_text = prefix + text
        offset = len(prefix)

        # Token-tabanlı bölme — char ofsetleri late chunking için korunur.
        chunk_size = doc.max_chunk_tokens or 512
        chunk_overlap = settings.PRESS_CHUNK_OVERLAP_TOKENS

        parts_with_offsets = split_with_offsets(
            text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            tokenizer_name=doc.tokenizer_name,
        )

        chunks = []
        for i, (part, (raw_start, raw_end)) in enumerate(parts_with_offsets):
            span = (offset + raw_start, offset + raw_end)
            chunks.append({
                "text": prefix + part,
                "span": span,
                "metadata": {
                    "document_id": doc.document_id,
                    "document_type": doc.document_type,
                    "source_name": gazete,
                    "date": tarih,
                    "year": doc.year,
                    "author": yazar,
                    "author_role": doc.author_role,
                    "source_title": baslik,
                    "topics": konular,
                    "chunk_index": i,
                },
            })

        return full_text, chunks

    def compute_content_hash(self, doc: DocumentInput) -> str:
        text = (doc.metadata or {}).get("dokuman_metni", "")
        return hashlib.sha256(text.encode()).hexdigest()[:32]
