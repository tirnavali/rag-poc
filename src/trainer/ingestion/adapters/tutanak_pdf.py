"""TBMM minutes PDF adapter — whole-document chunking with Docling.

One PDF = one session = one document.
No per-speaker splitting. Late chunking preserves full-session context
across all chunks.

The adapter receives metadata from DocumentInput (JSON manifest)
and uses Docling for PDF parsing. Speaker/session info comes from
DocumentInput, NOT from folder structure.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Optional

from src.common.parsing.docling_manager import DoclingManager
from src.config import settings
from src.trainer.ingestion.adapters.base import DocumentAdapter, DocumentInput
from src.trainer.ingestion.downloader import resolve_source


class TutanakPdfAdapter(DocumentAdapter):
    """TBMM minutes PDF → Docling → structural chunks.

    Metadata (period, session, date, author) comes from DocumentInput.
    Content comes from Docling-parsed PDF.
    """

    def __init__(self, docling: Optional[DoclingManager] = None):
        # Injected instance (if any) is treated as the OCR=True variant.
        # The no-OCR variant is created lazily on first use.
        self._docling_ocr: Optional[DoclingManager] = docling
        self._docling_no_ocr: Optional[DoclingManager] = None

    def _get_docling(self, use_ocr: bool) -> DoclingManager:
        if use_ocr:
            if self._docling_ocr is None:
                self._docling_ocr = DoclingManager(do_ocr=True)
            return self._docling_ocr
        else:
            if self._docling_no_ocr is None:
                self._docling_no_ocr = DoclingManager(do_ocr=False)
            return self._docling_no_ocr

    def parse(self, doc: DocumentInput) -> tuple[str, list[dict]]:
        if not doc.document_source:
            raise ValueError(f"tutanak adapter requires document_source (PDF path): {doc.document_id}")

        local_source, _etag, _last_mod = resolve_source(
            doc.document_source, doc.collection_name, doc.document_id
        )

        docling = self._get_docling(doc.ocr)
        full_text, raw_chunks = docling.convert_and_pack(
            local_source,
            min_chars=doc.min_chunk_chars or settings.MINUTES_MIN_CHUNK_CHARS,
            max_chars=doc.max_chunk_chars or settings.MINUTES_TARGET_CHUNK_CHARS,
            document_type=doc.document_type,
            initial_author=doc.author,
            initial_role=doc.author_role,
        )

        chunks = []
        for chunk in raw_chunks:
            meta = chunk["metadata"].copy() if "metadata" in chunk else {}

            # Override with canonical metadata from DocumentInput
            meta["document_id"] = doc.document_id
            meta["document_type"] = doc.document_type

            if doc.period is not None:
                meta["period"] = doc.period
            if doc.legislative_year is not None:
                meta["legislative_year"] = doc.legislative_year
            if doc.session is not None:
                meta["session"] = doc.session
            if doc.document_date:
                meta["date"] = doc.document_date
                meta["year"] = doc.year if doc.year is not None else int(doc.document_date[:4])
            if doc.author and not meta.get("author"):
                meta["author"] = doc.author
            if doc.author_role and not meta.get("author_role"):
                meta["author_role"] = doc.author_role
            if doc.source_name:
                meta["source_name"] = doc.source_name
            if doc.title:
                meta["source_title"] = doc.title
            if doc.topics:
                meta["topics"] = doc.topics

            # Merge type-specific metadata from DocumentInput
            if doc.metadata:
                for k, v in doc.metadata.items():
                    if k not in meta:
                        meta[k] = v

            chunks.append({
                "text": chunk["text"],
                "span": chunk.get("span"),
                "metadata": meta,
            })

        return full_text, chunks

    def compute_content_hash(self, doc: DocumentInput) -> str:
        if not doc.document_source:
            raise ValueError(f"tutanak adapter requires document_source for hash: {doc.document_id}")
        local_source, _etag, _last_mod = resolve_source(
            doc.document_source, doc.collection_name, doc.document_id
        )
        return _fast_file_hash(Path(local_source))


def _fast_file_hash(path: Path) -> str:
    """Fast hash: SHA-256(first 1MB) + file_size."""
    h = hashlib.sha256()
    size = path.stat().st_size
    with open(path, "rb") as f:
        h.update(f.read(1024 * 1024))
    h.update(str(size).encode())
    return h.hexdigest()[:32]
