"""Generic PDF report adapter — Docling → structural chunks.

Minimal metadata: just what DocumentInput provides.
No TBMM-specific fields.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from src.common.parsing.docling_manager import DoclingManager
from src.config import settings
from src.trainer.ingestion.adapters.base import DocumentAdapter, DocumentInput
from src.trainer.ingestion.adapters.tutanak_pdf import _fast_file_hash
from src.trainer.ingestion.downloader import resolve_source


class PdfReportAdapter(DocumentAdapter):
    """Generic PDF → Docling → structural chunks.

    Same parsing as TutanakPdfAdapter but without TBMM-specific
    metadata merging. Uses canonical fields only.
    """

    def __init__(self, docling: Optional[DoclingManager] = None):
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
            raise ValueError(
                f"pdf_report adapter requires document_source (PDF path): {doc.document_id}"
            )

        local_source, _etag, _last_mod = resolve_source(
            doc.document_source, doc.collection_name, doc.document_id
        )

        full_text, raw_chunks = self._get_docling(doc.ocr).convert_and_pack(
            local_source,
            min_chars=doc.min_chunk_chars or settings.MINUTES_MIN_CHUNK_CHARS,
            max_chars=doc.max_chunk_chars or settings.MINUTES_TARGET_CHUNK_CHARS,
        )

        chunks = []
        for chunk in raw_chunks:
            meta = chunk["metadata"].copy() if "metadata" in chunk else {}

            meta["document_id"] = doc.document_id
            meta["document_type"] = doc.document_type

            if doc.document_date:
                meta["date"] = doc.document_date
                meta["year"] = doc.year if doc.year is not None else int(doc.document_date[:4])
            if doc.author:
                meta["author"] = doc.author
            if doc.author_role:
                meta["author_role"] = doc.author_role
            if doc.source_name:
                meta["source_name"] = doc.source_name
            if doc.title:
                meta["source_title"] = doc.title
            if doc.topics:
                meta["topics"] = doc.topics
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
            raise ValueError(
                f"pdf_report adapter requires document_source for hash: {doc.document_id}"
            )
        local_source, _etag, _last_mod = resolve_source(
            doc.document_source, doc.collection_name, doc.document_id
        )
        return _fast_file_hash(Path(local_source))
