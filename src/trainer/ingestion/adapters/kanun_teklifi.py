"""Kanun teklifi (parliamentary bill) PDF adapter.

Same parsing as TutanakPdfAdapter but with kanun_teklifi-specific metadata:
- esas_no, ozet, durum, detay_url → metadata
- author / author_role for MP info
- document_source is always a PDF URL (or local path)

Uses Docling for PDF parsing and supports URL sources via resolve_source().
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from src.common.parsing.docling_manager import DoclingManager
from src.config import settings
from src.trainer.ingestion.adapters.base import DocumentAdapter, DocumentInput
from src.trainer.ingestion.adapters.tutanak_pdf import _fast_file_hash
from src.trainer.ingestion.downloader import resolve_source


class KanunTeklifiAdapter(DocumentAdapter):
    """Kanun teklifi PDF → Docling → structural chunks.

    Metadata comes from DocumentInput (JSON manifest).
    PDF content comes from Docling-parsed file (local or URL).
    """

    def __init__(self, docling: Optional[DoclingManager] = None):
        self._docling_ocr: Optional[DoclingManager] = docling
        self._docling_no_ocr: Optional[DoclingManager] = None

    def _get_docling(self, use_ocr: bool) -> DoclingManager:
        """OCR gereksinimine göre uygun DoclingManager nesnesini döner (Lazy-loading).

        Args:
            use_ocr: OCR (Optik Karakter Tanıma) kullanılıp kullanılmayacağı.

        Returns:
            Yapılandırılmış DoclingManager örneği.
        """
        if use_ocr:
            if self._docling_ocr is None:
                self._docling_ocr = DoclingManager(do_ocr=True)
            return self._docling_ocr
        else:
            if self._docling_no_ocr is None:
                self._docling_no_ocr = DoclingManager(do_ocr=False)
            return self._docling_no_ocr

    def parse(self, doc: DocumentInput) -> tuple[str, list[dict]]:
        """Kanun teklifi dökümanını PDF kaynağından okur, Docling ile metne dönüştürür
        ve parçalara (chunk) ayırır.
        Döküman içeriğine ek olarak teklife özel üstverileri (meta) her parçaya ekler.
        Args:
            doc: İşlenecek dökümanın giriş verileri ve ayarları.

        Returns:
            Dökümanın tam metni ve parçalanmış hali (liste içinde sözlükler).
        """
        if not doc.document_source:
            raise ValueError(
                f"kanun_teklifi adapter requires document_source (PDF path/URL): {doc.document_id}"
            )

        local_source, _etag, _last_mod = resolve_source(
            doc.document_source, doc.collection_name, doc.document_id
        )

        full_text, raw_chunks = self._get_docling(doc.ocr).convert_and_pack(
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

            meta["document_id"] = doc.document_id
            meta["document_type"] = doc.document_type

            if doc.period is not None:
                meta["period"] = doc.period
            if doc.legislative_year is not None:
                meta["legislative_year"] = doc.legislative_year
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

            # Kanun teklifi-specific metadata
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
        """Döküman kaynağının (PDF) içeriğine dayalı hızlı bir hash değeri hesaplar.

        Bu hash, dökümanın değişip değişmediğini kontrol etmek için (deduplication) kullanılır.

        Args:
            doc: Kaynak bilgisini içeren döküman nesnesi.

        Returns:
            Dosya içeriğinin hash stringi.
        """
        if not doc.document_source:
            raise ValueError(
                f"kanun_teklifi adapter requires document_source for hash: {doc.document_id}"
            )
        local_source, _etag, _last_mod = resolve_source(
            doc.document_source, doc.collection_name, doc.document_id
        )
        return _fast_file_hash(Path(local_source))
