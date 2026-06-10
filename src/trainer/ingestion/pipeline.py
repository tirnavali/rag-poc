"""Uçtan uca ingestion boru hattı — manifest tabanlı dedup ile.

Bir belgenin izlediği yol (konsoldaki [n/6] etiketleri bu akışla eşleşir):

    manifest.json (DocumentInput)
        │
        ▼
    [1/6 MANIFEST]  content_hash / ETag kontrolü ──► değişmemişse SKIP
        │
        ▼
    [2/6 PARSE]     adapter → DoclingManager → MarkdownConverter
        │           (PDF → OCR → atomlar + DoclingDocument; parse_cache/)
        ▼
    [3/6 CHUNK]     HybridChunker / author-aware / greedy paketleme
        │           chunk ID: {document_id}_{i} — deterministik, dedup-safe
        ▼
    [4/6 EMBED]     Late chunking (Jina, tüm belge bağlamı) ya da standart embed
        │
        ▼
    [5/6 UPSERT]    ChromaDB'ye vektör + metadata yazımı
        │
        ▼
    [6/6 DONE]      Manifest'e done + chunk_count + ETag kaydı

Her bileşen (parser, embedder) CollectionSpec üzerinden enjekte edilir;
ayrıntılı mimari ve disk artefakt haritası için bkz. README.md (bu dizinde).
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from pathlib import Path

from src.common.parsing.docling_manager import DoclingManager
from src.common.chroma import open_or_create_collection
from src.config import settings
from src.config.collections import CollectionSpec
from src.trainer.ingestion.adapters.base import DocumentInput, IngestResult, DocumentAdapter
from src.trainer.ingestion.adapters import get_adapter
from src.trainer.ingestion.downloader import is_url, validate_url
from src.trainer.ingestion.manifest import DocumentManifest
from src.trainer.ingestion.embedder import LocalLateChunkingEmbedder


class IngestionPipeline:
    """Uçtan uca veri yükleme boru hattı.

    Her bileşen (parser, embedder) CollectionSpec üzerinden enjekte edilir.
    Manifest tabanlı dedup: aynı document_id + aynı content_hash → atlanır.
    """

    def __init__(
        self,
        spec: CollectionSpec,
        parser=None,
        embedder=None,
        manifest: Optional[DocumentManifest] = None,
    ):
        self.spec = spec
        self.docling = parser or DoclingManager(
            tokenizer_name=spec.embed_model if spec.supports_late_chunking else None,
            max_chunk_tokens=spec.max_chunk_tokens,
            min_chunk_tokens=spec.min_chunk_tokens,
        )
        self.manifest = manifest or DocumentManifest()

        # Embedder: spec'e göre late chunking veya standart
        if embedder is None:
            if spec.supports_late_chunking:
                self.embedder = LocalLateChunkingEmbedder(
                    model_name=spec.embed_model,
                    max_context_tokens=spec.max_context_tokens,
                    overlap_tokens=spec.overlap_tokens,
                )
            else:
                from langchain_ollama import OllamaEmbeddings
                self.embedder = OllamaEmbeddings(
                    model=spec.embed_model,
                    base_url=settings.OLLAMA_HOST,
                )
        else:
            self.embedder = embedder

        print(
            f"--- ChromaDB başlatılıyor: {spec.db_path} "
            f"(Koleksiyon: {spec.name}) ---"
        )
        self.client, self.collection = open_or_create_collection(
            spec.db_path, spec.name
        )

    # ─── Properties ─────────────────────────────────────────────────

    @property
    def db_path(self) -> Path:
        return self.spec.db_path

    @property
    def collection_name(self) -> str:
        return self.spec.name

    # ─── Primary: run_document ───────────────────────────────────────

    def run_document(self, doc: DocumentInput, force: bool = False) -> IngestResult:
        """Tek bir DocumentInput işler.

        Akış:
        1. Manifest kontrolü (skip / update / new)
        2. Adapter ile parse
        3. Chunk ID üret (deterministik)
        4. Embed (late chunking veya standart)
        5. ChromaDB upsert
        6. Manifest güncelle

        Returns:
            IngestResult with status ("done", "skipped", "failed")
        """
        print(f"\n[PIPELINE] {doc.document_id} ({doc.document_type})")

        # Inject spec-level chunk params if not overridden per-document
        if doc.min_chunk_chars is None or doc.max_chunk_chars is None:
            doc = DocumentInput(**{
                **doc.to_dict(),
                "min_chunk_chars": doc.min_chunk_chars or self.spec.min_chunk_chars,
                "max_chunk_chars": doc.max_chunk_chars or self.spec.max_chunk_chars,
            })

        # 1. Manifest kontrolü
        source_etag: Optional[str] = None
        source_last_modified: Optional[str] = None
        try:
            adapter = get_adapter(doc.document_type)
            existing = self.manifest.get(doc.document_id, doc.collection_name)

            # URL akıllı skip: ETag/Last-Modified kontrolü
            content_hash: Optional[str] = doc.content_hash

            if not content_hash and doc.document_source and is_url(doc.document_source):
                if existing and existing.source_etag:
                    # HEAD at, ETag karşılaştır
                    ok, _msg = validate_url(doc.document_source)
                    if ok:
                        import urllib.request
                        req = urllib.request.Request(doc.document_source, method="HEAD")
                        with urllib.request.urlopen(req, timeout=10) as resp:
                            current_etag = resp.headers.get("ETag")
                            current_lm = resp.headers.get("Last-Modified")
                        if current_etag and current_etag == existing.source_etag:
                            print(f"  [1/6 MANIFEST] SKIP — ETag değişmemiş, indirme yapılmıyor")
                            return IngestResult(
                                document_id=doc.document_id,
                                status="skipped",
                                chunk_count=existing.chunk_count,
                                reason="already_ingested (etag match)",
                            )
                        source_etag = current_etag
                        source_last_modified = current_lm

            if content_hash is None:
                content_hash = adapter.compute_content_hash(doc)
            doc = DocumentInput(**{**doc.to_dict(), "content_hash": content_hash})
        except Exception as e:
            self.manifest.upsert(doc, status="failed", error_message=str(e))
            return IngestResult(document_id=doc.document_id, status="failed", reason=str(e))

        existing = self.manifest.get(doc.document_id, doc.collection_name)
        if not force and existing and existing.content_hash == doc.content_hash and existing.status == "done":
            print(f"  [1/6 MANIFEST] SKIP — Zaten başarıyla işlenmiş (content_hash eşleşiyor)")
            return IngestResult(
                document_id=doc.document_id,
                status="skipped",
                chunk_count=existing.chunk_count,
                reason="already_ingested",
            )

        if existing and existing.content_hash != doc.content_hash:
            print(f"  [1/6 MANIFEST] UPDATE — İçerik değişmiş, eski chunk'lar siliniyor...")
            self._delete_chunks(doc.document_id)

        # 2. Parse
        print(f"  [2/6 PARSE] Adapter: {doc.document_type}")
        try:
            full_text, chunks = adapter.parse(doc)
        except Exception as e:
            self.manifest.upsert(doc, status="failed", error_message=str(e))
            return IngestResult(
                document_id=doc.document_id,
                status="failed",
                reason=f"parse_error: {e}",
            )

        if not chunks:
            self.manifest.upsert(doc, status="done", chunk_count=0)
            return IngestResult(
                document_id=doc.document_id,
                status="done",
                chunk_count=0,
                reason="no_chunks",
            )

        # Tier-1 OCR kalite bayrağı (chunk metadata'sından, bkz. quality.py)
        flagged_chunks = sum(
            1 for c in chunks if c.get("metadata", {}).get("ocr_flagged")
        )
        if flagged_chunks:
            print(
                f"  [WARN] OCR kalite bayrağı: {doc.document_id} — düşük kalite "
                f"sinyali ({flagged_chunks}/{len(chunks)} parça işaretli)"
            )

        # 3. Chunk ID üret
        print(f"  [3/6 CHUNK] {len(chunks)} parça, ID formatı: {doc.document_id}_{{i}}")
        chunk_ids = [f"{doc.document_id}_{i}" for i in range(len(chunks))]
        documents = [c["text"] for c in chunks]
        
        # Sanitize metadatas for ChromaDB compatibility (only allow str, int, float, bool)
        metadatas = []
        for c in chunks:
            clean_meta = {}
            for k, v in c["metadata"].items():
                if v is None:
                    continue
                if isinstance(v, (str, int, float, bool)):
                    clean_meta[k] = v
                elif isinstance(v, (list, tuple, set)):
                    clean_meta[k] = ", ".join(str(item) for item in v)
                elif isinstance(v, dict):
                    clean_meta[k] = json.dumps(v, ensure_ascii=False)
                else:
                    clean_meta[k] = str(v)
            metadatas.append(clean_meta)

        # 4. Embed
        try:
            if self.spec.supports_late_chunking and hasattr(self.embedder, "embed_with_late_chunking_windowed"):
                spans = [c.get("span") for c in chunks]
                all_have_spans = all(s is not None for s in spans)
                if all_have_spans:
                    print(
                        f"  [4/6 EMBED] Late Chunking ({len(chunks)} parça, "
                        f"context={self.spec.max_context_tokens})..."
                    )
                    embeddings = self.embedder.embed_with_late_chunking_windowed(
                        full_text,
                        spans,
                        max_tokens=self.spec.max_context_tokens,
                        overlap_tokens=self.spec.overlap_tokens,
                    )
                else:
                    # Bazı chunk'larda span eksik — standart embed
                    embeddings = self.embedder.embed_documents(documents)
            else:
                print(f"  [4/6 EMBED] Standart Embed ({len(chunks)} parça)...")
                embeddings = self.embedder.embed_documents(documents)
        except Exception as e:
            self.manifest.upsert(doc, status="failed", error_message=str(e))
            return IngestResult(
                document_id=doc.document_id,
                status="failed",
                reason=f"embed_error: {e}",
            )

        # 5. ChromaDB upsert
        print(f"  [5/6 UPSERT] ChromaDB: {self.spec.name}")
        try:
            self.collection.upsert(
                ids=chunk_ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
            )
        except Exception as e:
            self.manifest.upsert(doc, status="failed", error_message=str(e))
            return IngestResult(
                document_id=doc.document_id,
                status="failed",
                reason=f"upsert_error: {e}",
            )

        # 6. Manifest güncelle
        self.manifest.upsert(
            doc,
            status="done",
            chunk_count=len(chunks),
            source_etag=source_etag,
            source_last_modified=source_last_modified,
            quality={"ocr_flagged": bool(flagged_chunks)},
        )
        print(f"  [6/6 DONE] {len(chunks)} parça eklendi.")

        return IngestResult(
            document_id=doc.document_id,
            status="done",
            chunk_count=len(chunks),
        )

    # ─── Batch processing ────────────────────────────────────────────

    def run_batch(self, documents: List[DocumentInput], force: bool = False) -> List[IngestResult]:
        """Birden fazla dokümanı sırayla işler."""
        results = []
        for doc in documents:
            try:
                result = self.run_document(doc, force=force)
                results.append(result)
            except Exception as e:
                print(f"  [ERROR] {doc.document_id}: {e}")
                self.manifest.upsert(
                    doc, status="failed", error_message=str(e)
                )
                results.append(
                    IngestResult(
                        document_id=doc.document_id,
                        status="failed",
                        reason=str(e),
                    )
                )
        return results

    # ─── Chunk deletion ──────────────────────────────────────────────

    def _delete_chunks(self, document_id: str) -> int:
        """Bir document_id'ye ait tüm chunk'ları ChromaDB'den sil."""
        prefix = f"{document_id}_"
        # ChromaDB'nin where filtresiyle prefix desteği yok.
        # Pratik çözüm: koleksiyondaki tüm ID'leri çek, prefix eşleşenleri bul, delete.
        try:
            all_ids = self.collection.get(include=[])["ids"]
            to_delete = [id_ for id_ in all_ids if id_.startswith(prefix)]
            if to_delete:
                self.collection.delete(ids=to_delete)
                print(f"  [DELETE] {len(to_delete)} eski parça silindi.")
                return len(to_delete)
        except Exception as e:
            print(f"  [WARN] Chunk silme başarısız: {e}")
        return 0
