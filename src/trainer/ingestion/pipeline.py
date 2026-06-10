"""Bu modül ingestion akışının orkestrasyonundan sorumludur. Parse (KATMAN 1)
ve chunk (KATMAN 2) işleri kendi modüllerine delege edilir — bu dosya
yalnızca aşamaları birbirine bağlar, ölçümler ve manifest durumunu yönetir.

Mimari rolü: ingestion pipeline'ının KATMAN 3'ü (INGEST).

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
        │           span coverage kontrolü — %100 değilse late chunking düşer
        ▼
    [4/6 EMBED]     Late chunking (Jina, tüm belge bağlamı) ya da standart embed
        │
        ▼
    [5/6 UPSERT]    ChromaDB'ye vektör + metadata yazımı
        │
        ▼
    [6/6 DONE]      Manifest + perf raporu yazımı, data_lake/reports/ sidecar JSON

Aşama özet tablosu — her aşamanın girdi/çıktısı ve hangi modülde yaşadığı:

  ┌───┬──────────┬──────────────────────┬──────────────────────────────┬───────────────────────────┐
  │ # │ Aşama    │ Girdi                │ Çıktı                        │ Modül                     │
  ├───┼──────────┼──────────────────────┼──────────────────────────────┼───────────────────────────┤
  │ 1 │ MANIFEST │ DocumentInput        │ skip / update / new kararı   │ manifest.py               │
  │ 2 │ PARSE    │ dosya yolu           │ atomlar + markdown + quality │ markdown_converter.py     │
  │ 3 │ CHUNK    │ ParsedDocument       │ chunk'lar (ocr_flagged dahil)│ docling_manager.py        │
  │ 4 │ EMBED    │ chunk metinleri+span │ embedding'ler                │ embedder.py               │
  │ 5 │ UPSERT   │ id + vector + meta   │ ChromaDB kayıtları           │ common/chroma             │
  │ 6 │ DONE     │ ölçümler + bayraklar │ manifest + reports/*.json    │ manifest.py + bu dosya    │
  └───┴──────────┴──────────────────────┴──────────────────────────────┴───────────────────────────┘

Her bileşen (parser, embedder) CollectionSpec üzerinden enjekte edilir;
ayrıntılı mimari ve disk artefakt haritası için bkz. README.md (bu dizinde).
"""
from __future__ import annotations

import json
import os
import statistics
import time
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    Progress, SpinnerColumn, TextColumn,
    BarColumn, MofNCompleteColumn, TimeElapsedColumn,
)
from rich.table import Table

from src.common.parsing.docling_manager import DoclingManager
from src.common.chroma import open_or_create_collection
from src.config import settings
from src.config.collections import CollectionSpec
from src.trainer.ingestion.adapters.base import DocumentInput, IngestResult, DocumentAdapter
from src.trainer.ingestion.adapters import get_adapter
from src.trainer.ingestion.downloader import is_url, validate_url
from src.trainer.ingestion.manifest import DocumentManifest
from src.trainer.ingestion.embedder import LocalLateChunkingEmbedder

_console = Console(highlight=False)


def _ms(t_start: float) -> int:
    """Saniyeyi milisaniyeye çevirir."""
    return int((time.perf_counter() - t_start) * 1000)


def _write_report(
    document_id: str,
    collection: str,
    status: str,
    timings: dict,
    chunk_stats: dict,
    warnings: list,
) -> None:
    """Her belge için data_lake/reports/{document_id}.json sidecar yazar."""
    try:
        reports_dir = settings.REPORTS_DIR
        reports_dir.mkdir(parents=True, exist_ok=True)
        safe_id = document_id.replace("/", "_").replace("\\", "_")
        path = reports_dir / f"{safe_id}.json"
        path.write_text(
            json.dumps(
                {
                    "document_id": document_id,
                    "collection": collection,
                    "status": status,
                    "ingest_time": datetime.now(timezone.utc).isoformat(),
                    "timings": timings,
                    "chunk": chunk_stats,
                    "warnings": warnings,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    except Exception as e:
        _console.print(f"  [dim][WARN] Rapor yazılamadı: {e}[/dim]")


def _chunk_stats(chunks: list, spans: list) -> dict:
    """Chunk listesinden özet istatistikler üretir."""
    n = len(chunks)
    if not n:
        return {"chunk_count": 0}
    char_counts = [len(c["text"]) for c in chunks]
    spans_present = sum(1 for s in spans if s is not None)
    chunk_types = {}
    for c in chunks:
        t = c.get("metadata", {}).get("type", "unknown")
        chunk_types[t] = chunk_types.get(t, 0) + 1
    return {
        "chunk_count": n,
        "span_coverage_pct": round(spans_present / n * 100, 1),
        "span_missing_count": n - spans_present,
        "char_min": min(char_counts),
        "char_median": round(statistics.median(char_counts)),
        "char_max": max(char_counts),
        "chunk_types": chunk_types,
    }


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
                    embed_dim=spec.embed_dim,
                )
            else:
                from langchain_ollama import OllamaEmbeddings
                self.embedder = OllamaEmbeddings(
                    model=spec.embed_model,
                    base_url=settings.OLLAMA_HOST,
                )
        else:
            self.embedder = embedder

        _console.print(
            f"[dim]--- ChromaDB başlatılıyor: {spec.db_path} "
            f"(Koleksiyon: {spec.name}) ---[/dim]"
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
        """Tek bir DocumentInput işler ve uçuş kaydı yazar.

        Akış (zamanlama ve span coverage dahil):
        1. Manifest kontrolü (skip / update / new)
        2. Adapter ile parse
        3. Chunk ID üret + span coverage kontrolü
        4. Embed (late chunking veya standart; span eksikse fallback uyarısı)
        5. ChromaDB upsert
        6. Manifest + perf raporu (data_lake/reports/) güncelle

        Returns:
            IngestResult with status ("done", "skipped", "failed")
        """
        t_total = time.perf_counter()
        _console.print(
            f"\n[bold][PIPELINE][/bold] [cyan]{doc.document_id}[/cyan] "
            f"[dim]({doc.document_type})[/dim]"
        )

        warnings: list[str] = []
        timings: dict[str, int] = {}
        cstats: dict = {}
        embed_mode = "standard"

        # Inject spec-level chunk params if not overridden per-document
        if doc.min_chunk_chars is None or doc.max_chunk_chars is None:
            doc = DocumentInput(**{
                **doc.to_dict(),
                "min_chunk_chars": doc.min_chunk_chars or self.spec.min_chunk_chars,
                "max_chunk_chars": doc.max_chunk_chars or self.spec.max_chunk_chars,
            })

        # ── 1. Manifest kontrolü ────────────────────────────────────
        t1 = time.perf_counter()
        source_etag: Optional[str] = None
        source_last_modified: Optional[str] = None
        try:
            adapter = get_adapter(doc.document_type)
            existing = self.manifest.get(doc.document_id, doc.collection_name)

            content_hash: Optional[str] = doc.content_hash

            if not content_hash and doc.document_source and is_url(doc.document_source):
                if existing and existing.source_etag:
                    ok, _msg = validate_url(doc.document_source)
                    if ok:
                        import urllib.request
                        req = urllib.request.Request(doc.document_source, method="HEAD")
                        with urllib.request.urlopen(req, timeout=10) as resp:
                            current_etag = resp.headers.get("ETag")
                            current_lm = resp.headers.get("Last-Modified")
                        if current_etag and current_etag == existing.source_etag:
                            _console.print(
                                f"  [dim][1/6 MANIFEST][/dim] SKIP — ETag değişmemiş"
                            )
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

        timings["manifest_ms"] = _ms(t1)
        existing = self.manifest.get(doc.document_id, doc.collection_name)
        if not force and existing and existing.content_hash == doc.content_hash and existing.status == "done":
            _console.print(
                f"  [dim][1/6 MANIFEST][/dim] SKIP — content_hash eşleşiyor "
                f"[dim]({timings['manifest_ms']}ms)[/dim]"
            )
            return IngestResult(
                document_id=doc.document_id,
                status="skipped",
                chunk_count=existing.chunk_count,
                reason="already_ingested",
            )

        if existing and existing.content_hash != doc.content_hash:
            _console.print(
                f"  [dim][1/6 MANIFEST][/dim] UPDATE — içerik değişmiş, eski chunk'lar siliniyor..."
            )
            self._delete_chunks(doc.document_id)

        _console.print(
            f"  [dim][1/6 MANIFEST][/dim] ✓ [dim]{timings['manifest_ms']}ms[/dim]"
        )

        # ── 2. Parse ────────────────────────────────────────────────
        _console.print(f"  [dim][2/6 PARSE][/dim] adapter: {doc.document_type}")
        t2 = time.perf_counter()
        try:
            full_text, chunks = adapter.parse(doc)
        except Exception as e:
            self.manifest.upsert(doc, status="failed", error_message=str(e))
            return IngestResult(
                document_id=doc.document_id,
                status="failed",
                reason=f"parse_error: {e}",
            )

        timings["parse_ms"] = _ms(t2)
        _console.print(
            f"  [dim][2/6 PARSE][/dim] ✓ [dim]{timings['parse_ms']}ms[/dim]"
            f"  · {len(chunks)} parça · {len(full_text):,} karakter"
        )

        if not chunks:
            self.manifest.upsert(doc, status="done", chunk_count=0)
            _write_report(doc.document_id, doc.collection_name, "done",
                          timings, {"chunk_count": 0}, warnings)
            return IngestResult(
                document_id=doc.document_id,
                status="done",
                chunk_count=0,
                reason="no_chunks",
            )

        # OCR kalite bayrağı (chunk metadata'dan)
        flagged_chunks = sum(1 for c in chunks if c.get("metadata", {}).get("ocr_flagged"))
        if flagged_chunks:
            w = f"ocr_flagged:{flagged_chunks}/{len(chunks)}"
            warnings.append(w)
            _console.print(
                f"  [bold yellow]⚠ OCR[/bold yellow] kalite bayrağı — "
                f"{flagged_chunks}/{len(chunks)} parça işaretli"
            )

        # ── 3. Chunk ID + span coverage kontrolü ────────────────────
        t3 = time.perf_counter()
        spans = [c.get("span") for c in chunks]
        spans_present = sum(1 for s in spans if s is not None)
        span_coverage_pct = spans_present / len(chunks) * 100

        cstats = _chunk_stats(chunks, spans)
        chunk_ids = [f"{doc.document_id}_{i}" for i in range(len(chunks))]
        documents_text = [c["text"] for c in chunks]

        if spans_present < len(chunks):
            missing = len(chunks) - spans_present
            w = f"span_missing:{missing}/{len(chunks)}"
            warnings.append(w)
            _console.print(
                f"  [bold yellow]⚠ SPAN COVERAGE[/bold yellow] "
                f"{missing}/{len(chunks)} chunk'ta span eksik "
                f"([yellow]{100 - span_coverage_pct:.0f}%[/yellow] kayıp) — "
                f"[yellow]late chunking bu belge için standart embed'e düşüyor[/yellow]"
            )
        else:
            _console.print(
                f"  [dim][3/6 CHUNK][/dim] ✓ [dim]{_ms(t3)}ms[/dim]"
                f"  · {len(chunks)} parça · span %100"
            )

        # Sanitize metadatas for ChromaDB (str, int, float, bool only)
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

        # ── 4. Embed ────────────────────────────────────────────────
        t4 = time.perf_counter()
        try:
            if self.spec.supports_late_chunking and hasattr(
                self.embedder, "embed_with_late_chunking_windowed"
            ):
                all_have_spans = all(s is not None for s in spans)
                if all_have_spans:
                    embed_mode = "late_chunking"
                    _console.print(
                        f"  [dim][4/6 EMBED][/dim] late chunking "
                        f"({len(chunks)} parça, context={self.spec.max_context_tokens})..."
                    )
                    embeddings = self.embedder.embed_with_late_chunking_windowed(
                        full_text,
                        spans,
                        max_tokens=self.spec.max_context_tokens,
                        overlap_tokens=self.spec.overlap_tokens,
                    )
                else:
                    embed_mode = "late_chunking_fallback"
                    _console.print(
                        f"  [dim][4/6 EMBED][/dim] [yellow]standart embed (span eksikliği nedeniyle fallback)[/yellow]"
                        f" ({len(chunks)} parça)..."
                    )
                    embeddings = self.embedder.embed_documents(documents_text)
            else:
                _console.print(
                    f"  [dim][4/6 EMBED][/dim] standart embed ({len(chunks)} parça)..."
                )
                embeddings = self.embedder.embed_documents(documents_text)
        except Exception as e:
            self.manifest.upsert(doc, status="failed", error_message=str(e))
            return IngestResult(
                document_id=doc.document_id,
                status="failed",
                reason=f"embed_error: {e}",
            )

        timings["embed_ms"] = _ms(t4)
        _console.print(
            f"  [dim][4/6 EMBED][/dim] ✓ [dim]{timings['embed_ms']}ms[/dim]"
            f"  · mod: {embed_mode}"
        )

        # ── 5. ChromaDB upsert ───────────────────────────────────────
        t5 = time.perf_counter()
        _console.print(f"  [dim][5/6 UPSERT][/dim] ChromaDB: {self.spec.name}")
        try:
            self.collection.upsert(
                ids=chunk_ids,
                embeddings=embeddings,
                documents=documents_text,
                metadatas=metadatas,
            )
        except Exception as e:
            self.manifest.upsert(doc, status="failed", error_message=str(e))
            return IngestResult(
                document_id=doc.document_id,
                status="failed",
                reason=f"upsert_error: {e}",
            )

        timings["upsert_ms"] = _ms(t5)
        timings["total_ms"] = _ms(t_total)

        # Perf özeti (manifest + rapor için)
        perf = {
            **timings,
            "span_coverage_pct": round(span_coverage_pct, 1),
            "embed_mode": embed_mode,
            "chunk_count": len(chunks),
        }

        # ── 6. Manifest + rapor güncelle ────────────────────────────
        self.manifest.upsert(
            doc,
            status="done",
            chunk_count=len(chunks),
            source_etag=source_etag,
            source_last_modified=source_last_modified,
            quality={"ocr_flagged": bool(flagged_chunks)},
            perf=perf,
        )
        _write_report(
            doc.document_id, doc.collection_name, "done",
            timings, {**cstats, "embed_mode": embed_mode}, warnings,
        )

        # ── Özet paneli ─────────────────────────────────────────────
        warn_str = (
            f"  [bold yellow]⚠ {' · '.join(warnings)}[/bold yellow]"
            if warnings
            else "  [green]✓ uyarı yok[/green]"
        )
        _console.print(
            Panel(
                f"[dim]manifest[/dim] {timings['manifest_ms']}ms  "
                f"[dim]parse[/dim] {timings['parse_ms']}ms  "
                f"[dim]embed[/dim] {timings['embed_ms']}ms  "
                f"[dim]upsert[/dim] {timings['upsert_ms']}ms  "
                f"[bold]toplam[/bold] {timings['total_ms']}ms\n"
                f"{len(chunks)} parça · span {span_coverage_pct:.0f}%"
                f"{warn_str}",
                title=f"[bold {'yellow' if warnings else 'green'}]{doc.document_id}[/bold {'yellow' if warnings else 'green'}]",
                border_style="yellow" if warnings else "green",
                expand=False,
            )
        )

        return IngestResult(
            document_id=doc.document_id,
            status="done",
            chunk_count=len(chunks),
        )

    # ─── Batch processing ────────────────────────────────────────────

    def run_batch(self, documents: List[DocumentInput], force: bool = False) -> List[IngestResult]:
        """Birden fazla dokümanı sırayla işler; Rich progress bar gösterir."""
        results = []
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=_console,
            transient=False,
        ) as progress:
            task = progress.add_task(
                f"[cyan]{self.spec.name}[/cyan]",
                total=len(documents),
            )
            for doc in documents:
                progress.update(
                    task,
                    description=f"[cyan]{doc.document_id[:50]}[/cyan]",
                )
                try:
                    result = self.run_document(doc, force=force)
                    results.append(result)
                except Exception as e:
                    _console.print(f"  [red][ERROR][/red] {doc.document_id}: {e}")
                    self.manifest.upsert(doc, status="failed", error_message=str(e))
                    results.append(
                        IngestResult(
                            document_id=doc.document_id,
                            status="failed",
                            reason=str(e),
                        )
                    )
                finally:
                    progress.advance(task)

        return results

    # ─── Chunk deletion ──────────────────────────────────────────────

    def _delete_chunks(self, document_id: str) -> int:
        """Bir document_id'ye ait tüm chunk'ları ChromaDB'den sil."""
        prefix = f"{document_id}_"
        try:
            all_ids = self.collection.get(include=[])["ids"]
            to_delete = [id_ for id_ in all_ids if id_.startswith(prefix)]
            if to_delete:
                self.collection.delete(ids=to_delete)
                _console.print(f"  [dim][DELETE] {len(to_delete)} eski parça silindi.[/dim]")
                return len(to_delete)
        except Exception as e:
            _console.print(f"  [yellow][WARN][/yellow] Chunk silme başarısız: {e}")
        return 0
