"""Document manifest — SQLite-backed state tracker for ingestion.

Tracks every document ever ingested:
- document_id (primary key) for deduplication
- content_hash for change detection
- status for lifecycle tracking
- All canonical metadata fields for querying

Design principle: SINGLE shared manifest for ALL collections.
This enables cross-collection queries ("show me all tutanak ever ingested").
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from src.config import settings
from src.trainer.ingestion.adapters.base import DocumentInput, ManifestRecord


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS document_manifest (
    document_id       TEXT NOT NULL,
    collection_name   TEXT NOT NULL,
    document_source   TEXT,
    document_type     TEXT NOT NULL,
    content_hash      TEXT NOT NULL,
    status            TEXT NOT NULL DEFAULT 'pending',
    chunk_count       INTEGER DEFAULT 0,
    document_date     TEXT,
    year              INTEGER,
    period            INTEGER,
    legislative_year  INTEGER,
    session           INTEGER,
    author            TEXT,
    author_role       TEXT,
    source_name       TEXT,
    title             TEXT,
    topics            TEXT,
    ingest_time       TEXT NOT NULL,
    last_modified     TEXT NOT NULL,
    error_message     TEXT,
    metadata_json     TEXT,
    source_etag       TEXT,
    source_last_modified TEXT,
    ocr               INTEGER NOT NULL DEFAULT 1,
    quality_json      TEXT,
    PRIMARY KEY (document_id, collection_name)
);

CREATE INDEX IF NOT EXISTS idx_manifest_type ON document_manifest(document_type);
CREATE INDEX IF NOT EXISTS idx_manifest_period ON document_manifest(period);
CREATE INDEX IF NOT EXISTS idx_manifest_year ON document_manifest(year);
CREATE INDEX IF NOT EXISTS idx_manifest_status ON document_manifest(status);
CREATE INDEX IF NOT EXISTS idx_manifest_collection ON document_manifest(collection_name);
CREATE INDEX IF NOT EXISTS idx_manifest_author ON document_manifest(author);
CREATE INDEX IF NOT EXISTS idx_manifest_source_name ON document_manifest(source_name);
"""


class DocumentManifest:
    """SQLite-backed document ingestion manifest.

    Provides CRUD operations, change detection, and status queries.
    Thread-safe for single-process use (SQLite handles locking).
    """

    def __init__(self, db_path: Path = settings.MANIFEST_DB):
        self.db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path))
        self._conn.row_factory = sqlite3.Row
        self._ensure_schema()

    def __del__(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass

    def _ensure_schema(self) -> None:
        """Create tables and indices if they don't exist."""
        self._conn.executescript(SCHEMA_SQL)
        self._migrate()
        self._conn.commit()

    def _migrate(self) -> None:
        """Add columns introduced after initial schema — safe to run repeatedly."""
        new_columns = [
            ("source_etag", "TEXT"),
            ("source_last_modified", "TEXT"),
            ("ocr", "INTEGER NOT NULL DEFAULT 1"),
            ("quality_json", "TEXT"),
        ]
        for col, typedef in new_columns:
            try:
                self._conn.execute(
                    f"ALTER TABLE document_manifest ADD COLUMN {col} {typedef}"
                )
            except sqlite3.OperationalError:
                pass  # column already exists

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def get(self, document_id: str, collection_name: str) -> Optional[ManifestRecord]:
        """Lookup a manifest record by document_id and collection."""
        row = self._conn.execute(
            "SELECT * FROM document_manifest WHERE document_id = ? AND collection_name = ?",
            (document_id, collection_name),
        ).fetchone()
        return self._row_to_record(row) if row else None

    def upsert(self, doc: DocumentInput, status: str, chunk_count: int = 0,
               error_message: Optional[str] = None,
               source_etag: Optional[str] = None,
               source_last_modified: Optional[str] = None,
               quality: Optional[dict] = None) -> None:
        """Insert or update a manifest record.

        quality: opsiyonel Tier-1 OCR kalite özeti (örn. {"ocr_flagged": true}).
        None geçilirse mevcut değer korunur (COALESCE).
        """
        now = _iso_now()
        existing = self.get(doc.document_id, doc.collection_name)
        if existing:
            ingest_time = existing.ingest_time  # Preserve original
        else:
            ingest_time = now

        self._conn.execute(
            """
            INSERT INTO document_manifest
            (document_id, collection_name, document_source, document_type, content_hash,
             status, chunk_count, document_date, year, period, legislative_year, session,
             author, author_role, source_name, title, topics,
             ingest_time, last_modified, error_message, metadata_json,
             source_etag, source_last_modified, ocr, quality_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(document_id, collection_name) DO UPDATE SET
                document_source = excluded.document_source,
                document_type = excluded.document_type,
                content_hash = excluded.content_hash,
                status = excluded.status,
                chunk_count = excluded.chunk_count,
                document_date = excluded.document_date,
                year = excluded.year,
                period = excluded.period,
                legislative_year = excluded.legislative_year,
                session = excluded.session,
                author = excluded.author,
                author_role = excluded.author_role,
                source_name = excluded.source_name,
                title = excluded.title,
                topics = excluded.topics,
                last_modified = excluded.last_modified,
                error_message = excluded.error_message,
                metadata_json = excluded.metadata_json,
                source_etag = excluded.source_etag,
                source_last_modified = excluded.source_last_modified,
                ocr = excluded.ocr,
                quality_json = COALESCE(excluded.quality_json, document_manifest.quality_json)
            """,
            (
                doc.document_id, doc.collection_name, doc.document_source, doc.document_type, doc.content_hash,
                status, chunk_count, doc.document_date, doc.year, doc.period, doc.legislative_year, doc.session,
                doc.author, doc.author_role, doc.source_name, doc.title, doc.topics,
                ingest_time, now, error_message,
                json.dumps(doc.metadata, ensure_ascii=False) if doc.metadata else None,
                source_etag,
                source_last_modified,
                int(doc.ocr),
                json.dumps(quality, ensure_ascii=False) if quality else None,
            ),
        )
        self._conn.commit()

    def delete(self, document_id: str, collection_name: str) -> None:
        """Delete a manifest record."""
        self._conn.execute(
            "DELETE FROM document_manifest WHERE document_id = ? AND collection_name = ?",
            (document_id, collection_name),
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Batch queries
    # ------------------------------------------------------------------

    def diff(self, documents: list[DocumentInput]) -> tuple[list[DocumentInput], list[DocumentInput], list[DocumentInput]]:
        """Compare a list of DocumentInputs against the manifest.

        Returns:
            (new_docs, changed_docs, unchanged_docs)
        """
        if not documents:
            return [], [], []

        # We need to check existence per (ID, collection)
        new_docs: list[DocumentInput] = []
        changed_docs: list[DocumentInput] = []
        unchanged_docs: list[DocumentInput] = []

        # Optimization: Fetch all relevant records in one query if possible
        # For simplicity in this POC, we'll fetch them individually or use a temporary table
        # Given small batches in ingestion, individual gets are acceptable
        for doc in documents:
            existing = self.get(doc.document_id, doc.collection_name)
            if existing is None:
                new_docs.append(doc)
            elif existing.content_hash != doc.content_hash:
                changed_docs.append(doc)
            else:
                unchanged_docs.append(doc)

        return new_docs, changed_docs, unchanged_docs

    def list_by_type(self, document_type: str) -> list[ManifestRecord]:
        rows = self._conn.execute(
            "SELECT * FROM document_manifest WHERE document_type = ? ORDER BY document_id",
            (document_type,),
        ).fetchall()
        return [self._row_to_record(r) for r in rows]

    def list_by_period(self, period: int) -> list[ManifestRecord]:
        rows = self._conn.execute(
            "SELECT * FROM document_manifest WHERE period = ? ORDER BY document_date",
            (period,),
        ).fetchall()
        return [self._row_to_record(r) for r in rows]

    def list_by_collection(self, collection_name: str) -> list[ManifestRecord]:
        rows = self._conn.execute(
            "SELECT * FROM document_manifest WHERE collection_name = ? ORDER BY document_id",
            (collection_name,),
        ).fetchall()
        return [self._row_to_record(r) for r in rows]

    def list_by_status(self, status: str) -> list[ManifestRecord]:
        rows = self._conn.execute(
            "SELECT * FROM document_manifest WHERE status = ? ORDER BY document_id",
            (status,),
        ).fetchall()
        return [self._row_to_record(r) for r in rows]

    def count_by_status(self) -> dict[str, int]:
        rows = self._conn.execute(
            "SELECT status, COUNT(*) FROM document_manifest GROUP BY status"
        ).fetchall()
        return {r[0]: r[1] for r in rows}

    def count_by_collection(self) -> dict[str, dict[str, int]]:
        rows = self._conn.execute(
            "SELECT collection_name, document_type, status, COUNT(*) "
            "FROM document_manifest GROUP BY collection_name, document_type, status"
        ).fetchall()
        result: dict = {}
        for col, dtype, status, count in rows:
            result.setdefault(col, {}).setdefault(dtype, {})[status] = count
        return result

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> ManifestRecord:
        return ManifestRecord(
            document_id=row["document_id"],
            document_source=row["document_source"],
            document_type=row["document_type"],
            collection_name=row["collection_name"],
            content_hash=row["content_hash"],
            status=row["status"],
            chunk_count=row["chunk_count"],
            document_date=row["document_date"],
            year=row["year"],
            period=row["period"],
            legislative_year=row["legislative_year"],
            session=row["session"],
            author=row["author"],
            author_role=row["author_role"],
            source_name=row["source_name"],
            title=row["title"],
            topics=row["topics"],
            ingest_time=row["ingest_time"],
            last_modified=row["last_modified"],
            error_message=row["error_message"],
            metadata_json=row["metadata_json"],
            ocr=bool(row["ocr"]),
            source_etag=row["source_etag"],
            source_last_modified=row["source_last_modified"],
            quality_json=row["quality_json"],
        )


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()
