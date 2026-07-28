"""Resolve chunk_ids to character-range spans via parse cache.

Shared by scripts/vector_explorer.py (UI) and src/evaluator/benchmark.py (metrics).
Extracted from vector_explorer to decouple from Streamlit.
"""

import re
import json
import sqlite3
import hashlib
from pathlib import Path
from src.config import settings


def _file_hash(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(4096), b""):
            h.update(block)
    return h.hexdigest()


def _atompack_chunk_cache_key(spec, ocr_base: str) -> str:
    """docling_manager.pack (atom-token-pack yol) ile birebir aynı chunk cache anahtarı.

    Token-tabanlı: tokenizer_name=spec.embed_model + max/min_chunk_tokens
    (+ koşullu _ovl eki + author_tag).
    """
    author_tag = (
        f"_author_{spec.doc_type.value}" if getattr(spec, "doc_type", None) else ""
    )
    overlap = getattr(spec, "chunk_overlap_tokens", 0) or 0
    overlap_tag = f"_ovl{overlap}" if overlap else ""
    return hashlib.md5(
        f"{ocr_base}_atompack_{spec.embed_model}_{spec.max_chunk_tokens}"
        f"_{spec.min_chunk_tokens}{overlap_tag}{author_tag}".encode()
    ).hexdigest()


def chunk_id_to_span(chunk_id: str, spec, manifest_conn=None) -> dict | None:
    """Resolve single chunk_id to {'document_id', 'char_start', 'char_end', 'text'}.

    Returns None on failure (bad format, missing manifest row, missing cache, etc).
    If manifest_conn provided, reuses it; else opens fresh connection.
    """
    match = re.match(r"^(.+)_(\d+)$", chunk_id)
    if not match:
        return None

    doc_id = match.group(1)
    chunk_idx = int(match.group(2))

    own_conn = manifest_conn is None
    try:
        if own_conn:
            conn = sqlite3.connect(str(settings.MANIFEST_DB))
            conn.row_factory = sqlite3.Row
        else:
            conn = manifest_conn

        row = conn.execute(
            "SELECT document_source FROM document_manifest WHERE document_id = ?",
            (doc_id,),
        ).fetchone()

        if not row or not row["document_source"]:
            return None

        src_path = row["document_source"]
        if not Path(src_path).exists():
            return None

        file_hash = _file_hash(src_path)
        ocr_engine = settings.OCR_ENGINE
        ocr_tag = ""
        ocr_base = f"{file_hash}_{ocr_engine}{ocr_tag}"
        chunk_cache_key = _atompack_chunk_cache_key(spec, ocr_base)
        cache_file = settings.PARSE_CACHE_DIR / f"{chunk_cache_key}.json"

        if not cache_file.exists():
            return None

        data = json.loads(cache_file.read_text(encoding="utf-8"))
        chunks = data.get("chunks", [])
        if chunk_idx >= len(chunks):
            return None

        chunk = chunks[chunk_idx]
        span = chunk.get("span")
        if not span or len(span) < 2:
            return None

        return {
            "document_id": doc_id,
            "char_start": span[0],
            "char_end": span[1],
            "text": chunk.get("text", ""),
            "_chunk_id": chunk_id,
            "_text": chunk.get("text", ""),
        }

    except Exception:
        return None
    finally:
        if own_conn:
            conn.close()


def resolve_spans_from_cache(chunk_ids: list[str], spec) -> tuple[list[dict], list[str]]:
    """Batch resolve chunk_ids to spans.

    Returns (spans, errors). Each span has keys: document_id, char_start, char_end,
    _preview, _chunk_id. The _-prefixed keys are metadata (to be stripped before
    persistence, existing convention in vector_explorer).
    """
    if not chunk_ids:
        return [], []

    cache_dir = settings.PARSE_CACHE_DIR
    manifest_path = settings.MANIFEST_DB

    conn = sqlite3.connect(str(manifest_path))
    conn.row_factory = sqlite3.Row

    spans = []
    errors = []

    try:
        for chunk_id in chunk_ids:
            match = re.match(r"^(.+)_(\d+)$", chunk_id)
            if not match:
                errors.append(f"`{chunk_id}` — format tanınmadı (beklenen: doc_id_N)")
                continue

            doc_id = match.group(1)
            chunk_idx = int(match.group(2))

            row = conn.execute(
                "SELECT document_source FROM document_manifest WHERE document_id = ?",
                (doc_id,),
            ).fetchone()

            if not row or not row["document_source"]:
                errors.append(f"`{chunk_id}` — manifest'te bulunamadı")
                continue

            src_path = row["document_source"]

            if not Path(src_path).exists():
                errors.append(f"`{chunk_id}` — kaynak dosya bulunamadı: {src_path}")
                continue

            file_hash = _file_hash(src_path)
            ocr_engine = settings.OCR_ENGINE
            ocr_tag = ""
            ocr_base = f"{file_hash}_{ocr_engine}{ocr_tag}"
            chunk_cache_key = _atompack_chunk_cache_key(spec, ocr_base)
            cache_file = cache_dir / f"{chunk_cache_key}.json"

            if not cache_file.exists():
                errors.append(
                    f"`{chunk_id}` — cache bulunamadı "
                    f"(koleksiyon: {spec.name}, max_tok={spec.max_chunk_tokens}, min_tok={spec.min_chunk_tokens})"
                )
                continue

            try:
                data = json.loads(cache_file.read_text(encoding="utf-8"))
                chunks = data.get("chunks", [])
                if chunk_idx >= len(chunks):
                    errors.append(f"`{chunk_id}` — cache'de {len(chunks)} chunk var, index {chunk_idx} dışarıda")
                    continue
                chunk = chunks[chunk_idx]
                span = chunk.get("span")
                if not span or len(span) < 2:
                    errors.append(f"`{chunk_id}` — span bilgisi yok")
                    continue
                spans.append({
                    "document_id": doc_id,
                    "char_start": span[0],
                    "char_end": span[1],
                    "_preview": chunk["text"][:120],
                    "_text": chunk["text"],
                    "_chunk_id": chunk_id,
                })
            except Exception as e:
                errors.append(f"`{chunk_id}` — cache okuma hatası: {e}")

    finally:
        conn.close()

    return spans, errors
