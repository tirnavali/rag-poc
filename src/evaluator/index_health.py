"""Index health checks: FTS↔SQLite parity, orphan Chroma IDs, model availability."""
from __future__ import annotations

import os
import sqlite3
import subprocess

from src.config import settings
from src.common.chroma import open_collection


def check_press_fts_parity() -> dict:
    """Verify that kupurler and kupurler_fts have the same row count."""
    try:
        conn = sqlite3.connect(str(settings.PRESS_SQLITE))
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM kupurler")
        main_count = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM kupurler_fts")
        fts_count = cur.fetchone()[0]
        conn.close()
        ok = main_count == fts_count
        return {"check": "press_fts_parity", "ok": ok, "main": main_count, "fts": fts_count}
    except Exception as e:
        return {"check": "press_fts_parity", "ok": False, "error": str(e)}


def check_chroma_sqlite_parity(source_type: str) -> dict:
    """Verify every ChromaDB chunk ID resolves to an existing SQLite row."""
    if source_type == "gazete":
        chroma_path = settings.PRESS_CHROMA
        collection_name = settings.PRESS_COLLECTION
        sqlite_path = settings.PRESS_SQLITE
        table = "kupurler"
        id_col = "KAYIT_NO"
    else:
        chroma_path = settings.MINUTES_CHROMA
        collection_name = settings.MINUTES_COLLECTION
        sqlite_path = settings.MINUTES_SQLITE
        table = "tutanak_minutes"
        id_col = "id"

    orphans = 0
    total = 0
    try:
        _, col = open_collection(chroma_path, collection_name)
        conn = sqlite3.connect(str(sqlite_path))
        cur = conn.cursor()
        # Sample check: get all IDs from Chroma (may be large — chunk by 1000)
        result = col.get(include=[])
        for cid in result.get("ids", []):
            total += 1
            try:
                row_id = int(cid.split("_")[0])
            except ValueError:
                orphans += 1
                continue
            cur.execute(f"SELECT 1 FROM {table} WHERE {id_col} = ?", (row_id,))
            if cur.fetchone() is None:
                orphans += 1
        conn.close()
        return {"check": f"{source_type}_chroma_parity", "ok": orphans == 0, "total": total, "orphans": orphans}
    except Exception as e:
        return {"check": f"{source_type}_chroma_parity", "ok": False, "error": str(e)}


def check_ollama_model(model: str = settings.EMBED_MODEL) -> dict:
    """Verify the Ollama embedding model is available locally."""
    try:
        env = os.environ.copy()
        env["OLLAMA_HOST"] = settings.OLLAMA_HOST
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=10, env=env)
        available = model.split(":")[0] in result.stdout
        return {"check": "ollama_model", "ok": available, "model": model}
    except Exception as e:
        return {"check": "ollama_model", "ok": False, "error": str(e)}


def run_all_checks() -> list[dict]:
    checks = [
        check_press_fts_parity(),
        check_chroma_sqlite_parity("gazete"),
        check_ollama_model(settings.EMBED_MODEL),
        check_ollama_model(settings.LLM_MODEL),
    ]
    # Minutes check only if DB exists
    if settings.MINUTES_SQLITE.exists():
        checks.append(check_chroma_sqlite_parity("minutes"))
    return checks
