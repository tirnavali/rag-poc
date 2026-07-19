"""ChromaDB client helpers.

All chromadb.* imports are confined to this module. To swap the vector DB
backend (Qdrant, pgvector, etc.) rewrite only this file — callers use the
helpers below and stay DB-agnostic.
"""
from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import Optional

# Disable ChromaDB's anonymized telemetry before the package is imported. Recent
# chromadb releases ship a telemetry client whose capture() signature drifted,
# spamming "Failed to send telemetry event ... capture() takes 1 positional
# argument but 3 were given" on every client start and query. The per-client
# Settings(anonymized_telemetry=False) below is not honored early enough on some
# versions, so we also set the env var here (all chromadb access funnels through
# this module, imported before any client is built).
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
# Silence posthog telemetry error spam
logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)

import chromadb
from chromadb.config import Settings


def open_collection(path: Path | str, name: str):
    """Open an existing ChromaDB collection at an absolute path."""
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"ChromaDB directory does not exist at: {path_obj}")
    client = chromadb.PersistentClient(
        path=str(path),
        settings=Settings(anonymized_telemetry=False),
    )
    return client, client.get_collection(name=name)


def open_or_create_collection(path: Path | str, name: str):
    """Open or create a ChromaDB collection with cosine similarity."""
    client = chromadb.PersistentClient(
        path=str(path),
        settings=Settings(anonymized_telemetry=False),
    )
    collection = client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
    )
    return client, collection


def query_collection(
    collection,
    query_embedding: list[float],
    n_results: int,
    where_filter: Optional[dict] = None,
    include: tuple[str, ...] = ("documents", "metadatas", "distances"),
) -> dict:
    """Wraps collection.query() with the project's standard arg shape."""
    opts: dict = {
        "query_embeddings": [query_embedding],
        "n_results": n_results,
        "include": list(include),
    }
    if where_filter:
        opts["where"] = where_filter
    return collection.query(**opts)


def where_year_filter(years: list[int], field: str = "year") -> Optional[dict]:
    """Build a Chroma $eq / $or year filter dict. Returns None if years is empty."""
    if not years:
        return None
    conds = [{field: {"$eq": y}} for y in years]
    return conds[0] if len(conds) == 1 else {"$or": conds}


def get_by_ids(
    collection,
    ids: list[str],
    include: tuple[str, ...] = ("documents", "metadatas"),
) -> dict:
    """Fetch chunks by exact id (no ANN / no embedding). Ids absent from the collection
    are silently omitted by Chroma. Returns the FLAT get() shape
    ``{"ids": [...], "documents": [...], "metadatas": [...]}`` — note this differs from
    query()'s nested ``[[...]]`` and does NOT preserve input-id order.

    Confines the ``collection.get`` call to this module per the DB-swap contract.
    """
    if not ids:
        return {"ids": [], "documents": [], "metadatas": []}
    return collection.get(ids=list(ids), include=list(include))
