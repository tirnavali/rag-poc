"""ChromaDB client helpers.

All chromadb.* imports are confined to this module. To swap the vector DB
backend (Qdrant, pgvector, etc.) rewrite only this file — callers use the
helpers below and stay DB-agnostic.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings


def open_collection(path: Path | str, name: str):
    """Open an existing ChromaDB collection at an absolute path."""
    client = chromadb.PersistentClient(
        path=str(path),
        settings=Settings(anonymized_telemetry=False),
    )
    return client, client.get_collection(name=name)


def open_or_create_collection(path: Path | str, name: str):
    """Open or create a ChromaDB collection with cosine similarity."""
    client = chromadb.PersistentClient(path=str(path))
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
