"""Shared vector search primitive — single Chroma+rerank path for production + benchmark.

No date filtering, no post-processing, no RetrievalResult shaping.
That logic lives in the wrappers (VectorRetriever for production, RetrievalBenchmark for evals).

All chromadb.* calls go through src/common/chroma helpers; DB swap requires only that file.
"""
from __future__ import annotations

import math
import time
from typing import TYPE_CHECKING, Optional

from src.common.chroma import open_collection, query_collection
from src.common.embeddings import build_embedder_for_spec

if TYPE_CHECKING:
    from src.config.collections import CollectionSpec


class VectorSearch:
    """Unified vector search engine. Wraps ChromaDB + optional cross-encoder rerank."""

    def __init__(self, spec: "CollectionSpec"):
        self.spec = spec
        self.embedder = build_embedder_for_spec(spec)
        _, self.collection = open_collection(spec.db_path, spec.name)

    def search(
        self,
        query: str,
        *,
        top_k: int,
        fetch_k: Optional[int] = None,
        where_filter: Optional[dict] = None,
        reranker=None,
        timings: Optional[dict] = None,
    ) -> list[dict]:
        """Search and optionally rerank. Returns up to top_k results.

        Args:
            query: user search text
            top_k: final result count
            fetch_k: candidates to pull before reranking (default: max(top_k*4, 20))
            where_filter: Chroma where dict (e.g. year filter) — passed through unchanged
            reranker: optional CrossEncoderReranker instance; if None, sort by distance
            timings: optional dict; if provided, filled with per-phase wall-clock in ms
                (embed_ms, ann_ms, rerank_ms). When None there is zero overhead and the
                production path is unchanged.

        Returns:
            List of dicts: {id, doc, meta, dist, rerank_score or None}
        """
        n = fetch_k or max(top_k * 4, 20)

        _t = time.perf_counter()
        q_vec = self.embedder.embed_query(query)
        if timings is not None:
            timings["embed_ms"] = (time.perf_counter() - _t) * 1000.0

        _t = time.perf_counter()
        res = query_collection(
            self.collection,
            q_vec,
            n_results=n,
            where_filter=where_filter,
        )
        if timings is not None:
            timings["ann_ms"] = (time.perf_counter() - _t) * 1000.0

        # Unpack Chroma result shape: ids/docs/metas/distances are all [[...]]
        candidates: list[tuple[str, str, dict, float]] = []
        for i, cid in enumerate(res.get("ids", [[]])[0]):
            doc = res["documents"][0][i] if res.get("documents") else ""
            meta = dict(res["metadatas"][0][i] or {}) if res.get("metadatas") else {}
            dist = res["distances"][0][i] if res.get("distances") else 0.0
            candidates.append((cid, doc, meta, dist))

        # Optional reranking
        _t = time.perf_counter()
        id_to_rerank_score: dict[str, float] = {}
        if reranker is not None and candidates:
            pairs = [(cid, doc) for cid, doc, _, _ in candidates]
            reranked = reranker.rerank(query, pairs, top_n=len(candidates))
            id_to_rerank_score = {cid: score for cid, score in reranked}
        if timings is not None:
            timings["rerank_ms"] = (time.perf_counter() - _t) * 1000.0

        # Build result list
        results: list[dict] = []
        if id_to_rerank_score:
            # Sort by rerank score desc; candidates not scored by reranker go last
            ordered = sorted(
                candidates,
                key=lambda c: id_to_rerank_score.get(c[0], float("-inf")),
                reverse=True,
            )
            for cid, doc, meta, dist in ordered[:top_k]:
                raw = id_to_rerank_score.get(cid, float("-inf"))
                sigmoid = 1.0 / (1.0 + math.exp(-raw))
                results.append({"id": cid, "doc": doc, "meta": meta, "dist": dist, "rerank_score": sigmoid})
        else:
            for cid, doc, meta, dist in candidates[:top_k]:
                results.append({"id": cid, "doc": doc, "meta": meta, "dist": dist, "rerank_score": None})

        return results
