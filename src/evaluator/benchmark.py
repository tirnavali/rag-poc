"""RetrievalBenchmark — standalone retrieval evaluator for any collection.

Unlike the full `RAGService` evaluation harness, this module targets a *single*
ChromaDB collection with a known embedding model.  It is designed for A/B
experiments (e.g. Jina v3 vs Nomic on the same document set) where you want to
measure pure retrieval quality without the complexity of the full RAG pipeline.

Note: benchmark uses pure vector ANN (no date filtering). Production retrieval
applies year filters from the query; benchmark scores measure ANN only.

Usage:
    from src.config.collections import get_spec
    from src.evaluator.benchmark import RetrievalBenchmark

    bench = RetrievalBenchmark(get_spec("tbmm_minutes_docling_jina_v3"))
    report = bench.evaluate(fixture_queries, k_values=[1, 3, 5, 10])
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Optional

from src.config import settings
from src.config.collections import CollectionSpec
from src.evaluator import retrieval_metrics as rm
from src.evaluator import span_metrics as sm
from src.common.span_resolver import chunk_id_to_span
from src.retriever.vector_search import VectorSearch


class RetrievalBenchmark:
    """Query a single ChromaDB collection and compute retrieval metrics."""

    def __init__(self, spec: CollectionSpec):
        self.spec = spec
        self.search_engine = VectorSearch(spec)
        # Back-compat for tests/inspectors
        self.embedder = self.search_engine.embedder
        self.collection = self.search_engine.collection

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = 5,
        reranker=None,
        fetch_k: Optional[int] = None,
        where_filter: Optional[dict] = None,
    ) -> list[dict]:
        """Vector search (optionally with cross-encoder rerank).

        Returns a list of result dicts with keys:
            id, text, meta, score, raw_score

        Note: pure ANN by default unless where_filter is explicitly provided.
        """
        if fetch_k is None:
            fetch_k = max(top_k * 4, 20)

        raw = self.search_engine.search(
            query,
            top_k=top_k,
            fetch_k=fetch_k,
            where_filter=where_filter,
            reranker=reranker,
        )

        # Reformat for benchmark: text key, score from dist/rerank_score
        results = []
        for r in raw:
            if r["rerank_score"] is not None:
                score = r["rerank_score"]
                raw_score = 0.0  # Reranker doesn't expose raw score here; unused in benchmark
            else:
                score = max(0.0, 1.0 - r["dist"])
                raw_score = None
            results.append(
                {
                    "id": r["id"],
                    "text": r["doc"],
                    "meta": r["meta"],
                    "score": score,
                    "raw_score": raw_score,
                }
            )
        return results

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        queries: list[dict],
        k_values: tuple[int, ...] = (1, 3, 5, 10),
        reranker=None,
        fetch_k: Optional[int] = None,
    ) -> dict[str, Any]:
        """Run retrieval over a fixture and compute precision, recall, MRR, NDCG.

        Args:
            queries: List of query dicts. Expected keys:
                - "id" (str)
                - "query" (str)
                - "relevant_kayit_nos" (list[int])  OR
                - "relevant_chunk_ids" (list[str])
            k_values: Cut-off ranks to evaluate.
            reranker: Optional CrossEncoderReranker instance.
            fetch_k: Number of candidates to fetch from ChromaDB before rerank.

        Returns:
            A report dict with per-query results and aggregate averages.
        """
        results: list[dict] = []

        for item in queries:
            qid = item.get("id", "?")
            query = item["query"]

            retrieved = self.search(
                query, top_k=max(k_values), reranker=reranker, fetch_k=fetch_k
            )

            # Dispatch: page_overlap → span_overlap (special) → hybrid token+chunk (golden) → legacy
            golden_pages = item.get("relevant_pages")
            golden = item.get("gold_evidence_spans") or item.get("relevant_spans")
            if golden_pages:
                # Page-level match path: matches chunk's page(s) metadata against relevant_pages
                relevant_keys = set()
                for p_entry in golden_pages:
                    doc_id = p_entry["document_id"]
                    pages_list = p_entry.get("pages")
                    if pages_list is not None:
                        for p in pages_list:
                            relevant_keys.add(f"{doc_id}#page_{p}")
                    elif "page" in p_entry:
                        relevant_keys.add(f"{doc_id}#page_{p_entry['page']}")

                retrieved_hits = []
                retrieved_keys_by_rank = []
                for r in retrieved:
                    meta = r.get("meta") or {}
                    doc_id = meta.get("document_id")
                    
                    pages = meta.get("pages", [])
                    if not pages and "page" in meta:
                        pages = [meta["page"]]
                    
                    chunk_keys = {f"{doc_id}#page_{p}" for p in pages if p is not None}
                    
                    is_hit = bool(chunk_keys & relevant_keys)
                    retrieved_hits.append(is_hit)
                    retrieved_keys_by_rank.append(chunk_keys)

                metrics = {}
                for k in k_values:
                    hits_at_k = sum(1 for h in retrieved_hits[:k] if h)
                    metrics[f"precision_{k}"] = hits_at_k / k if k > 0 else 0.0
                    
                    union_retrieved_keys = set()
                    for keys in retrieved_keys_by_rank[:k]:
                        union_retrieved_keys.update(keys)
                    found_keys = union_retrieved_keys & relevant_keys
                    metrics[f"recall_{k}"] = len(found_keys) / len(relevant_keys) if relevant_keys else 0.0
                    
                    metrics[f"hit_rate_{k}"] = 1.0 if any(retrieved_hits[:k]) else 0.0

                first_hit_rank = next((rank for rank, h in enumerate(retrieved_hits, 1) if h), None)
                metrics["mrr"] = 1.0 / first_hit_rank if first_hit_rank is not None else 0.0
                
                gains_10 = [1 if h else 0 for h in retrieved_hits[:10]]
                n_rel = min(10, len(relevant_keys))
                ideal_10 = [1] * n_rel + [0] * (10 - n_rel)
                
                def dcg(hits_list):
                    return sum(rel / math.log2(i + 2) for i, rel in enumerate(hits_list))
                
                actual_dcg = dcg(gains_10)
                ideal_dcg = dcg(ideal_10)
                metrics["ndcg_10"] = actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0

                results.append(
                    {
                        "id": qid,
                        "query": query,
                        "metrics": metrics,
                        "retrieved_ids": [r["id"] for r in retrieved[: max(k_values)]],
                        "relevant_ids": sorted(list(relevant_keys)),
                        "matcher": "page_overlap",
                    }
                )
            elif golden:
                # Span-overlap path: char-range based ground truth.
                retrieved_spans = []
                for r in retrieved:
                    s = chunk_id_to_span(r["id"], self.spec)
                    if s:
                        retrieved_spans.append(s)

                metrics: dict[str, float] = {}
                for k in k_values:
                    metrics[f"precision_{k}"] = sm.precision_at_k_span(retrieved_spans, golden, k)
                    metrics[f"recall_{k}"] = sm.recall_at_k_span(retrieved_spans, golden, k)
                    metrics[f"hit_rate_{k}"] = 1.0 if metrics[f"precision_{k}"] > 0 else 0.0
                    metrics[f"evidence_coverage_{k}"] = sm.evidence_coverage_at_k(retrieved_spans, golden, k)
                metrics["mrr"] = sm.mrr_span(retrieved_spans, golden)
                metrics["ndcg_10"] = 0.0  # NDCG over overlapping spans is ill-defined

                results.append(
                    {
                        "id": qid,
                        "query": query,
                        "metrics": metrics,
                        "retrieved_ids": [r["id"] for r in retrieved[: max(k_values)]],
                        "relevant_ids": [
                            f"{g['document_id']}@{g['char_start']}-{g['char_end']}"
                            for g in golden
                        ],
                        "matcher": "span_overlap",
                    }
                )
            else:
                metrics: dict[str, float] = {}
                matcher_parts: list[str] = []

                # Token overlap from excerpts (if present)
                excerpts = item.get("excerpts")
                if excerpts:
                    retrieved_texts = [r["text"] for r in retrieved]
                    for k in k_values:
                        metrics[f"token_recall_{k}"] = sm.token_recall_at_k(retrieved_texts, excerpts, k)
                        metrics[f"token_precision_{k}"] = sm.token_precision_at_k(retrieved_texts, excerpts, k)
                        metrics[f"token_iou_{k}"] = sm.token_iou_at_k(retrieved_texts, excerpts, k)
                    matcher_parts.append("token_overlap")

                # Rank metrics from chunk_ids or kayit_nos (if present)
                relevant_ids: set[str | int] = set()
                id_key = None
                if "relevant_kayit_nos" in item:
                    relevant_ids = set(item["relevant_kayit_nos"])
                    id_key = "kayit_no"
                elif "relevant_chunk_ids" in item:
                    relevant_ids = set(item["relevant_chunk_ids"])
                    id_key = "chunk_id"

                if id_key == "kayit_no":
                    retrieved_ids: list[int | str] = []
                    for r in retrieved:
                        prefix = r["id"].split("_")[0]
                        try:
                            retrieved_ids.append(int(prefix))
                        except ValueError:
                            retrieved_ids.append(prefix)
                else:
                    retrieved_ids = [r["id"] for r in retrieved]

                if id_key:
                    for k in k_values:
                        metrics[f"precision_{k}"] = rm.precision_at_k(retrieved_ids, relevant_ids, k)
                        metrics[f"recall_{k}"] = rm.recall_at_k(retrieved_ids, relevant_ids, k)
                        metrics[f"hit_rate_{k}"] = rm.hit_rate_at_k(retrieved_ids, relevant_ids, k)
                    metrics["mrr"] = rm.mrr(retrieved_ids, relevant_ids)
                    metrics["ndcg_10"] = rm.ndcg_at_k(retrieved_ids, relevant_ids, 10)
                    matcher_parts.append(id_key)
                else:
                    metrics["mrr"] = 0.0
                    metrics["ndcg_10"] = 0.0

                results.append(
                    {
                        "id": qid,
                        "query": query,
                        "metrics": metrics,
                        "retrieved_ids": retrieved_ids[: max(k_values)],
                        "relevant_ids": list(relevant_ids) if relevant_ids else (excerpts or []),
                        "matcher": "+".join(matcher_parts) or "none",
                    }
                )

        # Aggregate
        aggregates = _aggregate(results, k_values)

        return {
            "spec": {
                "name": self.spec.name,
                "embed_model": self.spec.embed_model,
                "doc_type": self.spec.doc_type.value,
            },
            "results": results,
            "aggregate": aggregates,
        }


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _aggregate(results: list[dict], k_values: tuple[int, ...]) -> dict[str, Any]:
    """Compute mean values across all queries for each metric."""
    if not results:
        return {}

    agg: dict[str, list[float]] = {}
    for entry in results:
        for key, val in entry["metrics"].items():
            agg.setdefault(key, []).append(float(val))

    return {key: sum(vals) / len(vals) for key, vals in agg.items()}
