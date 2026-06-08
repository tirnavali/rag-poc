"""Benchmark late-chunking retrieval quality at different context window sizes.

Usage:
    USE_LOCAL_LATE_CHUNKING=1 python scripts/bench_embedding_context.py \\
        --doc path/to/long_minutes.pdf \\
        --max-tokens 4096 8192 \\
        --queries tests/fixtures/eval_queries_tr.json

For 32k, switch model to jinaai/jina-embeddings-v4 (context window
is now model-specific via src/config/collections.py).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import List, Tuple

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.trainer.ingestion.embedder import LocalLateChunkingEmbedder
from src.config import settings
from src.config.collections import MODEL_SPECS


def _cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    return dot / (norm_a * norm_b + 1e-9)


def _load_queries(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    # Support {"queries": [...]} wrapper
    return data.get("queries", [])


def _parse_doc(doc_path: str) -> Tuple[str, List[Tuple[int, int]]]:
    """Return (full_text, spans) from a document.

    Tries Docling first (for PDF/DOCX); falls back to plain-text read.
    Spans are 1500-char non-overlapping windows over full_text.
    """
    suffix = Path(doc_path).suffix.lower()
    full_text = ""

    if suffix in (".pdf", ".docx"):
        try:
            from src.common.parsing.docling_manager import DoclingManager
            dm = DoclingManager()
            full_text, chunks = dm.convert_and_pack(
                doc_path,
                min_chars=settings.MINUTES_MIN_CHUNK_CHARS,
                max_chars=settings.MINUTES_TARGET_CHUNK_CHARS,
            )
            if chunks:
                spans = [c["span"] for c in chunks]
                return full_text, spans
        except Exception as e:
            print(f"[WARN] Docling parse failed ({e}), falling back to text read.")

    # Plain text fallback
    with open(doc_path, "r", encoding="utf-8", errors="replace") as f:
        full_text = f.read()

    chunk_size = settings.MINUTES_TARGET_CHUNK_CHARS
    spans = []
    pos = 0
    while pos < len(full_text):
        end = min(pos + chunk_size, len(full_text))
        spans.append((pos, end))
        pos = end

    return full_text, spans


def _precision_at_k(ranked: List[int], relevant: set, k: int) -> float:
    hits = sum(1 for r in ranked[:k] if r in relevant)
    return hits / k


def _mrr(ranked: List[int], relevant: set) -> float:
    for rank, r in enumerate(ranked, start=1):
        if r in relevant:
            return 1.0 / rank
    return 0.0


def _ndcg_at_k(ranked: List[int], relevant: set, k: int) -> float:
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, r in enumerate(ranked[:k], start=1)
        if r in relevant
    )
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(relevant), k) + 1))
    return dcg / ideal if ideal > 0 else 0.0


def run_bench(
    doc_path: str,
    queries_path: str,
    max_tokens_list: List[int],
    model_name: str = settings.JINA_LOCAL_MODEL,
) -> None:
    print(f"Loading model: {model_name}")
    _mspec = MODEL_SPECS.get(model_name, {})
    embedder = LocalLateChunkingEmbedder(
        model_name=model_name,
        max_context_tokens=_mspec.get("max_context_tokens", 8192),
        overlap_tokens=_mspec.get("overlap_tokens", 128),
    )

    print(f"Parsing document: {doc_path}")
    full_text, spans = _parse_doc(doc_path)
    n_chunks = len(spans)
    print(f"  {len(full_text):,} chars, {n_chunks} spans")

    queries = _load_queries(queries_path)
    if not queries:
        print("No queries found — exiting.")
        return
    print(f"  {len(queries)} queries loaded")

    header = f"{'max_tokens':>10} | {'MRR':>6} | {'NDCG@10':>7} | {'P@5':>6} | {'avg_cos':>8}"
    sep = "-" * len(header)
    print()
    print(header)
    print(sep)

    for max_tok in max_tokens_list:
        # Embed all chunks with this window size
        chunk_vecs = embedder.embed_with_late_chunking_windowed(
            full_text, spans, max_tokens=max_tok
        )

        mrr_scores, ndcg_scores, p5_scores, cos_scores = [], [], [], []

        for q in queries:
            query_text = q.get("query") or q.get("soru") or ""
            if not query_text:
                continue

            # Relevant span indices — use "relevant_chunks" field if present,
            # otherwise treat top-1 cosine as relevant (self-supervised proxy)
            relevant_set: set[int] = set(q.get("relevant_chunks", []))

            q_vec = embedder.embed_query(query_text)
            sims = [(_cosine(q_vec, cv), i) for i, cv in enumerate(chunk_vecs)]
            sims.sort(key=lambda x: -x[0])
            ranked = [i for _, i in sims]

            # Self-supervised fallback: if no ground truth, top result = relevant
            if not relevant_set:
                relevant_set = {ranked[0]}

            avg_cos = sum(s for s, _ in sims[:5]) / min(5, len(sims))
            cos_scores.append(avg_cos)
            mrr_scores.append(_mrr(ranked, relevant_set))
            ndcg_scores.append(_ndcg_at_k(ranked, relevant_set, 10))
            p5_scores.append(_precision_at_k(ranked, relevant_set, 5))

        def avg(lst):
            return sum(lst) / len(lst) if lst else 0.0

        print(
            f"{max_tok:>10} | {avg(mrr_scores):>6.3f} | {avg(ndcg_scores):>7.3f} | "
            f"{avg(p5_scores):>6.3f} | {avg(cos_scores):>8.4f}"
        )

    print(sep)
    print()
    print("Note: without ground-truth 'relevant_chunks' in the query file,")
    print("MRR/NDCG/P@5 use self-supervised top-1 proxy (avg_cos is still meaningful).")


def main():
    parser = argparse.ArgumentParser(description="Benchmark late-chunking context window sizes")
    parser.add_argument("--doc", required=True, help="Path to long document (PDF, DOCX, or TXT)")
    parser.add_argument(
        "--max-tokens", type=int, nargs="+", default=[4096, 8192],
        metavar="N", dest="max_tokens",
        help="Context window sizes to compare (default: 4096 8192)",
    )
    parser.add_argument(
        "--queries", default="tests/fixtures/eval_queries_tr.json",
        help="Path to golden queries JSON",
    )
    parser.add_argument(
        "--model", default=settings.JINA_LOCAL_MODEL,
        help="Jina model name (default: settings.JINA_LOCAL_MODEL)",
    )
    args = parser.parse_args()

    run_bench(
        doc_path=args.doc,
        queries_path=args.queries,
        max_tokens_list=args.max_tokens,
        model_name=args.model,
    )


if __name__ == "__main__":
    main()
