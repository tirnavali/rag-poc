"""Production retriever: wraps VectorSearch with date filtering, post-processing, and RetrievalResult shape.

Renamed from hybrid.py to reflect reality: single-collection mode dropped BM25, only ANN+rerank remain.
"""
from __future__ import annotations

from typing import Optional

from src.common.chroma import where_year_filter
from src.common.dates import extract_dates
from src.common.protocols import RetrievalResult
from src.common.text import extract_relevant_windows
from src.config import settings
from src.config.collections import CollectionSpec
from src.config.document_types import format_prefix
from src.retriever.vector_search import VectorSearch


class VectorRetriever:
    """Production retriever: date-aware, post-processed, RetrievalResult shape."""

    def __init__(self, spec: CollectionSpec) -> None:
        self.spec = spec
        self.search = VectorSearch(spec)

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = settings.RETRIEVE_TOP_K,
        fetch_k: int = settings.RETRIEVE_FETCH_K,
        mufettis_mode: bool = False,
        where_filter: Optional[dict] = None,
    ) -> RetrievalResult:
        """Retrieve and post-process. Applies date filtering + metadata prefix + window cropping.

        Args:
            query: user search text (may contain date hints)
            top_k: final result count
            fetch_k: candidates before reranking
            mufettis_mode: use deep research settings (40 results, 150 fetch)
            where_filter: pre-extracted filter dictionary (bypasses automatic date extraction)

        Returns:
            RetrievalResult TypedDict with documents/metadatas/distances in list-of-lists shape.
        """
        if mufettis_mode:
            top_k = settings.MUFETTIS_TOP_K
            fetch_k = settings.MUFETTIS_FETCH_K

        # Parse dates from query and build where filter if not provided
        if where_filter is None:
            parsed_dates = extract_dates(query)
            years = parsed_dates.get("years", [])
            exact_dates = parsed_dates.get("exact_dates", [])
            year_from_exact = [int(d[:4]) for d in exact_dates if d]
            all_years = list(set([int(y) for y in years] + year_from_exact))
            where_filter = where_year_filter(all_years)
        else:
            # Try to extract year from where_filter to populate parsed_dates for compatibility
            years_found = []
            def _find_years(d):
                if isinstance(d, dict):
                    for k, v in d.items():
                        if k == "year" and isinstance(v, dict) and "$eq" in v:
                            years_found.append(str(v["$eq"]))
                        elif k in ("$and", "$or") and isinstance(v, list):
                            for item in v:
                                _find_years(item)
            _find_years(where_filter)
            parsed_dates = {"years": years_found, "exact_dates": []}

        # Build reranker if enabled
        reranker = None
        if settings.USE_RERANKER:
            from src.retriever.reranker import CrossEncoderReranker
            reranker = CrossEncoderReranker()

        # Search
        raw = self.search.search(
            query,
            top_k=top_k,
            fetch_k=fetch_k,
            where_filter=where_filter,
            reranker=reranker,
        )

        # Post-process: crop + add metadata prefix
        final_docs: list[str] = []
        final_metas: list[dict] = []
        final_dists: list[float] = []

        for r in raw:
            doc_text = extract_relevant_windows(r["doc"], query)
            meta = r["meta"]
            meta["chunk_id"] = r["id"]

            prefix = format_prefix(meta, self.spec.doc_type)

            final_docs.append(prefix + doc_text)
            final_metas.append(meta)

            if r["rerank_score"] is not None:
                final_dists.append(1.0 - r["rerank_score"])
            else:
                final_dists.append(r["dist"])

        is_minutes = any(kw in query.lower() for kw in settings.MINUTES_KEYWORDS)

        return RetrievalResult(
            documents=[final_docs],
            metadatas=[final_metas],
            distances=[final_dists],
            is_minutes=is_minutes,
            parsed_dates=parsed_dates,
            expanded_query=None,
            fallback_level=None,
        )

    def inspect_record(self, source_db: str, chunk_id: str) -> Optional[dict]:
        """Fetch a single chunk from ChromaDB by its ID."""
        try:
            res = self.search.collection.get(
                ids=[chunk_id],
                include=["documents", "metadatas"],
            )
            if not res["ids"]:
                return None
            meta = dict(res["metadatas"][0] or {})
            meta["content"] = res["documents"][0]
            meta["chunk_id"] = chunk_id
            return meta
        except Exception as e:
            print(f"Chroma inspect error: {e}")
            return None
