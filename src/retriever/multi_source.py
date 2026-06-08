"""Multi-source retriever with RRF fusion.

Enables query agents to fan-out across document types (gazete + tutanak + onerge)
and fuse results using Reciprocal Rank Fusion.
"""
from __future__ import annotations

from src.common.protocols import RetrievalResult
from src.config import settings
from src.config.collections import (
    DEFAULT_COLLECTION_FOR_TYPE,
    CollectionSpec,
    get_default_spec,
)
from src.config.document_types import DocumentType
from src.retriever.vector_retriever import VectorRetriever


class MultiSourceRetriever:
    """Fan-out retriever across multiple document types with RRF fusion.

    Each source (gazete, tutanak, onerge) is queried independently using its
    default collection, then results are fused via Reciprocal Rank Fusion.
    """

    def __init__(
        self,
        specs: dict[DocumentType, CollectionSpec] | list[CollectionSpec] | None = None,
    ) -> None:
        """Initialize with collection or document type specs.

        Args:
            specs: Either:
                - Dict[DocumentType, CollectionSpec] (legacy mode, by doc type)
                - List[CollectionSpec] (new multi-collection mode)
                - None (use defaults by document type)
        """
        if specs is None:
            specs = {
                dt: get_default_spec(dt)
                for dt in DEFAULT_COLLECTION_FOR_TYPE.keys()
            }

        if isinstance(specs, list):
            # Multi-collection mode: specs is list of CollectionSpec
            # Use collection name as key for retrievers dict
            self.retrievers: dict[str, VectorRetriever] = {
                spec.name: VectorRetriever(spec) for spec in specs
            }
        else:
            # Legacy mode: specs is dict[DocumentType, CollectionSpec]
            self.retrievers: dict[DocumentType, VectorRetriever] = {
                dt: VectorRetriever(spec) for dt, spec in specs.items()
            }

    @classmethod
    def from_defaults(
        cls,
        doc_types: list[DocumentType] | None = None,
    ) -> MultiSourceRetriever:
        """Build from default collections for given document types.

        Args:
            doc_types: Types to include. None → all defaults (GAZETE, TUTANAK, ONERGE).

        Returns:
            MultiSourceRetriever instance.
        """
        if doc_types is None:
            doc_types = list(DEFAULT_COLLECTION_FOR_TYPE.keys())

        specs = {dt: get_default_spec(dt) for dt in doc_types}
        return cls(specs)

    def retrieve_per_collection(
        self,
        query: str,
        *,
        top_k: int = settings.RETRIEVE_TOP_K,
    ) -> dict[str, RetrievalResult]:
        """Query each collection independently, return top_k per collection.

        No RRF fusion. Results are grouped by collection name.

        Args:
            query: Search text.
            top_k: Results per collection.

        Returns:
            Dict[collection_name, RetrievalResult] — one result per collection.
        """
        results: dict[str, RetrievalResult] = {}
        for name, retriever in self.retrievers.items():
            results[name] = retriever.retrieve(query, top_k=top_k)
        return results

    def retrieve_balanced(
        self,
        query: str,
        *,
        per_collection_k: int | None = None,
    ) -> RetrievalResult:
        """Fetch equal results from each collection, concatenate (no RRF).

        Each collection contributes up to per_collection_k results (or its
        context_weight if per_collection_k is None). Results are grouped
        by collection in iteration order.

        Args:
            query: Search text.
            per_collection_k: Results per collection. None → use spec.context_weight.

        Returns:
            RetrievalResult — all collections' results concatenated.
        """
        all_docs, all_metas, all_dists = [], [], []

        for name, retriever in self.retrievers.items():
            k = per_collection_k if per_collection_k is not None else retriever.spec.context_weight
            result = retriever.retrieve(query, top_k=k)

            doc_type = retriever.spec.doc_type.value

            for doc, meta, dist in zip(
                result["documents"][0],
                result["metadatas"][0],
                result["distances"][0],
            ):
                meta_with_source = {
                    **meta,
                    "collection": name,
                    "doc_type": doc_type,
                }
                all_docs.append(doc)
                all_metas.append(meta_with_source)
                all_dists.append(dist)

        is_minutes = any(kw in query.lower() for kw in settings.MINUTES_KEYWORDS)

        return RetrievalResult(
            documents=[all_docs],
            metadatas=[all_metas],
            distances=[all_dists],
            is_minutes=is_minutes,
            parsed_dates={},
            expanded_query=None,
            fallback_level=None,
        )

    def retrieve(
        self,
        query: str,
        *,
        doc_types: list[DocumentType] | None = None,
        top_k: int = settings.RETRIEVE_TOP_K,
        per_source_k: int = 20,
        mufettis_mode: bool = False,
    ) -> RetrievalResult:
        """Fan-out retrieve from multiple sources and fuse with RRF.

        Args:
            query: Search text.
            doc_types: Types to search. None → all available retrievers.
            top_k: Final fused result count.
            per_source_k: Candidates per source before fusion.
            mufettis_mode: Deep research mode (uses MUFETTIS_* settings).

        Returns:
            RetrievalResult with fused documents.
        """
        if mufettis_mode:
            top_k = settings.MUFETTIS_TOP_K
            per_source_k = settings.MUFETTIS_FETCH_K

        # Determine which sources to query:
        # - If doc_types provided (legacy mode), filter by DocumentType
        # - Otherwise, use all available sources (dict/list keys)
        # NOTE: doc_types parameter works correctly in legacy dict mode (DocumentType keys).
        # In list mode (string keys), the comparison will always be false, so all sources
        # are used. This is backward compatible: list mode ignores doc_types filtering.
        if doc_types:
            active = [k for k in self.retrievers.keys() if k in doc_types]
        else:
            active = list(self.retrievers.keys())

        # Fan-out: retrieve from each source
        per_source: dict[DocumentType | str, RetrievalResult] = {}
        for source_key in active:
            if source_key in self.retrievers:
                per_source[source_key] = self.retrievers[source_key].retrieve(
                    query,
                    top_k=per_source_k,
                    mufettis_mode=False,  # Don't double-apply mufettis settings
                )

        return _rrf_fuse(per_source, query, top_k)


def _rrf_fuse(
    per_source: dict,
    query: str,
    top_k: int,
    k: int = 60,
) -> RetrievalResult:
    """Reciprocal Rank Fusion across multiple sources.

    Score per result: sum(1.0 / (k + rank)) across sources.
    Deduplicates by (document_id, chunk_index).
    Adds "collection" field to metadata for source attribution.

    Args:
        per_source: Dict[source_name, RetrievalResult] from each retriever.
        query: Original query (for keyword detection).
        top_k: Final result count.
        k: RRF k parameter (60 is standard).

    Returns:
        RetrievalResult with fused top_k documents.

    Note:
        When the same chunk appears in multiple sources (deduplication by uid),
        the distance from the last source processing that chunk is kept in records.
        RRF score accumulates across all sources regardless.
    """
    # Collect all (doc, meta, dist) with RRF scores
    scores: dict[str, float] = {}  # uid → RRF score
    records: dict[str, tuple] = {}  # uid → (doc_text, meta, dist)

    for source_name, result in per_source.items():
        docs = result["documents"][0]
        metas = result["metadatas"][0]
        dists = result["distances"][0]

        for rank, (doc, meta, dist) in enumerate(
            zip(docs, metas, dists), start=1
        ):
            # Dedup key: document_id + chunk_index
            uid = f"{meta.get('document_id', '')}_c{meta.get('chunk_index', rank)}"
            scores[uid] = scores.get(uid, 0.0) + 1.0 / (k + rank)

            # Add collection/source attribution to metadata
            meta_with_source = {**meta, "collection": source_name}
            records[uid] = (doc, meta_with_source, dist)

    # Sort by RRF score descending, take top_k
    top = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    final_docs, final_metas, final_dists = [], [], []
    for uid, _score in top:
        doc, meta, dist = records[uid]
        final_docs.append(doc)
        final_metas.append(meta)
        final_dists.append(dist)

    is_minutes = any(kw in query.lower() for kw in settings.MINUTES_KEYWORDS)

    return RetrievalResult(
        documents=[final_docs],
        metadatas=[final_metas],
        distances=[final_dists],
        is_minutes=is_minutes,
        parsed_dates={},  # multi-source: each source handled its own date parsing
        expanded_query=None,
    )
