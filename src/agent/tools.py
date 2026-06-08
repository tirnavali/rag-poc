"""Tool implementations for the Planning Agent.

Each tool wraps an existing pipeline component and logs trace events.
Collections and model specs are loaded from models.yaml via collections.py.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from src.common.chroma import where_year_filter
from src.common.dates import extract_dates
from src.common.text import extract_relevant_windows
from src.config import settings
from src.config.collections import COLLECTIONS, CollectionSpec
from src.config.document_types import format_prefix, normalize_metadata
from src.retriever.context import build_context
from src.retriever.vector_search import VectorSearch

if TYPE_CHECKING:
    from src.agent.tracer import PipelineTracer
    from src.common.llm_client_pool import LLMClientPool
    from src.config.pipeline_loader import PipelineConfig
    from src.retriever.reranker import CrossEncoderReranker


class SearchTool:
    """Executes a search against a collection with given query and filters.

    Collection specs (including per-collection embedder) are loaded from
    models.yaml via src/config/collections.py.
    """

    def __init__(self, config: PipelineConfig, client_pool: LLMClientPool) -> None:
        self._config = config
        self._pool = client_pool
        self._search_cache: dict[str, tuple[VectorSearch, CollectionSpec]] = {}
        self._reranker: Optional[CrossEncoderReranker] = None
        if config.retrieval.reranker_enabled:
            from src.retriever.reranker import CrossEncoderReranker
            self._reranker = CrossEncoderReranker(config.retrieval.reranker_model)

    def _get_search(self, collection_key: str) -> tuple[VectorSearch, CollectionSpec]:
        """Get or create a VectorSearch for the given collection key.

        The collection key (e.g. 'tbmm_tutanaklar_nomic_v2') maps to a CollectionSpec
        loaded from models.yaml, which includes the per-collection embedder.
        """
        if collection_key not in self._search_cache:
            spec = COLLECTIONS[collection_key]
            self._search_cache[collection_key] = (VectorSearch(spec), spec)
        return self._search_cache[collection_key]

    def search(
        self,
        collection_key: str,
        query_text: str,
        filters: dict | None = None,
        top_k: int = 5,
    ) -> dict:
        """Search a collection and return formatted results.

        Args:
            collection_key: collection key as defined in models.yaml
            query_text: search query text
            filters: optional filter dict (year, author, etc.)
            top_k: number of results

        Returns:
            Dict with documents, metadatas, distances lists.
        """
        where_filter = filters
        if where_filter is None:
            parsed_dates = extract_dates(query_text)
            years = parsed_dates.get("years", [])
            exact_dates = parsed_dates.get("exact_dates", [])
            year_from_exact = [int(d[:4]) for d in exact_dates if d]
            all_years = list({int(y) for y in years} | set(year_from_exact))
            where_filter = where_year_filter(all_years)

        search, spec = self._get_search(collection_key)

        raw = search.search(
            query_text,
            top_k=top_k,
            fetch_k=max(top_k * 4, 20),
            where_filter=where_filter,
            reranker=self._reranker,
        )

        final_docs: list[str] = []
        final_metas: list[dict] = []
        final_dists: list[float] = []

        for r in raw:
            doc_text = extract_relevant_windows(r["doc"], query_text)
            meta = normalize_metadata(r["meta"])
            meta["chunk_id"] = r["id"]
            meta["_source_collection"] = collection_key

            prefix = format_prefix(meta, spec.doc_type)
            final_docs.append(prefix + doc_text)
            final_metas.append(meta)

            if r["rerank_score"] is not None:
                final_dists.append(1.0 - r["rerank_score"])
            else:
                final_dists.append(r["dist"])

        return {
            "documents": final_docs,
            "metadatas": final_metas,
            "distances": final_dists,
        }


class ContextBuilderTool:
    """Builds context string from retrieval results."""

    def __init__(self, config: PipelineConfig) -> None:
        self._config = config

    def build(
        self,
        all_results: list[dict],
    ) -> tuple[str, list[dict]]:
        """Merge results from multiple collections and build context.

        Args:
            all_results: list of result dicts from SearchTool.search()

        Returns:
            (context_text, sources_list)
        """
        merged_docs = []
        merged_metas = []
        merged_dists = []

        for result in all_results:
            merged_docs.extend(result.get("documents", []))
            merged_metas.extend(result.get("metadatas", []))
            merged_dists.extend(result.get("distances", []))

        threshold = self._config.retrieval.distance_threshold
        max_chars = self._config.retrieval.context_max_chars
        total_max_chars = self._config.retrieval.context_total_max_chars

        filtered_docs = []
        filtered_metas = []
        filtered_dists = []
        for doc, meta, dist in zip(merged_docs, merged_metas, merged_dists):
            if dist <= threshold:
                filtered_docs.append(doc)
                filtered_metas.append(meta)
                filtered_dists.append(dist)

        fake_result = {
            "documents": [filtered_docs],
            "metadatas": [filtered_metas],
            "distances": [filtered_dists],
        }

        ctx = build_context(
            fake_result,
            max_chars=max_chars,
            total_max_chars=total_max_chars,
            distance_threshold=threshold,
        )

        return ctx, filtered_metas


class AnswerTool:
    """Calls the answering agent LLM to generate a response."""

    def __init__(self, client_pool: LLMClientPool, config: PipelineConfig) -> None:
        self._pool = client_pool
        self._config = config

    def generate(
        self,
        query: str,
        context: str,
        *,
        mufettis_mode: bool = False,
    ) -> tuple[str, str]:
        """Generate answer via the answering agent LLM.

        Returns:
            (thinking, content) tuple.
        """
        ans_cfg = self._config.answering
        block_name = ans_cfg.block
        model_key = ans_cfg.model_key

        client = self._pool.get_client(block_name)
        model = self._pool.get_model_for_block(block_name, model_key)

        from src.generator.prompts import MUFETTIS_SYS_PROMPT, SYS_PROMPT

        user_msg = f"BAĞLAM:\n{context}\n\nSORU: {query}"
        sys_prompt = MUFETTIS_SYS_PROMPT if mufettis_mode else SYS_PROMPT

        temperature = ans_cfg.temperature
        num_predict = min(
            ans_cfg.num_predict,
            self._config.blocks[block_name].max_num_predict,
        )

        options = {
            "temperature": temperature,
            "num_predict": num_predict,
            "num_ctx": self._config.blocks[block_name].max_num_ctx,
        }

        thinking = ""
        content = ""

        stream = client.chat(
            model=model,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_msg},
            ],
            options=options,
            stream=True,
            think=ans_cfg.think if ans_cfg.think is not None else False,
        )

        for chunk in stream:
            if hasattr(chunk.message, "thinking") and chunk.message.thinking:
                thinking += chunk.message.thinking
            if hasattr(chunk.message, "content") and chunk.message.content:
                content += chunk.message.content

        if not content.strip():
            content = "Arşivde bu soruyu yanıtlayacak yeterli bilgi bulunamadı."

        return thinking, content
