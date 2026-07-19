"""Tool implementations for the Planning Agent.

Each tool wraps an existing pipeline component and logs trace events.
Collections and model specs are loaded from models.yaml via collections.py.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from src.common.chroma import get_by_ids, where_year_filter
from src.common.dates import extract_dates
from src.common.text import expand_parliamentary_synonyms, extract_relevant_windows
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

    @property
    def reranker_enabled(self) -> bool:
        return self._reranker is not None

    def rerank(self, query: str, chunks: list[tuple[str, str]], top_n: int) -> dict[str, float]:
        """Rerank (chunk_id, text) pairs in a single pass; returns {chunk_id: score}."""
        if self._reranker is None or not chunks:
            return {}
        reranked = self._reranker.rerank(query, chunks, top_n=top_n)
        return dict(reranked)

    def search(
        self,
        collection_key: str,
        query_text: str,
        filters: dict | None = None,
        top_k: int = 10,
        apply_reranker: bool = True,
    ) -> dict:
        """Search a collection and return formatted results.

        Args:
            collection_key: collection key as defined in models.yaml
            query_text: search query text
            filters: optional filter dict (year, author, etc.)
            top_k: number of results
            apply_reranker: set False to skip reranking and get raw ANN-ranked
                candidates — use when multiple query variants for the same
                collection will be merged before a single downstream rerank
                pass, so the cross-encoder only scores the merged pool once.

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

        # Colloquial parliamentary jargon (e.g. "kadük") rarely matches the corpus's
        # own official phrasing in embedding space — expand deterministically before
        # vector search. Date parsing above stays on the raw text; expansion only
        # affects the search query + window-highlighting (both benefit from also
        # matching the official term's spans within retrieved docs).
        expanded_query = expand_parliamentary_synonyms(query_text)

        search, spec = self._get_search(collection_key)

        raw = search.search(
            expanded_query,
            top_k=top_k,
            fetch_k=max(top_k * 4, 20),
            where_filter=where_filter,
            reranker=self._reranker if apply_reranker else None,
        )

        final_docs: list[str] = []
        final_metas: list[dict] = []
        final_dists: list[float] = []

        for r in raw:
            doc_text = extract_relevant_windows(r["doc"], expanded_query)
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

    def fetch_neighbors(
        self,
        collection_key: str,
        document_id: str,
        anchor_index: int,
        radius: int,
        max_total: Optional[int] = None,
    ) -> dict:
        """Fetch a document-internal chunk-order window [anchor-radius, anchor+radius]
        (excluding the anchor itself) by id-construction — NO ANN / embedding.

        Chunk ids are ``{document_id}_{i}`` with ``i`` the 0-based reading order
        (pipeline.py), so neighbors are just the anchor's index ± radius. Ids past the
        document end are silently dropped by Chroma's get(). Returns the
        ``{documents, metadatas, distances}`` shape that ``_dict_to_chunks`` consumes;
        distances are 0.0 (these are deterministic fetches, not ranked hits). metadatas
        are REQUIRED — the consumer reads chunk_id/document_id/doc_type from them.
        """
        lo = max(0, anchor_index - radius)
        hi = anchor_index + radius
        ids = [f"{document_id}_{i}" for i in range(lo, hi + 1) if i != anchor_index]
        if max_total is not None and len(ids) > max_total:
            # Keep the ids closest to the anchor (both directions) within budget.
            ids.sort(key=lambda cid: abs(int(cid.rsplit("_", 1)[1]) - anchor_index))
            ids = ids[:max_total]
        if not ids:
            return {"documents": [], "metadatas": [], "distances": []}

        search, spec = self._get_search(collection_key)
        res = get_by_ids(search.collection, ids, include=("documents", "metadatas"))

        docs = res.get("documents") or []
        metas = res.get("metadatas") or []
        got_ids = res.get("ids") or []
        final_docs: list[str] = []
        final_metas: list[dict] = []
        for cid, doc_text, raw_meta in zip(got_ids, docs, metas):
            # Mirror search()'s formatting but WITHOUT extract_relevant_windows — we want
            # the neighbor's full body, not a query-highlighted window.
            meta = normalize_metadata(raw_meta or {})
            meta["chunk_id"] = cid
            meta["_source_collection"] = collection_key
            prefix = format_prefix(meta, spec.doc_type)
            final_docs.append(prefix + (doc_text or ""))
            final_metas.append(meta)
        return {
            "documents": final_docs,
            "metadatas": final_metas,
            "distances": [0.0] * len(final_docs),
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

    # Query types that need best-effort SYNTHESIS from partial/tangential evidence
    # rather than the strict "refuse if not a direct hit" behavior. Enumeration and
    # summary/comparison/reasoning queries span many chunks that rarely match the
    # question verbatim, so the strict SYS_PROMPT made the model return empty.
    SYNTHESIS_QUERY_TYPES = {"comprehensive", "summary", "comparison", "reasoning"}

    @staticmethod
    def _select_system_prompt(mufettis_mode: bool, query_type: str | None):
        """Pick the answering system prompt.

        müfettiş → deep-research report; synthesis-needing query types → synthesis
        (never-empty) prompt; everything else (fact/policy) → strict prompt that
        honestly refuses when the archive has no relevant evidence.
        """
        from src.generator.prompts import MUFETTIS_SYS_PROMPT, SYNTHESIS_SYS_PROMPT, SYS_PROMPT
        if mufettis_mode:
            return MUFETTIS_SYS_PROMPT
        if query_type in AnswerTool.SYNTHESIS_QUERY_TYPES:
            return SYNTHESIS_SYS_PROMPT
        return SYS_PROMPT

    def generate(
        self,
        query: str,
        context: str,
        *,
        mufettis_mode: bool = False,
        chat_history: list | None = None,
        stream_callback: callable = None,
        query_type: str | None = None,
        answer_directive: str | None = None,
    ) -> tuple[str, str]:
        """Generate answer via the answering agent LLM.

        Args:
            stream_callback: if given, called with ``{"type": "content"|"thinking",
                "content": <delta>}`` for each token as it arrives, so the UI can
                render the answer progressively instead of all at once.
            query_type: planner query_type; selects the system prompt (synthesis vs
                strict) so comprehensive/summary answers aren't dropped as empty.
            answer_directive: optional playbook directive (research_strategies.md)
                appended on top of the selected system prompt; no-op when empty.

        Returns:
            (thinking, content) tuple (full accumulated text).
        """
        ans_cfg = self._config.answering
        block_name = ans_cfg.block
        model_key = ans_cfg.model_key

        client = self._pool.get_client(block_name)
        model = self._pool.get_model_for_block(block_name, model_key)

        user_msg = f"BAĞLAM:\n{context}\n\nSORU: {query}"
        sys_prompt = self._select_system_prompt(mufettis_mode, query_type)
        if answer_directive:
            sys_prompt = f"{sys_prompt}\n\nEK YÖNERGE:\n{answer_directive}"

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

        history_messages = []
        for m in (chat_history or []):
            msg_content = m["content"]
            if m["role"] == "assistant":
                msg_content = msg_content[:1500]
            history_messages.append({"role": m["role"], "content": msg_content})

        stream = client.chat(
            model=model,
            messages=[
                {"role": "system", "content": sys_prompt},
                *history_messages,
                {"role": "user", "content": user_msg},
            ],
            options=options,
            stream=True,
            think=ans_cfg.think if ans_cfg.think is not None else False,
        )

        def _emit(kind: str, delta: str) -> None:
            if stream_callback is not None and delta:
                try:
                    stream_callback({"type": kind, "content": delta})
                except Exception:
                    pass

        for chunk in stream:
            if hasattr(chunk.message, "thinking") and chunk.message.thinking:
                thinking += chunk.message.thinking
                _emit("thinking", chunk.message.thinking)
            if hasattr(chunk.message, "content") and chunk.message.content:
                content += chunk.message.content
                _emit("content", chunk.message.content)

        if not content.strip():
            content = "Arşivde bu soruyu yanıtlayacak yeterli bilgi bulunamadı."
            _emit("content", content)  # nothing streamed → emit the fallback once

        return thinking, content
