"""RAGService: the top-level façade that wires Retriever + Generator.

This replaces RAGSystem as the single entry-point for chat.py and app.py.
It intentionally exposes the same call signatures that the UI currently uses
so the UI layer can be migrated incrementally.
"""
from __future__ import annotations

from typing import Iterable, Optional

from src.common.protocols import RetrievalResult, StreamChunk
from src.common.tracer import PipelineTracer
from src.config import settings
from src.config.collections import get_spec
from src.generator.ollama_generator import OllamaGenerator
from src.retriever.context import build_context
from src.retriever.vector_retriever import VectorRetriever
from src.generator.filter_extractor import FilterExtractor


def _summarize_filters(criteria) -> str:
    parts = []
    if criteria.year:
        parts.append(f"year={criteria.year}")
    if criteria.year_lte:
        parts.append(f"year<={criteria.year_lte}")
    if criteria.year_gte:
        parts.append(f"year>={criteria.year_gte}")
    if criteria.author:
        parts.append(f"author={criteria.author}")
    if criteria.author_role:
        parts.append(f"role={criteria.author_role}")
    if criteria.source_name:
        parts.append(f"source={criteria.source_name}")
    if criteria.period:
        parts.append(f"period={criteria.period}")
    if criteria.session:
        parts.append(f"session={criteria.session}")
    if criteria.document_type:
        parts.append(f"type={criteria.document_type}")
    return ", ".join(parts) if parts else "none"


def _summarize_where_filter(where: Optional[dict]) -> str:
    """Summarize ChromaDB where filter for trace display."""
    if where is None:
        return "none"
    parts = []
    if isinstance(where, dict):
        if "year" in where:
            parts.append(f"year={where['year']}")
        if "$or" in where:
            years = [c.get("$eq") for c in where["$or"] if isinstance(c, dict) and "$eq" in c]
            parts.append(f"year in {years}")
    return ", ".join(parts) if parts else "custom"


class _noop_ctx:
    """No-op context manager for optional tracer support."""
    def __enter__(self):
        return None
    def __exit__(self, *args):
        pass


class RAGService:
    def __init__(self, pipeline_config_path: Optional[str] = None) -> None:
        self.retriever = self._init_retriever()
        self.generator = OllamaGenerator()
        self.filter_extractor = FilterExtractor()
        self._pipeline_config_path = pipeline_config_path
        self._agent = None
        self._orchestrator = None

    @staticmethod
    def _init_retriever() -> Optional[VectorRetriever]:
        """Try to open the default collection; return None if it doesn't exist yet."""
        try:
            from chromadb.errors import InvalidCollectionException, NotFoundError
            return VectorRetriever(get_spec(settings.DEFAULT_COLLECTION))
        except (InvalidCollectionException, NotFoundError, ValueError):
            return None
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Retrieval — delegates to HybridRetriever
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: int = settings.RETRIEVE_TOP_K,
        fetch_k: int = settings.RETRIEVE_FETCH_K,
        mufettis_mode: bool = False,
        where_filter: Optional[dict] = None,
        tracer: Optional[PipelineTracer] = None,
    ) -> RetrievalResult:
        def _empty(r: RetrievalResult) -> bool:
            return len(r["documents"][0]) == 0

        # Mimari not: Fallback cascade orkestrasyonu bilerek burada (RAGService
        # façade katmanı) tutulur, VectorRetriever'da değil. Retriever alt katmandır
        # ve filtreden bağımsızdır: yalnızca ham `where_filter` dict alıp ANN çalıştırır,
        # FilterCriteria/FilterExtractor bilgisini taşımaz. Cascade ise filtre alan
        # bilgisine ihtiyaç duyar (önce en güvenilmez alan olan author düşürülür).
        # Döngüyü retriever'a taşımak, alt katmanın generator katmanına bağımlı olmasına
        # (katman yönü ihlali) yol açardı. Sorumluluk ayrımı:
        #   - politika (neyi hangi sırada gevşet) -> FilterExtractor.fallback_chain
        #   - arama (where dict ile ANN)          -> VectorRetriever
        #   - orkestrasyon (extract -> aday döngüsü -> ilk dolu sonuç) -> RAGService

        # If where_filter is not provided, extract dynamically
        if where_filter is None:
            with (tracer.phase("filter_extraction", model=self.filter_extractor.model, details={"original_query": query}) if tracer else _noop_ctx()) as ctx:
                extracted = self.filter_extractor.extract(query)
                hints_found = self.filter_extractor.has_filter_hints(query)
                refined_query = extracted.refined_query.strip() or query
                criteria = extracted.filters
                candidates = self.filter_extractor.fallback_chain(criteria)
                if ctx:
                    ctx.update_details(
                        hints_found=hints_found,
                        refined_query=refined_query,
                        filters=_summarize_filters(criteria),
                        removed_words=extracted.removed_words or [],
                        fallback_chain=[c[0] or "full" for c in candidates],
                    )
        else:
            # Explicit filter passed by caller: no fallback cascade
            refined_query = query
            candidates = [(None, where_filter)]

        result = None
        fallback_level = None

        # Iterate cascade candidates, stop at first non-empty result
        if self.retriever is None:
            raise RuntimeError(
                f"Varsayılan koleksiyon '{settings.DEFAULT_COLLECTION}' bulunamadı. "
                "Lütfen önce veriyi indeksleyin veya RAG_DEFAULT_COLLECTION ortam değişkenini ayarlayın."
            )

        for level, where in candidates:
            with (tracer.phase("retrieval", details={"fallback_level": level or "full", "collection": self.retriever.spec.name, "where_filter_summary": _summarize_where_filter(where)}) if tracer else _noop_ctx()) as ctx:
                result = self.retriever.retrieve(
                    refined_query,
                    top_k=top_k,
                    fetch_k=fetch_k,
                    mufettis_mode=mufettis_mode,
                    where_filter=where,
                )
                if ctx:
                    ctx.update_details(
                        result_count=len(result["documents"][0]),
                    )
                if not _empty(result):
                    fallback_level = level
                    break

        # If all candidates empty, keep the last result (semantic_only)
        if result is None:
            result = self.retriever.retrieve(
                refined_query,
                top_k=top_k,
                fetch_k=fetch_k,
                mufettis_mode=mufettis_mode,
                where_filter=None,
            )
            fallback_level = "semantic_only"

        result["fallback_level"] = fallback_level

        # Query expansion for müfettiş mode (expands then re-retrieves)
        if mufettis_mode:
            expanded = self.generator.expand_query(refined_query)
            combined_query = f"{refined_query} {expanded}"
            result = self.retriever.retrieve(
                combined_query,
                top_k=settings.MUFETTIS_TOP_K,
                fetch_k=settings.MUFETTIS_FETCH_K,
                mufettis_mode=True,
                where_filter=where_filter,
            )
            result["expanded_query"] = expanded
        return result

    def build_context(
        self,
        results: RetrievalResult,
        max_chars: int = settings.CONTEXT_BUILD_DEFAULT_MAX,
        distance_threshold: float = settings.DISTANCE_THRESHOLD,
        total_max_chars: int = settings.CONTEXT_BUILD_DEFAULT_TOTAL,
    ) -> str:
        return build_context(
            results,
            max_chars=max_chars,
            distance_threshold=distance_threshold,
            total_max_chars=total_max_chars,
        )

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def ask_from_results(
        self, query: str, results: RetrievalResult
    ) -> tuple[str, str]:
        context = build_context(results)
        if not context.strip():
            return "", "Arşivde bu soruyu yanıtlayacak yeterli bilgi bulunamadı."
        return self.generator.answer(query, context)

    def ask_stream(
        self, query: str, mufettis_mode: bool = False, tracer: Optional[PipelineTracer] = None, results: Optional[RetrievalResult] = None
    ) -> Iterable[StreamChunk]:
        if mufettis_mode:
            from src.generator.deep_pipeline import DeepPipeline
            yield from DeepPipeline(self).run(query)
            return
        if results is None:
            results = self.retrieve(query, mufettis_mode=False, tracer=tracer)
        ctx_args = {
            "max_chars": settings.CONTEXT_MAX_CHARS,
            "total_max_chars": settings.CONTEXT_TOTAL_MAX,
        }
        with (tracer.phase("context_building", details={"distance_threshold": settings.DISTANCE_THRESHOLD}) if tracer else _noop_ctx()) as ctx:
            context = build_context(results, **ctx_args)
            if ctx:
                total_chunks = len(results["documents"][0])
                kept = sum(1 for d in results["distances"][0] if d is not None and d <= settings.DISTANCE_THRESHOLD)
                ctx.update_details(
                    total_chunks=total_chunks,
                    kept_chunks=kept,
                    context_chars=len(context),
                )
        if not context.strip():
            yield StreamChunk(type="content", content="Arşivde bu soruyu yanıtlayacak yeterli bilgi bulunamadı.")
            return
        with (tracer.phase("generation", model=settings.LLM_MODEL, details={"context_chars": len(context)}) if tracer else _noop_ctx()):
            yield from self.generator.stream(query, context, mufettis_mode=False)

    def ask(self, query: str, debug: bool = False) -> tuple[str, str]:
        results = self.retrieve(query)
        if debug:
            docs = results.get("documents", [[]])[0]
            dists = results.get("distances", [[]])[0]
            metas = results.get("metadatas", [[]])[0]
            print(f"\n[DEBUG] Sorgu: {repr(query)}")
            print(f"[DEBUG] Bulunan {len(docs)} chunk:")
            for i, (doc, meta, dist) in enumerate(zip(docs, metas, dists)):
                print(f"  [{i}] dist={dist:.4f} | {meta.get('publication')} | {meta.get('title','')[:50]}")
                print(f"       metin: {repr(doc[:80])}")
        context = build_context(results)
        if debug:
            print(f"[DEBUG] Context uzunluğu: {len(context)} karakter")
            print(f"[DEBUG] Context boş mu: {not context.strip()}")
            if context.strip():
                print(f"[DEBUG] Context (ilk 300 chr):\n{context[:300]}")
        if not context.strip():
            return "", "Arşivde bu soruyu yanıtlayacak yeterli bilgi bulunamadı."
        return self.generator.answer(query, context)

    # ------------------------------------------------------------------
    # Source inspection (for /kaynak N)
    # ------------------------------------------------------------------

    def inspect_record(self, source_db: str, chunk_id: str) -> Optional[dict]:
        from src.config.collections import get_default_spec
        from src.config.document_types import DocumentType
        from src.common.chroma import open_collection

        _DB_TO_TYPE = {
            "gazete": DocumentType.GAZETE,
            "minutes": DocumentType.TUTANAK,
            "onerge": DocumentType.ONERGE,
        }
        doc_type = _DB_TO_TYPE.get(source_db)
        if doc_type is None:
            return self.retriever.inspect_record(source_db, chunk_id)

        spec = get_default_spec(doc_type)
        try:
            _, collection = open_collection(spec.db_path, spec.name)
            res = collection.get(ids=[chunk_id], include=["documents", "metadatas"])
            if not res["ids"]:
                return None
            meta = dict(res["metadatas"][0] or {})
            meta["content"] = res["documents"][0]
            meta["chunk_id"] = chunk_id
            return meta
        except Exception as e:
            print(f"Chroma inspect error: {e}")
            return None

    # ------------------------------------------------------------------
    # Agent pipeline (Planning Agent orchestration)
    # ------------------------------------------------------------------

    def _get_agent(self):
        """Lazy-init the Planning Agent with pipeline config."""
        if self._agent is not None:
            return self._agent

        from src.config.pipeline_loader import load_pipeline_config
        from src.common.llm_client_pool import LLMClientPool
        from src.agent.planner import PlanningAgent

        config = load_pipeline_config(self._pipeline_config_path)
        if config is None:
            raise RuntimeError(
                "pipeline.yaml not found. Run without --agent or provide --pipeline <path>"
            )

        pool = LLMClientPool.from_config(config)
        self._agent = PlanningAgent(config, pool)
        return self._agent

    def _get_orchestrator(self):
        """Lazy-init the OrchestratorAgent with pipeline config."""
        if self._orchestrator is not None:
            return self._orchestrator

        from src.agent.orchestrator import OrchestratorAgent
        from src.common.llm_client_pool import LLMClientPool
        from src.config.pipeline_loader import load_pipeline_config

        config = load_pipeline_config(self._pipeline_config_path)
        if config is None:
            raise RuntimeError(
                "pipeline.yaml not found. Run without --agent or provide --pipeline <path>"
            )

        pool = LLMClientPool.from_config(config)
        self._orchestrator = OrchestratorAgent(config, pool)
        return self._orchestrator

    def _orchestrator_enabled(self) -> bool:
        """Read the orchestrator feature flag without forcing agent construction."""
        from src.config.pipeline_loader import load_pipeline_config
        config = load_pipeline_config(self._pipeline_config_path)
        return bool(config is not None and getattr(config, "orchestrator", None) and config.orchestrator.enabled)

    def run_agent(
        self,
        query: str,
        on_phase=None,
        session_collections: Optional[list[str]] = None,
        stream_callback=None,
    ):
        """Run the agent pipeline.

        Dispatches to OrchestratorAgent when `pipeline.yaml: orchestrator.enabled`
        is true; otherwise runs the legacy PlanningAgent.

        Args:
            query: user query
            on_phase: optional callback(name, block, model, details) for legacy path
            session_collections: collections the user selected at session start
                (orchestrator path uses this for policy intersection)
            stream_callback: optional callable invoked by the orchestrator path
                when the final answer is ready; reserved for future token-level
                streaming.

        Returns:
            AgentOutput with answer, thinking, trace, plan, validation, sources.
        """
        if self._orchestrator_enabled():
            orchestrator = self._get_orchestrator()
            return orchestrator.run(
                query,
                session_collections or [],
                stream_callback=stream_callback,
            )

        from src.agent.tracer import PipelineTracer as AgentPipelineTracer

        agent = self._get_agent()
        tracer = AgentPipelineTracer(on_phase=on_phase)
        return agent.run(query, trace=tracer)
