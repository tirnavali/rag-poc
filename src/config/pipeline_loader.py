"""YAML pipeline configuration loader.

Loads pipeline.yaml and provides typed access to deployment blocks,
agent configuration, and retrieval parameters. Model specs and collection
registry are loaded from models.yaml via src/config/collections.py.

Falls back to settings.py if the YAML file is not found (backwards compatibility).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

import yaml

from src.config import settings
from src.config.settings import PROJECT_ROOT


class DeploymentBlock:
    """Represents a single LLM inference machine/service."""

    def __init__(self, name: str, config: dict) -> None:
        self.name = name
        self.host = config.get("host", "http://localhost:11434")
        self.purpose = config.get("purpose", "")
        self.models = config.get("models", {})
        self.timeout_seconds = config.get("timeout_seconds", 30)
        self.retries = config.get("retries", 1)
        self.max_num_ctx = config.get("max_num_ctx", 32768)
        self.max_num_predict = config.get("max_num_predict", 4096)

    def get_model(self, key: str) -> str:
        return self.models.get(key, "")

    def __repr__(self) -> str:
        return f"DeploymentBlock({self.name}, host={self.host})"


class AgentConfig:
    """Agent sub-configuration (planner, answering, sanitizer)."""

    def __init__(self, config: dict) -> None:
        self.block = config.get("block", "fast-01")
        self.model_key = config.get("model_key", "answer")
        self.temperature = config.get("temperature", 0.1)
        self.num_ctx = config.get("num_ctx", 32768)
        self.num_predict = config.get("num_predict", 4096)
        self.max_chars = config.get("max_chars", 2000)
        self.max_retries = config.get("max_retries", 1)
        self.retry_prompt = config.get("retry_prompt", "")
        self.validation_criteria = config.get("validation_criteria", [])
        self.think = config.get("think", None)


class PlannerConfig:
    """Planning Agent configuration."""

    def __init__(self, config: dict) -> None:
        self.block = config.get("block", "fast-01")
        self.model_key = config.get("model_key", "planner")
        self.default_query_count = config.get("default_query_count", 2)
        self.search_strategy = config.get("search_strategy", "auto")
        self.plan_prompt = config.get("plan_prompt", "")
        self.think = config.get("think", None)

        rr = config.get("re_retrieval", {})
        self.re_retrieval_enabled = rr.get("enabled", True)
        self.re_retrieval_max_retries = rr.get("max_retries", 1)
        self.re_retrieval_min_results = rr.get("trigger_min_results", 3)
        self.re_retrieval_strategy = rr.get("strategy", "broaden_filters")
        self.re_retrieval_prompt = rr.get("prompt", "")
        self.re_retrieval_on_quality_failure = rr.get("on_quality_failure", True)

        fb = config.get("fallback", {})
        self.fallback_strategy = fb.get("strategy", "broadcast")
        self.fallback_collections = fb.get("default_collections", [])
        self.fallback_queries = fb.get("default_queries", [])


class RetrievalConfig:
    """Retrieval parameters from YAML."""

    def __init__(self, config: dict) -> None:
        rerank = config.get("reranker", {})
        self.reranker_enabled = rerank.get("enabled", True)
        self.reranker_model = rerank.get("model", settings.RERANK_MODEL)
        self.reranker_fetch_k = rerank.get("fetch_k", settings.RERANK_FETCH_K)
        self.reranker_coarse_k = rerank.get("coarse_k", settings.RERANK_COARSE_K)
        self.reranker_final_k = rerank.get("final_k", settings.RERANK_FINAL_K)

        ctx = config.get("context", {})
        self.context_max_chars = ctx.get("max_chars", settings.CONTEXT_MAX_CHARS)
        self.context_total_max_chars = ctx.get("total_max_chars", settings.CONTEXT_TOTAL_MAX)
        self.distance_threshold = ctx.get("distance_threshold", settings.DISTANCE_THRESHOLD)
        self.window_size = ctx.get("window_size", settings.WINDOW_SIZE)
        self.window_max_total = ctx.get("window_max_total", settings.WINDOW_MAX_TOTAL)


class OrchestratorConfig:
    """Feature flag for new orchestrator pipeline."""

    def __init__(self, config: dict) -> None:
        self.enabled = bool(config.get("enabled", False))


class BadWordsFilterConfig:
    """Pre-planner profanity / abuse filter configuration (no LLM)."""

    def __init__(self, config: dict) -> None:
        self.enabled = bool(config.get("enabled", True))
        raw_msg = config.get("response_message")
        if not raw_msg:
            raw_msg = "Lütfen saygılı dil kullanın. Sorgunuzda uygun olmayan kelime tespit edildi."
        self.response_message = raw_msg.strip()
        self.bad_words: list[str] = list(config.get("bad_words", []))
        self.bad_word_patterns: list[str] = list(config.get("bad_word_patterns", []))

    # Adapter properties so BadWordsFilter can consume this directly via Protocol
    @property
    def bad_words_enabled(self) -> bool:
        return self.enabled

    @property
    def bad_words_response_message(self) -> str:
        return self.response_message


class ClassifierConfig:
    """Pre-planner scope classifier configuration."""

    def __init__(self, config: dict) -> None:
        self.enabled = bool(config.get("enabled", True))
        self.block = config.get("block", "fast-01")
        self.model_key = config.get("model_key", "classifier")
        self.temperature = float(config.get("temperature", 0.0))
        self.confidence_threshold = float(config.get("confidence_threshold", 0.6))
        self.think = config.get("think", False)
        self.prompt = config.get("prompt", "")


class SuggesterConfig:
    """Off-domain in-domain suggestion generator configuration."""

    def __init__(self, config: dict) -> None:
        self.block = config.get("block", "fast-01")
        self.model_key = config.get("model_key", "suggester")
        self.temperature = float(config.get("temperature", 0.3))
        self.think = config.get("think", False)
        self.suggestion_count = int(config.get("suggestion_count", 3))
        self.prompt = config.get("prompt", "")


class PolicyConfig:
    """Collection-access policy configuration."""

    def __init__(self, config: dict) -> None:
        self.mode = config.get("mode", "session_intersection")


class _AllocationBudget:
    """Tuple-like budget triple for one query_type."""

    def __init__(self, primary: int, reserve: int, fetch_k: int) -> None:
        self.primary = primary
        self.reserve = reserve
        self.fetch_k = fetch_k


class AllocationConfig:
    """Per-query-type retrieval budget configuration."""

    def __init__(self, config: dict) -> None:
        defaults = config.get("defaults", {})
        self._defaults = _AllocationBudget(
            primary=int(defaults.get("primary", 2)),
            reserve=int(defaults.get("reserve", 2)),
            fetch_k=int(defaults.get("fetch_k", 10)),
        )
        raw_by_qt = config.get("by_query_type", {})
        self._by_query_type: dict[str, _AllocationBudget] = {}
        for qt, cfg in raw_by_qt.items():
            self._by_query_type[qt] = _AllocationBudget(
                primary=int(cfg.get("primary", self._defaults.primary)),
                reserve=int(cfg.get("reserve", self._defaults.reserve)),
                fetch_k=int(cfg.get("fetch_k", self._defaults.fetch_k)),
            )
        self.max_per_document = int(config.get("max_per_document", 1))
        self.max_total_primary = int(config.get("max_total_primary", 12))

    def budget_for(self, query_type: str) -> _AllocationBudget:
        return self._by_query_type.get(query_type, self._defaults)


class _JudgeHeuristicConfig:
    def __init__(self, config: dict) -> None:
        self.min_chunks = int(config.get("min_chunks", 4))
        self.min_collection_coverage = int(config.get("min_collection_coverage", 2))
        self.min_rerank_score = float(config.get("min_rerank_score", 0.0))


class _JudgeLLMConfig:
    def __init__(self, config: dict) -> None:
        self.enabled = bool(config.get("enabled", True))
        self.block = config.get("block", "fast-01")
        self.model_key = config.get("model_key", "judge")
        band = config.get("borderline_band", [2, 4])
        self.borderline_band: tuple[int, int] = (int(band[0]), int(band[1]))
        self.max_borderline_score_floor = float(config.get("max_borderline_score_floor", 0.35))
        self.timeout_seconds = int(config.get("timeout_seconds", 5))


class JudgeConfig:
    """EvidenceJudge configuration (hybrid heuristic + LLM)."""

    def __init__(self, config: dict) -> None:
        self.mode = config.get("mode", "hybrid")
        self.heuristic = _JudgeHeuristicConfig(config.get("heuristic", {}))
        self.llm = _JudgeLLMConfig(config.get("llm", {}))
        self.max_expand_iterations = int(config.get("max_expand_iterations", 1))
        self.on_low_confidence = config.get("on_low_confidence", "expand")


class PipelineConfig:
    """Top-level pipeline configuration loaded from YAML."""

    def __init__(self, config: dict) -> None:
        blocks = config.get("deployment_blocks", {})
        self.blocks: dict[str, DeploymentBlock] = {
            name: DeploymentBlock(name, cfg) for name, cfg in blocks.items()
        }

        agent_cfg = config.get("agent", {})
        self.bad_words_filter = BadWordsFilterConfig(agent_cfg.get("bad_words_filter", {}))
        self.classifier = ClassifierConfig(agent_cfg.get("classifier", {}))
        self.suggester = SuggesterConfig(agent_cfg.get("suggester", {}))
        self.off_domain_response_template = agent_cfg.get(
            "off_domain_response_template", ""
        )
        self.off_domain_fallback_suggestions: list[str] = list(
            agent_cfg.get("off_domain_fallback_suggestions", [])
        )
        self.planner = PlannerConfig(agent_cfg.get("planner", {}))
        self.answering = AgentConfig(agent_cfg.get("answering", {}))
        self.sanitizer = AgentConfig(agent_cfg.get("sanitizer", {}))
        self.filter_extractor = AgentConfig(agent_cfg.get("filter_extractor", {
            "block": "fast-01",
            "model_key": "filter_extractor",
            "temperature": 0.0,
            "think": False
        }))

        self.retrieval = RetrievalConfig(config.get("retrieval", {}))

        # New orchestrator blocks (optional; safe defaults when missing)
        self.orchestrator = OrchestratorConfig(config.get("orchestrator", {}))
        self.policy = PolicyConfig(config.get("policy", {}))
        self.allocation = AllocationConfig(config.get("allocation", {}))
        self.judge = JudgeConfig(config.get("judge", {}))

    def get_block(self, name: str) -> DeploymentBlock:
        if name not in self.blocks:
            raise KeyError(
                f"Deployment block '{name}' not found. "
                f"Available: {list(self.blocks.keys())}"
            )
        return self.blocks[name]

    def get_collection_catalog(self) -> str:
        """Return a human-readable catalog of collections for the agent prompt.

        Only the default (canonical) collection per document type is listed —
        the `defaults` map in models.yaml. models.yaml also registers many
        experimental/comparison collections (e.g. minutes_jina_v4,
        tutanaklar_jina_v3_4k) that are not the live retrieval target; exposing
        them here makes the planner route to dead collections. The planner must
        only see one active collection per doc_type.
        """
        from src.config.collections import COLLECTIONS, DEFAULT_COLLECTION_FOR_TYPE
        from src.config.document_types import DOCUMENT_TYPES, DocumentType

        content_hints: dict[DocumentType, str] = {
            DocumentType.GAZETE: "basın/gazete/köşe yazısı arşivi",
            DocumentType.TUTANAK: "meclis görüşme kayıtları/oturum/birleşim",
            DocumentType.ONERGE: "kanun teklifi/önerge metinleri",
            DocumentType.CUSTOM: "özel kaynak",
        }

        lines = []
        for dt, key in DEFAULT_COLLECTION_FOR_TYPE.items():
            spec = COLLECTIONS.get(key)
            if spec is None:
                continue
            label = DOCUMENT_TYPES[dt].display_name_tr
            hint = content_hints.get(dt, "")
            descriptor = f"{label} — {hint}" if hint else label
            lines.append(
                f"- {key} ({descriptor}): doc_type={dt.value}, embedder={spec.embed_model}"
            )
        return "\n".join(lines)

    def get_collection_keys(self) -> list[str]:
        """Return all registered collection keys."""
        from src.config.collections import COLLECTIONS
        return list(COLLECTIONS.keys())


def load_pipeline_config(path: str | Path | None = None) -> PipelineConfig | None:
    """Load pipeline.yaml from the given path or default location.

    Returns None if the file doesn't exist (caller should fall back to settings.py).
    """
    if path is None:
        path = PROJECT_ROOT / "pipeline.yaml"

    config_path = Path(path)
    if not config_path.exists():
        return None

    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError(f"Invalid pipeline config at {config_path}: expected a YAML mapping")

    return PipelineConfig(raw)
