"""YAML pipeline configuration loader.

Loads pipeline.yaml and provides typed access to deployment blocks,
agent configuration, and retrieval parameters. Model specs and collection
registry are loaded from models.yaml via src/config/collections.py.

Falls back to settings.py if the YAML file is not found (backwards compatibility).
"""
from __future__ import annotations

import os
import re
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
        # Normal mode caps query diversification (one search per variant, RRF-fused).
        self.normal_max_query_variants = int(config.get("normal_max_query_variants", 5))
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
        # Stage-2: off by default. When disabled the orchestrator allows the
        # planner-suggested collections through unchanged.
        self.enabled = bool(config.get("enabled", False))
        self.mode = config.get("mode", "session_intersection")


class _QueryTypeBudget:
    """Retrieval budget for one query_type, plus optional context caps.

    ``fetch_k`` is the per-collection retrieval depth (passed as top_k to the
    search tool). ``max_total`` / ``max_per_document`` override the global assembly
    caps for this query_type when set (None → fall back to the global value), so
    'comprehensive' queries can assemble a much larger single context.
    """

    def __init__(
        self,
        fetch_k: int,
        max_total: int | None = None,
        max_per_document: int | None = None,
    ) -> None:
        self.fetch_k = fetch_k
        self.max_total = max_total
        self.max_per_document = max_per_document


class RetrievalBudgetConfig:
    """Per-query-type retrieval budget: fetch_k depth + assembly caps.

    The orchestrator maps the planner's ``query_type`` to a fetch_k here and the
    assembler reads the per-type / global context caps. This is a plain config
    lookup — there is no separate allocation stage.
    """

    def __init__(self, config: dict) -> None:
        defaults = config.get("defaults", {})
        self._defaults = _QueryTypeBudget(
            fetch_k=int(defaults.get("fetch_k", 10)),
        )
        raw_by_qt = config.get("by_query_type", {})
        self._by_query_type: dict[str, _QueryTypeBudget] = {}
        for qt, cfg in raw_by_qt.items():
            self._by_query_type[qt] = _QueryTypeBudget(
                fetch_k=int(cfg.get("fetch_k", self._defaults.fetch_k)),
                max_total=int(cfg["max_total"]) if cfg.get("max_total") is not None else None,
                max_per_document=int(cfg["max_per_document"]) if cfg.get("max_per_document") is not None else None,
            )
        self.max_per_document = int(config.get("max_per_document", 1))
        self.max_total_primary = int(config.get("max_total_primary", 12))
        # Enumerate/comprehensive gather tuning:
        # - facet_partition: also run one search per mined year facet, so each year
        #   in the corpus is covered instead of only the globally top-ranked region.
        # - fetch_k_max: depth-escalation ceiling; a dry round doubles fetch_k up to
        #   this before declaring saturation, reaching ranks beyond the initial top-K.
        self.enumerate_facet_partition = bool(config.get("enumerate_facet_partition", True))
        self.enumerate_fetch_k_max = int(config.get("enumerate_fetch_k_max", 120))

    def budget_for(self, query_type: str) -> _QueryTypeBudget:
        return self._by_query_type.get(query_type, self._defaults)

    def max_total_for(self, query_type: str) -> int:
        """Per-query-type assembled-context cap, falling back to the global value."""
        b = self._by_query_type.get(query_type)
        return b.max_total if b is not None and b.max_total is not None else self.max_total_primary

    def max_per_document_for(self, query_type: str) -> int:
        """Per-query-type per-document cap, falling back to the global value."""
        b = self._by_query_type.get(query_type)
        return b.max_per_document if b is not None and b.max_per_document is not None else self.max_per_document


class _JudgeHeuristicConfig:
    def __init__(self, config: dict) -> None:
        self.min_chunks = int(config.get("min_chunks", 4))
        self.min_collection_coverage = int(config.get("min_collection_coverage", 2))
        self.min_rerank_score = float(config.get("min_rerank_score", 0.0))
        # Escalation, not rejection: when count/coverage otherwise look sufficient
        # but the BEST match is still this weak, don't auto-answer — hand off to
        # the LLM judge to actually read the content and decide. A plain reject
        # threshold was tested empirically against the golden Q&A fixture and
        # rejected: correct-but-abstractly-phrased queries ("olumlu yanları",
        # "eleştiriler") score in the same low range as genuine topic mismatches,
        # so a numeric cutoff alone misfires on real, common queries.
        self.llm_escalation_score = float(config.get("llm_escalation_score", 0.3))


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
        # Hard cap on iterative gather rounds for 'comprehensive' (enumeration) queries.
        self.comprehensive_max_rounds = int(config.get("comprehensive_max_rounds", 3))
        self.on_low_confidence = config.get("on_low_confidence", "expand")


class _WindowExpandConfig:
    """Reflect window-expand (pencere-genişletme) knobs.

    After an adaptive hop retrieves, the top-scored chunk (anchor) has its
    document-internal chunk-order neighbors fetched (ANN-free) and spliced into the
    pool, so an identity-less region (e.g. a roll-call vote table) adjacent to a
    self-identifying anchor comes along in reading order. Default OFF; effective only
    when a strategy also opts in (``window_expand: true``). Names avoid RetrievalConfig's
    char-based ``window_size``/``window_max_total``.
    """

    def __init__(self, config: dict) -> None:
        self.enabled = bool(config.get("enabled", False))
        self.neighbor_radius = int(config.get("neighbor_radius", 2))
        self.max_neighbors_per_hop = int(config.get("max_neighbors_per_hop", 8))
        self.anchor_count = int(config.get("anchor_count", 1))


class ReflectConfig:
    """Adaptive reflect/re-plan node configuration.

    Governs the ``mode: adaptive`` multi-hop path (research_strategies.md). The
    reflect node runs a strategy-specific self-loop that executes the procedure
    recipe round by round; ``enabled`` is a dedicated kill-switch INDEPENDENT of
    the generic judge/expansion knobs (a query resolving to an adaptive strategy
    only reflects when this is True). The reflect LLM reuses the planner block, so
    no separate deployment block is needed. ``default_max_rounds`` is the fallback
    ceiling for an adaptive strategy that omits its own ``max_rounds`` (never the
    neutralized judge knobs). The evidence-* caps bound the deterministic compact
    evidence summary handed to the reflect LLM each round.
    """

    def __init__(self, config: dict) -> None:
        self.enabled = bool(config.get("enabled", True))
        self.default_max_rounds = int(config.get("default_max_rounds", 4))
        self.evidence_max_chunks = int(config.get("evidence_max_chunks", 10))
        self.evidence_char_cap_per_chunk = int(config.get("evidence_char_cap_per_chunk", 600))
        self.evidence_total_char_cap = int(config.get("evidence_total_char_cap", 6000))
        # Reflect SELF-loop'un hacim güvenlik tavanı — generic çevrimin query_type
        # tavanı DEĞİL. Kritik: reasoning/summary adaptive stratejilerde tek-atım ilk
        # retrieval zaten max_total_primary'yi (15) doldurur; o tavan kullanılırsa
        # reflect 1. turda "ceiling" ile durur ve HİÇ hop atmaz. Reflect'in gerçek
        # bağı max_rounds + done; bu yalnızca bağlam taşmasını önleyen üst sınır
        # (comprehensive'in kanıtlı-güvenli 50 chunk bağıyla hizalı).
        self.max_total_chunks = int(config.get("max_total_chunks", 50))
        self.window_expand = _WindowExpandConfig(config.get("window_expand", {}))


class ClarificationConfig:
    """Grounded clarification (did-you-mean) stage configuration.

    Probe-retrieves a small set, mines facets from their metadata, and — when the
    query is ambiguous — asks the user to narrow year/scope/topic. In
    non-interactive contexts (no callback) the strongest facet is auto-applied.
    """

    def __init__(self, config: dict) -> None:
        self.enabled = bool(config.get("enabled", True))
        self.probe_k = int(config.get("probe_k", 20))
        # Number of facet-grounded "rabbit hole" drill-down suggestions to surface
        # for broad/ambiguous queries (replaces the old hard-narrowing behavior).
        self.suggestion_count = int(config.get("suggestion_count", 3))
        # DEPRECATED — no longer used (query is no longer narrowed pre-answer).
        self.question_count = int(config.get("question_count", 3))
        self.max_turns_normal = int(config.get("max_turns_normal", 1))
        self.max_turns_deep = int(config.get("max_turns_deep", 2))
        # Ambiguity gate: ask only when the probe results are spread out.
        gate = config.get("ambiguity", {})
        self.min_distinct_years = int(gate.get("min_distinct_years", 3))
        self.dominance_ratio = float(gate.get("dominance_ratio", 0.6))
        # Also clarify when the query itself is broad/unscoped (no year + generic
        # "X hakkında bilgi" phrasing), independent of facet diversity.
        self.vague_query_clarify = bool(gate.get("vague_query_clarify", True))
        markers = gate.get("vague_markers")
        self.vague_markers = (
            [str(m).lower() for m in markers] if isinstance(markers, list) else None
        )
        # LLM used only to phrase the (deterministically mined) facets as questions.
        self.block = config.get("block", "fast-01")
        self.model_key = config.get("model_key", "planner")
        self.temperature = float(config.get("temperature", 0.2))
        self.think = config.get("think", False)
        self.prompt = config.get("prompt", "")


class StrategyPlaybook:
    """Parses ``research_strategies.md`` into a prompt-ready catalog + lookup map.

    Format: ``## <name>`` sections, each with unindented ``key: value`` lines;
    a value may continue onto following indented lines (see the file itself
    for the canonical example). Recognized keys:
      * ``triggers`` — comma-separated hint keywords.
      * ``query_type`` — drives retrieval budget + judge presets.
      * ``answer_directive`` — appended to the answering system prompt.
      * ``mode`` — ``static`` (default) or ``adaptive``; ``adaptive`` marks a
        multi-hop strategy whose ``procedure`` a reflect step consumes.
      * ``max_rounds`` — int cap on expansion rounds for this strategy (None = default).
      * ``anchor`` / ``target`` — the entity resolved first / the answer shape sought.
      * ``exclude_seen_chunks`` — ``true`` makes each reflect hop hold the exact chunk
        ids already surfaced this run out of its ranked pool, so fetch_k fills with novel
        chunks (see ``OrchestratorAgent._seen_chunk_ids`` / ``_run_retrieval``). Chunk-id,
        not a metadata ``$nin`` — never blacks out a whole sitting.
      * ``aliases`` — ``;``-separated ``term -> official_phrase`` pairs mapping a
        colloquial query word to its archive phrasing (term-hypothesis seed).
      * ``procedure`` — free-text multi-hop recipe (read by the reflect step).

    The first three keys are consumed today (catalog + query_type +
    answer_directive); the rest are parsed and exposed via ``get_strategy()`` for
    the adaptive reflect step, and are inert no-ops until it is wired in.

    Fail-open: a missing/unreadable file yields an empty catalog and an empty
    ``by_name`` map, so callers fall back to Faz A behavior (deterministic
    COMPREHENSIVE_KEYWORDS override only, no answer_directive) without error.
    Malformed optional fields (bad int, alias with no ``->``) are dropped, never
    raised, so one typo can't sink the whole playbook load.
    """

    _HEADER_RE = re.compile(r"^##\s+(.+?)\s*$")
    _FIELD_RE = re.compile(r"^([a-zA-Z_][a-zA-Z0-9_]*):\s*(.*)$")

    def __init__(self, path: "str | Path | None" = None) -> None:
        self._path = Path(path) if path is not None else (PROJECT_ROOT / "research_strategies.md")
        self.by_name: dict[str, dict[str, Any]] = {}
        self.catalog_text: str = ""
        self._load()

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            text = self._path.read_text(encoding="utf-8")
        except OSError:
            return
        self.by_name = self._parse(text)
        self.catalog_text = self._build_catalog(self.by_name)

    @classmethod
    def _parse(cls, text: str) -> dict[str, dict[str, Any]]:
        sections: dict[str, dict[str, Any]] = {}
        name: str | None = None
        fields: dict[str, str] = {}
        current_key: str | None = None

        def flush() -> None:
            if name is not None:
                sections[name] = cls._finalize(fields)

        for raw_line in text.splitlines():
            header = cls._HEADER_RE.match(raw_line)
            if header:
                flush()
                name = header.group(1).strip()
                fields = {}
                current_key = None
                continue
            if name is None:
                continue
            field = cls._FIELD_RE.match(raw_line)
            if field:
                current_key = field.group(1).strip()
                fields[current_key] = field.group(2).strip()
                continue
            if current_key is not None and raw_line.strip():
                fields[current_key] = (fields[current_key] + " " + raw_line.strip()).strip()
        flush()
        return sections

    @staticmethod
    def _finalize(fields: dict[str, str]) -> dict[str, Any]:
        triggers = [t.strip() for t in fields.get("triggers", "").split(",") if t.strip()]
        mode = fields.get("mode", "").strip().lower() or "static"
        if mode not in ("static", "adaptive"):
            mode = "static"
        return {
            "query_type": fields.get("query_type", "").strip() or None,
            "answer_directive": fields.get("answer_directive", "").strip(),
            "triggers": triggers,
            "mode": mode,
            "max_rounds": StrategyPlaybook._parse_int(fields.get("max_rounds", "")),
            "anchor": fields.get("anchor", "").strip() or None,
            "target": fields.get("target", "").strip() or None,
            "exclude_seen_chunks": StrategyPlaybook._parse_bool(fields.get("exclude_seen_chunks", "")),
            "window_expand": StrategyPlaybook._parse_bool(fields.get("window_expand", "")),
            "aliases": StrategyPlaybook._parse_aliases(fields.get("aliases", "")),
            "procedure": fields.get("procedure", "").strip(),
        }

    @staticmethod
    def _parse_int(raw: str) -> "int | None":
        raw = raw.strip()
        if not raw:
            return None
        try:
            return int(raw)
        except ValueError:
            return None

    @staticmethod
    def _parse_bool(raw: str) -> bool:
        """Truthy playbook flag (fail-safe: anything unrecognized is False)."""
        return raw.strip().lower() in ("1", "true", "yes", "evet", "on")

    @staticmethod
    def _parse_aliases(raw: str) -> "list[dict[str, str]]":
        """Parse ``;``-separated ``term -> official_phrase`` pairs (fail-open).

        Each pair seeds a colloquial→archive term mapping (e.g.
        ``genel gerekçe -> sıra sayısı raporu``). Entries lacking ``->`` or a
        non-empty side are skipped rather than raising, so a stray separator can't
        break the playbook load. Multiple aliases MUST be ``;``-separated: the
        parser collapses a multi-line value onto one space-joined line, so newline
        separation would be ambiguous.
        """
        aliases: list[dict[str, str]] = []
        for chunk in raw.split(";"):
            if "->" not in chunk:
                continue
            term, _, phrase = chunk.partition("->")
            term, phrase = term.strip(), phrase.strip()
            if term and phrase:
                aliases.append({"term": term, "official_phrase": phrase})
        return aliases

    @staticmethod
    def _build_catalog(by_name: dict[str, dict[str, Any]]) -> str:
        lines = []
        for name, spec in by_name.items():
            triggers = ", ".join(spec["triggers"]) if spec["triggers"] else "(tetikleyici yok)"
            qt = spec.get("query_type") or "fact"
            directive = spec.get("answer_directive") or ""
            hint = (directive[:90] + "…") if len(directive) > 90 else directive
            suffix = f": {hint}" if hint else ""
            lines.append(f"- {name} (query_type={qt}, tetikleyiciler: {triggers}){suffix}")
        return "\n".join(lines)


class PipelineConfig:
    """Top-level pipeline configuration loaded from YAML."""

    def __init__(self, config: dict) -> None:
        blocks = config.get("deployment_blocks", {})
        self.blocks: dict[str, DeploymentBlock] = {
            name: DeploymentBlock(name, cfg) for name, cfg in blocks.items()
        }

        # How long Ollama keeps each model resident (avoids the ~11s cold-load on
        # the first query after an idle gap). Duration string ("2h"), seconds int,
        # -1 = forever, 0 = unload immediately. None → Ollama default (5m).
        self.keep_alive = config.get("keep_alive", "2h")

        agent_cfg = config.get("agent", {})
        # When true, each stage's LLM reasoning/output is attached to its trace
        # event details so the UI can show per-stage thinking.
        self.expose_thinking = bool(agent_cfg.get("expose_thinking", True))
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
        self.reflect = ReflectConfig(agent_cfg.get("reflect", {}))
        self.clarification = ClarificationConfig(agent_cfg.get("clarification", {}))
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
        self.retrieval_budget = RetrievalBudgetConfig(config.get("retrieval_budget", {}))
        self.judge = JudgeConfig(config.get("judge", {}))
        self.strategy_playbook = StrategyPlaybook()

    def get_block(self, name: str) -> DeploymentBlock:
        if name not in self.blocks:
            raise KeyError(
                f"Deployment block '{name}' not found. "
                f"Available: {list(self.blocks.keys())}"
            )
        return self.blocks[name]

    def get_collection_catalog(self, allowed_keys: "set[str] | None" = None) -> str:
        """Return a human-readable catalog of collections for the agent prompt.

        Default (``allowed_keys=None``): only the canonical collection per
        document type is listed — the `defaults` map in models.yaml. models.yaml
        also registers many experimental/comparison collections (e.g.
        tbmm_minutes_docling_jina_v4) that are not the live retrieval target;
        exposing them makes the planner route to dead collections.

        When ``allowed_keys`` is given (the user's session selection), the catalog
        lists EXACTLY those registered collections — including non-default ones
        (e.g. a `test` collection). Otherwise selecting a non-default collection
        would yield an empty catalog and the planner, seeing nothing, would
        hallucinate an out-of-scope collection. Each line still carries
        ``doc_type=...`` so the planner learns the collection's type and routes
        correctly.

        Args:
            allowed_keys: when given, list these collection keys (the session
                selection); None lists the type defaults.
        """
        from src.config.collections import COLLECTIONS, DEFAULT_COLLECTION_FOR_TYPE
        from src.config.document_types import DOCUMENT_TYPES, DocumentType

        content_hints: dict[DocumentType, str] = {
            DocumentType.GAZETE: "basın/gazete/köşe yazısı arşivi",
            DocumentType.TUTANAK: "meclis görüşme kayıtları/oturum/birleşim",
            DocumentType.ONERGE: "kanun teklifi/önerge metinleri",
            DocumentType.CUSTOM: "özel kaynak",
        }

        if allowed_keys is not None:
            # Session selection: list exactly the chosen registered collections,
            # whatever their doc_type (not limited to the per-type defaults).
            keys = [k for k in allowed_keys if k in COLLECTIONS]
        else:
            keys = list(DEFAULT_COLLECTION_FOR_TYPE.values())

        lines = []
        for key in keys:
            spec = COLLECTIONS.get(key)
            if spec is None:
                continue
            dt = spec.doc_type
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

    def get_strategy_catalog(self) -> str:
        """Human-readable strategy catalog for the planner prompt (empty if no playbook)."""
        return self.strategy_playbook.catalog_text

    def get_strategy(self, name: str) -> "dict[str, Any] | None":
        """Look up a playbook strategy by name; None if undefined (fail-open)."""
        return self.strategy_playbook.by_name.get(name)


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
