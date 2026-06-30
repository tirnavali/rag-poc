"""Scope classifier — fast pre-planner gate that labels a query in_scope vs off_domain."""
from __future__ import annotations

import json
import logging

from src.agent.schemas import ScopeResult
from src.agent.tracer import PipelineTracer
from src.common.llm_client_pool import LLMClientPool
from src.common.llm_utils import extract_json_from_text
from src.config.pipeline_loader import PipelineConfig

logger = logging.getLogger(__name__)


class ScopeClassifier:
    """IntentAnalyzer: one LLM call returning {scope, confidence, selected_collections, reason}.

    Doubles as the mini intent analysis (tool/db selection): besides the in/off
    domain gate it picks which collection(s) the query points at. Fail-open: any
    LLM/parse failure returns ScopeResult(in_scope, 0.0, [], "") so the caller's
    threshold check allows the query through and the planner selects freely.
    """

    def __init__(self, pool: LLMClientPool, config: PipelineConfig) -> None:
        self._pool = pool
        self._config = config

    def classify(self, query: str, tracer: PipelineTracer) -> ScopeResult:
        cfg = self._config.classifier
        block_name = cfg.block
        model = self._pool.get_model_for_block(block_name, cfg.model_key)
        block = self._config.get_block(block_name)

        with tracer.phase(
            "classification",
            block=block_name,
            model=model,
            details={"query": query[:100]},
        ) as phase_ctx:
            try:
                client = self._pool.get_client(block_name)
                # Prompt may reference {catalog} for tool/db selection; format
                # defensively so an unparameterized prompt still works.
                system_prompt = cfg.prompt
                if "{catalog}" in system_prompt:
                    system_prompt = system_prompt.format(
                        catalog=self._config.get_collection_catalog()
                    )
                res = client.chat(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Sorgu: {query}"},
                    ],
                    options={
                        "temperature": cfg.temperature,
                        "num_predict": min(256, block.max_num_predict),
                    },
                    format="json",
                    think=bool(cfg.think) if cfg.think is not None else False,
                )
                data = json.loads(extract_json_from_text(res.message.content))
                if not isinstance(data, dict):
                    raise ValueError("classifier did not return a JSON object")
                raw_cols = data.get("selected_collections", []) or []
                selected = [str(c).strip() for c in raw_cols if str(c).strip()] if isinstance(raw_cols, list) else []
                # Tolerant: a malformed/missing scope falls open to in_scope rather
                # than raising (qwen occasionally emits an off-schema object).
                scope = data.get("scope") or data.get("Scope") or "in_scope"
                if scope not in ("in_scope", "off_domain"):
                    scope = "in_scope"
                result = ScopeResult(
                    scope=scope,
                    confidence=float(data.get("confidence", 0.0) or 0.0),
                    selected_collections=selected,
                    reason=str(data.get("reason", "")),
                )
            except Exception as e:
                logger.warning("ScopeClassifier failed (%s); failing open to in_scope", e)
                result = ScopeResult(scope="in_scope", confidence=0.0, selected_collections=[], reason="")

            phase_ctx.update_details(
                scope=result.scope,
                confidence=result.confidence,
                selected_collections=result.selected_collections,
                reason=result.reason[:120],
            )
            return result
