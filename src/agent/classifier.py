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
    """One LLM call that returns {scope, confidence, reason}.

    Fail-open: any LLM/parse failure returns ScopeResult(in_scope, 0.0, "")
    so the caller's threshold check naturally allows the query through.
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
                res = client.chat(
                    model=model,
                    messages=[
                        {"role": "system", "content": cfg.prompt},
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
                result = ScopeResult(
                    scope=data["scope"],
                    confidence=float(data.get("confidence", 0.0)),
                    reason=str(data.get("reason", "")),
                )
            except Exception as e:
                logger.warning("ScopeClassifier failed (%s); failing open to in_scope", e)
                result = ScopeResult(scope="in_scope", confidence=0.0, reason="")

            phase_ctx.update_details(
                scope=result.scope,
                confidence=result.confidence,
                reason=result.reason[:120],
            )
            return result
