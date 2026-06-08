"""Suggester — produces in-domain query suggestions for off-domain queries."""
from __future__ import annotations

import json
import logging

from src.agent.tracer import PipelineTracer
from src.common.llm_client_pool import LLMClientPool
from src.common.llm_utils import extract_json_from_text
from src.config.pipeline_loader import PipelineConfig

logger = logging.getLogger(__name__)


class Suggester:
    """One LLM call that returns 3 in-domain query suggestions.

    Fail-open: on any failure (LLM error, invalid JSON, missing field), use
    `off_domain_fallback_suggestions` from config. Filters out any suggestion
    that exactly matches the user's query (case-sensitive); pads from fallbacks
    when fewer than `suggestion_count` items remain; trims when more.
    """

    def __init__(self, pool: LLMClientPool, config: PipelineConfig) -> None:
        self._pool = pool
        self._config = config

    def suggest(self, query: str, tracer: PipelineTracer) -> list[str]:
        cfg = self._config.suggester
        block_name = cfg.block
        model = self._pool.get_model_for_block(block_name, cfg.model_key)
        block = self._config.get_block(block_name)
        target_count = cfg.suggestion_count
        fallbacks = list(self._config.off_domain_fallback_suggestions)

        with tracer.phase(
            "suggestion",
            block=block_name,
            model=model,
            details={"query": query[:100], "n": target_count},
        ) as phase_ctx:
            raw_suggestions: list[str] = []
            try:
                client = self._pool.get_client(block_name)
                system_prompt = cfg.prompt.format(catalog=self._config.get_collection_catalog())
                res = client.chat(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Off-domain sorgu: {query}"},
                    ],
                    options={
                        "temperature": cfg.temperature,
                        "num_predict": min(512, block.max_num_predict),
                    },
                    format="json",
                    think=bool(cfg.think) if cfg.think is not None else False,
                )
                data = json.loads(extract_json_from_text(res.message.content))
                items = data.get("suggestions", [])
                if isinstance(items, list):
                    raw_suggestions = [str(s).strip() for s in items if str(s).strip()]
            except Exception as e:
                logger.warning("Suggester failed (%s); using fallbacks", e)

            # Drop exact-echo of user query
            filtered = [s for s in raw_suggestions if s != query]

            # Pad from fallbacks, then trim to target_count
            i = 0
            while len(filtered) < target_count and i < len(fallbacks):
                if fallbacks[i] not in filtered and fallbacks[i] != query:
                    filtered.append(fallbacks[i])
                i += 1
            out = filtered[:target_count]

            phase_ctx.update_details(returned=len(out))
            return out
