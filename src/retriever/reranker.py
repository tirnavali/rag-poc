"""Cross-encoder reranker for the second retrieval stage (coarse → fine)."""
from __future__ import annotations

from sentence_transformers import CrossEncoder

from src.config import settings


class CrossEncoderReranker:
    _models: dict[str, CrossEncoder] = {}

    def __init__(self, model_name: str | None = None) -> None:
        self._model_name = model_name or settings.RERANK_MODEL

    @classmethod
    def _get_model(cls, model_name: str) -> CrossEncoder:
        if model_name not in cls._models:
            cls._models[model_name] = CrossEncoder(model_name)
        return cls._models[model_name]

    def rerank(
        self,
        query: str,
        candidates: list[tuple[str, str]],  # [(global_key, chunk_text), ...]
        top_n: int,
    ) -> list[tuple[str, float]]:
        """Score all (query, chunk) pairs; return top_n sorted by relevance score desc."""
        if not candidates:
            return []
        model = self._get_model(self._model_name)
        pairs = [(query, chunk_text) for _, chunk_text in candidates]
        scores = model.predict(pairs)
        ranked = sorted(
            zip([gk for gk, _ in candidates], scores),
            key=lambda x: x[1],
            reverse=True,
        )
        return ranked[:top_n]
