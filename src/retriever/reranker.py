import threading

from sentence_transformers import CrossEncoder

from src.config import settings


class CrossEncoderReranker:
    _models: dict[str, CrossEncoder] = {}
    _lock = threading.Lock()

    def __init__(self, model_name: str | None = None) -> None:
        self._model_name = model_name or settings.RERANK_MODEL

    @classmethod
    def _get_model(cls, model_name: str) -> CrossEncoder:
        with cls._lock:
            if model_name not in cls._models:
                import torch
                # Apple Silicon (MPS) üzerinde oluşan hataları önlemek için, 
                # eğer CUDA yoksa (Mac) zorunlu "cpu" kullanıyoruz, CUDA varsa (Asus) "cuda" kullanıyoruz.
                device = "cuda" if torch.cuda.is_available() else "cpu"
                cls._models[model_name] = CrossEncoder(model_name, device=device)
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
