"""Tests for windowed late chunking in LocalLateChunkingEmbedder."""
import math
import unittest
from unittest.mock import MagicMock
import torch


def _make_embedder():
    """Return a LocalLateChunkingEmbedder with no heavy model loaded."""
    from src.trainer.ingestion.embedder import LocalLateChunkingEmbedder
    return object.__new__(LocalLateChunkingEmbedder)


def _char_offsets(text: str) -> torch.Tensor:
    """One token per character; offset_mapping tensor shape (len(text), 2)."""
    n = len(text)
    off = torch.zeros(n, 2, dtype=torch.long)
    for i in range(n):
        off[i] = torch.tensor([i, i + 1])
    return off


class FakeBatch:
    """Tokenizer output that supports both pop() and __getitem__() for offsets."""

    def __init__(self, text: str, hidden_dim: int = 8):
        n = len(text)
        self._off = _char_offsets(text)   # (n, 2)
        self._ids = torch.ones(1, n, dtype=torch.long)
        self._hidden_dim = hidden_dim
        self._data = {"input_ids": self._ids}

    def pop(self, key):
        if key == "offset_mapping":
            return self._off.unsqueeze(0)  # (1, n, 2)
        return self._data.pop(key, None)

    def to(self, device):
        return self

    def __getitem__(self, key):
        if key == "offset_mapping":
            return self._off.unsqueeze(0)  # (1, n, 2)
        return self._data[key]

    def __iter__(self):
        return iter(self._data)

    def keys(self):
        return self._data.keys()


def _make_model(hidden_dim: int = 8):
    """Model that returns per-character deterministic embeddings (rand by position)."""
    torch.manual_seed(42)
    _cache: dict[int, torch.Tensor] = {}

    def forward(**kwargs):
        ids = kwargs.get("input_ids")
        if ids is None:
            ids = next(iter(kwargs.values()))
        n = ids.shape[1]
        if n not in _cache:
            _cache[n] = torch.randn(1, n, hidden_dim)
        out = MagicMock()
        out.last_hidden_state = _cache[n]
        return out

    m = MagicMock(side_effect=forward)
    m.device = torch.device("cpu")
    return m


def _wire(embedder, hidden_dim: int = 8):
    """Attach fake tokenizer + model to embedder."""
    embedder.model = _make_model(hidden_dim)

    def tokenize(text, **kwargs):
        return FakeBatch(text, hidden_dim)

    embedder.tokenizer = MagicMock(side_effect=tokenize)
    # object.__new__ skips __init__, so set these manually
    embedder.max_context_tokens = 8192
    embedder.overlap_tokens = 128
    return embedder


class TestWindowedLateChunking(unittest.TestCase):

    def test_short_doc_single_forward_pass(self):
        """Doc shorter than max_tokens → single forward pass (no windowing)."""
        embedder = _wire(_make_embedder())
        text = "hello world"  # 11 chars = 11 tokens
        spans = [(0, 5), (6, 11)]

        result = embedder.embed_with_late_chunking_windowed(text, spans, max_tokens=50)

        self.assertEqual(len(result), 2)
        # tokenizer called: 1x probe (windowed), 1x guard check, 1x actual encode
        self.assertLessEqual(embedder.tokenizer.call_count, 3)

    def test_long_doc_all_spans_get_non_zero_vector(self):
        """Doc longer than max_tokens → every span gets a non-zero vector."""
        embedder = _wire(_make_embedder())
        text = "a" * 100  # 100 tokens
        spans = [(i * 20, i * 20 + 10) for i in range(5)]

        result = embedder.embed_with_late_chunking_windowed(
            text, spans, max_tokens=30, overlap_tokens=5
        )

        self.assertEqual(len(result), 5)
        for i, vec in enumerate(result):
            self.assertGreater(len(vec), 0, f"span {i} got empty vector")
            self.assertFalse(
                all(v == 0.0 for v in vec),
                f"span {i} got all-zero fallback vector",
            )

    def test_span_at_window_boundary_gets_vector(self):
        """Span straddling a window boundary must land in at least one window."""
        embedder = _wire(_make_embedder())
        text = "x" * 60
        max_tokens, overlap = 20, 5
        # Span centred on the first window boundary
        boundary_char = max_tokens  # token window [0,20) ends at char 20
        boundary_span = (boundary_char - 3, boundary_char + 3)

        result = embedder.embed_with_late_chunking_windowed(
            text, [boundary_span], max_tokens=max_tokens, overlap_tokens=overlap
        )

        self.assertEqual(len(result), 1)
        self.assertFalse(
            all(v == 0.0 for v in result[0]),
            "boundary span got all-zero vector",
        )

    def test_windowed_and_direct_agree_on_short_doc(self):
        """Short doc: windowed result == direct embed_with_late_chunking result."""
        embedder = _wire(_make_embedder())
        text = "parliament debate"
        spans = [(0, 10), (11, 17)]

        direct = embedder.embed_with_late_chunking(text, spans)
        windowed = embedder.embed_with_late_chunking_windowed(text, spans, max_tokens=200)

        self.assertEqual(len(direct), len(windowed))
        for i, (dv, wv) in enumerate(zip(direct, windowed)):
            for d, w in zip(dv, wv):
                self.assertAlmostEqual(d, w, places=5, msg=f"span {i} value mismatch")


if __name__ == "__main__":
    unittest.main()
