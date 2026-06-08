"""Tests for segment_aware_pack."""
from __future__ import annotations

from src.common.parsing.author_extractor import TaggedAtom
from src.common.parsing.segment_pack import segment_aware_pack


def _ta(text, author="X", role=None, seg=0, cont=False, extra=None):
    return TaggedAtom(
        text=text,
        label="Text",
        author=author,
        author_role=role,
        segment_index=seg,
        is_continuation=cont,
        extra=extra or {},
    )


class TestSegmentPack:
    def test_empty_input(self):
        assert segment_aware_pack([], min_chars=100, max_chars=500) == []

    def test_single_atom(self):
        atoms = [_ta("Hello world", author="A", seg=1)]
        chunks = segment_aware_pack(atoms, min_chars=5, max_chars=100)
        assert len(chunks) == 1
        assert chunks[0]["metadata"]["author"] == "A"
        assert chunks[0]["metadata"]["authors_in_chunk"] == ["A"]
        assert chunks[0]["metadata"]["segment_indices"] == [1]
        assert chunks[0]["metadata"]["starts_mid_segment"] is False

    def test_soft_split_on_author_change(self):
        long_a = "A" * 600
        atoms = [
            _ta(long_a, author="Alice", seg=1),
            _ta("Bob's first line.", author="Bob", seg=2, cont=False),
            _ta("Bob's continuation.", author="Bob", seg=2, cont=True),
        ]
        chunks = segment_aware_pack(atoms, min_chars=500, max_chars=2000)
        assert len(chunks) == 2
        assert chunks[0]["metadata"]["author"] == "Alice"
        assert chunks[1]["metadata"]["author"] == "Bob"

    def test_no_split_when_below_min_chars(self):
        atoms = [
            _ta("Short A.", author="Alice", seg=1),
            _ta("Short B.", author="Bob", seg=2),
        ]
        chunks = segment_aware_pack(atoms, min_chars=500, max_chars=2000)
        # Not enough chars to trigger soft split → packed together
        assert len(chunks) == 1
        assert set(chunks[0]["metadata"]["authors_in_chunk"]) == {"Alice", "Bob"}

    def test_hard_split_at_max(self):
        big = "X" * 1000
        atoms = [
            _ta(big, author="A", seg=1),
            _ta(big, author="A", seg=1, cont=True),
        ]
        chunks = segment_aware_pack(atoms, min_chars=500, max_chars=1200)
        assert len(chunks) == 2

    def test_primary_author_by_char_weight(self):
        atoms = [
            _ta("X" * 100, author="Alice", seg=1),
            _ta("Y" * 800, author="Bob", seg=2),
        ]
        chunks = segment_aware_pack(atoms, min_chars=100000, max_chars=200000)
        assert chunks[0]["metadata"]["author"] == "Bob"
        assert chunks[0]["metadata"]["authors_in_chunk"] == ["Alice", "Bob"]

    def test_starts_mid_segment_when_first_is_continuation(self):
        # Hard-split forces Alice's two atoms into separate chunks → second chunk
        # starts with a continuation atom → starts_mid_segment=True
        atoms = [
            _ta("X" * 600, author="Alice", seg=1, cont=False),
            _ta("Y" * 600, author="Alice", seg=1, cont=True),
            _ta("Bob speaks now.", author="Bob", seg=2, cont=False),
        ]
        chunks = segment_aware_pack(atoms, min_chars=500, max_chars=800)
        mid_chunks = [c for c in chunks if c["metadata"]["starts_mid_segment"]]
        assert len(mid_chunks) >= 1
        for c in mid_chunks:
            assert c["metadata"]["author"] == "Alice"

    def test_continuation_prefix_injection(self):
        atoms = [
            _ta("X" * 600, author="Alice", seg=1, cont=False),
            _ta("Y" * 600, author="Alice", role="başkan", seg=1, cont=True),
        ]
        chunks = segment_aware_pack(
            atoms, min_chars=500, max_chars=1100, inject_continuation_prefix=True
        )
        mid = [c for c in chunks if c["metadata"]["starts_mid_segment"]]
        assert mid
        prefix = mid[0]["metadata"]["continuation_prefix"]
        assert prefix is not None
        assert "Alice" in prefix
        assert mid[0]["text"].startswith(prefix)

    def test_no_prefix_when_disabled(self):
        atoms = [
            _ta("X" * 600, author="Alice", seg=1, cont=False),
            _ta("Y" * 600, author="Alice", seg=1, cont=True),
        ]
        chunks = segment_aware_pack(
            atoms, min_chars=500, max_chars=1100, inject_continuation_prefix=False
        )
        for c in chunks:
            assert c["metadata"]["continuation_prefix"] is None
