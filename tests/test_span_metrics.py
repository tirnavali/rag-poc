"""Tests for span-overlap metrics."""

import pytest
from src.evaluator.span_metrics import (
    span_overlap,
    is_hit,
    precision_at_k_span,
    recall_at_k_span,
    mrr_span,
    evidence_coverage_at_k,
)


class TestSpanOverlap:
    def test_perfect_overlap(self):
        a = {"document_id": "d1", "char_start": 0, "char_end": 100}
        b = {"document_id": "d1", "char_start": 0, "char_end": 100}
        assert span_overlap(a, b) == 1.0

    def test_partial_overlap(self):
        a = {"document_id": "d1", "char_start": 0, "char_end": 100}
        b = {"document_id": "d1", "char_start": 50, "char_end": 150}
        # inter = 50, union = 100 + 100 - 50 = 150
        assert span_overlap(a, b) == 50.0 / 150.0

    def test_no_overlap(self):
        a = {"document_id": "d1", "char_start": 0, "char_end": 50}
        b = {"document_id": "d1", "char_start": 100, "char_end": 150}
        assert span_overlap(a, b) == 0.0

    def test_different_documents(self):
        a = {"document_id": "d1", "char_start": 0, "char_end": 100}
        b = {"document_id": "d2", "char_start": 0, "char_end": 100}
        assert span_overlap(a, b) == 0.0

    def test_contained_span(self):
        a = {"document_id": "d1", "char_start": 25, "char_end": 75}
        b = {"document_id": "d1", "char_start": 0, "char_end": 100}
        # inter = 50, union = 50 + 100 - 50 = 100
        assert span_overlap(a, b) == 0.5


class TestIsHit:
    def test_hit_with_intersection(self):
        retrieved = {"document_id": "d1", "char_start": 0, "char_end": 10}
        golden = [{"document_id": "d1", "char_start": 5, "char_end": 15}]
        assert is_hit(retrieved, golden, threshold=0.0) is True

    def test_no_hit_no_overlap(self):
        retrieved = {"document_id": "d1", "char_start": 0, "char_end": 10}
        golden = [{"document_id": "d1", "char_start": 100, "char_end": 110}]
        assert is_hit(retrieved, golden, threshold=0.0) is False

    def test_hit_multiple_golden(self):
        retrieved = {"document_id": "d1", "char_start": 50, "char_end": 150}
        golden = [
            {"document_id": "d1", "char_start": 0, "char_end": 40},  # no hit
            {"document_id": "d1", "char_start": 100, "char_end": 200},  # hit
        ]
        assert is_hit(retrieved, golden, threshold=0.0) is True

    def test_hit_with_threshold(self):
        # IoU = 0.2, threshold 0.1 → hit
        # IoU = 0.2, threshold 0.3 → no hit
        retrieved = {"document_id": "d1", "char_start": 0, "char_end": 100}
        golden = [{"document_id": "d1", "char_start": 80, "char_end": 120}]
        # inter = 20, union = 100 + 40 - 20 = 120, IoU = 20/120 = 0.1667
        assert is_hit(retrieved, golden, threshold=0.1) is True
        assert is_hit(retrieved, golden, threshold=0.2) is False


class TestPrecisionAtKSpan:
    def test_precision_basic(self):
        retrieved = [
            {"document_id": "d1", "char_start": 0, "char_end": 50},  # hit
            {"document_id": "d1", "char_start": 100, "char_end": 150},  # miss
            {"document_id": "d1", "char_start": 200, "char_end": 250},  # miss
        ]
        golden = [{"document_id": "d1", "char_start": 25, "char_end": 75}]
        assert precision_at_k_span(retrieved, golden, 1) == 1.0
        assert precision_at_k_span(retrieved, golden, 2) == 0.5
        assert precision_at_k_span(retrieved, golden, 3) == 1.0 / 3.0

    def test_precision_zero(self):
        retrieved = [
            {"document_id": "d1", "char_start": 0, "char_end": 50},
            {"document_id": "d1", "char_start": 100, "char_end": 150},
        ]
        golden = [{"document_id": "d1", "char_start": 200, "char_end": 250}]
        assert precision_at_k_span(retrieved, golden, 2) == 0.0

    def test_precision_empty_retrieved(self):
        assert precision_at_k_span([], [{"document_id": "d1", "char_start": 0, "char_end": 100}], 5) == 0.0


class TestRecallAtKSpan:
    def test_recall_basic(self):
        retrieved = [
            {"document_id": "d1", "char_start": 0, "char_end": 50},  # hits g1, g2
            {"document_id": "d1", "char_start": 100, "char_end": 150},  # hits g3
        ]
        golden = [
            {"document_id": "d1", "char_start": 25, "char_end": 75},  # hit by r0
            {"document_id": "d1", "char_start": 40, "char_end": 90},  # hit by r0
            {"document_id": "d1", "char_start": 125, "char_end": 175},  # hit by r1
        ]
        assert recall_at_k_span(retrieved, golden, 2) == 1.0

    def test_recall_partial(self):
        retrieved = [
            {"document_id": "d1", "char_start": 0, "char_end": 50},
        ]
        golden = [
            {"document_id": "d1", "char_start": 25, "char_end": 75},  # hit
            {"document_id": "d1", "char_start": 100, "char_end": 150},  # miss
        ]
        assert recall_at_k_span(retrieved, golden, 1) == 0.5

    def test_recall_empty_golden(self):
        retrieved = [{"document_id": "d1", "char_start": 0, "char_end": 50}]
        assert recall_at_k_span(retrieved, [], 1) == 0.0


class TestMRRSpan:
    def test_mrr_first_hit(self):
        retrieved = [
            {"document_id": "d1", "char_start": 0, "char_end": 50},  # hit
            {"document_id": "d1", "char_start": 100, "char_end": 150},
            {"document_id": "d1", "char_start": 200, "char_end": 250},
        ]
        golden = [{"document_id": "d1", "char_start": 25, "char_end": 75}]
        assert mrr_span(retrieved, golden) == 1.0

    def test_mrr_second_hit(self):
        retrieved = [
            {"document_id": "d1", "char_start": 0, "char_end": 50},  # miss
            {"document_id": "d1", "char_start": 100, "char_end": 150},  # hit
            {"document_id": "d1", "char_start": 200, "char_end": 250},
        ]
        golden = [{"document_id": "d1", "char_start": 125, "char_end": 175}]
        assert mrr_span(retrieved, golden) == 0.5

    def test_mrr_no_hit(self):
        retrieved = [
            {"document_id": "d1", "char_start": 0, "char_end": 50},
            {"document_id": "d1", "char_start": 100, "char_end": 150},
        ]
        golden = [{"document_id": "d1", "char_start": 200, "char_end": 250}]
        assert mrr_span(retrieved, golden) == 0.0


class TestEvidenceCoverage:
    def test_full_coverage(self):
        retrieved = [{"document_id": "d1", "char_start": 0, "char_end": 100}]
        golden = [{"document_id": "d1", "char_start": 0, "char_end": 100}]
        assert evidence_coverage_at_k(retrieved, golden, 1) == 1.0

    def test_partial_coverage(self):
        retrieved = [{"document_id": "d1", "char_start": 0, "char_end": 50}]
        golden = [{"document_id": "d1", "char_start": 0, "char_end": 100}]
        assert evidence_coverage_at_k(retrieved, golden, 1) == 0.5

    def test_overlapping_retrieved_no_double_count(self):
        # Two overlapping retrieved chunks covering 100 chars total
        retrieved = [
            {"document_id": "d1", "char_start": 0, "char_end": 60},
            {"document_id": "d1", "char_start": 40, "char_end": 100},
        ]
        golden = [{"document_id": "d1", "char_start": 0, "char_end": 100}]
        assert evidence_coverage_at_k(retrieved, golden, 2) == 1.0

    def test_different_doc_excluded(self):
        # Retrieved from different doc should not count
        retrieved = [{"document_id": "d2", "char_start": 0, "char_end": 100}]
        golden = [{"document_id": "d1", "char_start": 0, "char_end": 100}]
        assert evidence_coverage_at_k(retrieved, golden, 1) == 0.0

    def test_empty_golden(self):
        retrieved = [{"document_id": "d1", "char_start": 0, "char_end": 100}]
        assert evidence_coverage_at_k(retrieved, [], 1) == 0.0

    def test_empty_retrieved(self):
        golden = [{"document_id": "d1", "char_start": 0, "char_end": 100}]
        assert evidence_coverage_at_k([], golden, 1) == 0.0

    def test_k_cutoff(self):
        # Only top-k are evaluated
        retrieved = [
            {"document_id": "d1", "char_start": 0, "char_end": 50},  # partial
            {"document_id": "d1", "char_start": 50, "char_end": 100},  # covers rest
        ]
        golden = [{"document_id": "d1", "char_start": 0, "char_end": 100}]
        assert evidence_coverage_at_k(retrieved, golden, 1) == 0.5  # only top-1 used
        assert evidence_coverage_at_k(retrieved, golden, 2) == 1.0  # both used

    def test_multiple_golden_spans(self):
        # Multiple golden spans, total 200 chars
        retrieved = [
            {"document_id": "d1", "char_start": 0, "char_end": 50},  # covers 50/100 of g1
            {"document_id": "d1", "char_start": 150, "char_end": 200},  # covers all of g2
        ]
        golden = [
            {"document_id": "d1", "char_start": 0, "char_end": 100},  # 100 chars
            {"document_id": "d1", "char_start": 150, "char_end": 200},  # 50 chars
        ]
        assert evidence_coverage_at_k(retrieved, golden, 2) == 100.0 / 150.0
