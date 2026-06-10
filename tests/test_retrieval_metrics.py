"""Unit tests for retrieval quality metrics."""
import pytest
from src.evaluator.retrieval_metrics import (
    precision_at_k,
    recall_at_k,
    hit_rate_at_k,
    mrr,
    ndcg_at_k,
)

def test_precision_at_k():
    assert precision_at_k(["A", "B", "C"], {"A", "C"}, k=3) == 2.0 / 3.0
    assert precision_at_k(["A", "B", "C"], {"A", "C"}, k=1) == 1.0
    assert precision_at_k(["A", "B", "C"], {"A", "C"}, k=2) == 0.5
    assert precision_at_k([], {"A"}, k=5) == 0.0
    assert precision_at_k(["A"], {"A"}, k=0) == 0.0

def test_recall_at_k():
    assert recall_at_k(["A", "B", "C"], {"A", "C", "D"}, k=3) == 2.0 / 3.0
    assert recall_at_k(["A", "B", "C"], {"A", "C", "D"}, k=1) == 1.0 / 3.0
    assert recall_at_k(["A", "B", "C"], set(), k=3) == 0.0

def test_hit_rate_at_k():
    assert hit_rate_at_k(["A", "B", "C"], {"C"}, k=3) == 1.0
    assert hit_rate_at_k(["A", "B", "C"], {"C"}, k=2) == 0.0
    assert hit_rate_at_k([], {"A"}, k=5) == 0.0

def test_mrr():
    assert mrr(["A", "B", "C"], {"C"}) == 1.0 / 3.0
    assert mrr(["A", "B", "C"], {"B", "C"}) == 1.0 / 2.0
    assert mrr(["A", "B", "C"], {"D"}) == 0.0
    assert mrr([], {"A"}) == 0.0

def test_ndcg_at_k_absolute_vs_relative():
    # 5 relevant documents in database
    relevant = {"A", "B", "C", "D", "E"}
    
    # System retrieved only 1, and placed it at rank 1.
    # DCG = 1 / log2(2) = 1.0
    # IDCG (ideal) = 5 relevant documents, k=10, so ideal contains min(10, 5) = 5 ones:
    # [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
    # ideal_dcg = 1/log2(2) + 1/log2(3) + 1/log2(4) + 1/log2(5) + 1/log2(6)
    #           = 1 + 0.6309297535714574 + 0.5 + 0.43067655807339306 + 0.38685280723454163
    #           = 2.948459118879392
    # ndcg should be 1.0 / 2.948459118879392 = 0.33916021111666756
    
    val = ndcg_at_k(["A", "X", "Y", "Z"], relevant, k=10)
    assert abs(val - 0.339160211116) < 1e-6

    # If all 5 retrieved and placed at top: NDCG should be 1.0
    val_perfect = ndcg_at_k(["A", "B", "C", "D", "E", "X"], relevant, k=10)
    assert abs(val_perfect - 1.0) < 1e-6
    
    # If no relevant items retrieved
    assert ndcg_at_k(["X", "Y"], relevant, k=5) == 0.0
