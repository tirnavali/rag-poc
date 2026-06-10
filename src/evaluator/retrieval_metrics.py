"""Retrieval quality metrics.

BEGINNER GUIDE — what do these numbers mean?
---------------------------------------------
When we evaluate a search/retrieval system, we ask: "Given a query, did the
system find the right documents?"

We assume we have a *ground truth*: for each test query, we know which
archive records (identified by kayit_no) are the correct answers.

The system returns a ranked list of results. We then compare:
  - retrieved_ids: what the system found (ordered, best first)
  - relevant_ids:  what the correct answer is (a set, order doesn't matter)

Key insight: position matters. Finding the right document at rank 1 is much
better than finding it at rank 50.
"""
from __future__ import annotations

import math


def precision_at_k(retrieved_ids: list, relevant_ids: set, k: int) -> float:
    """Fraction of the top-k results that are actually correct.

    Intuition: "How much of what I returned was useful?"
    Range: 0.0 (nothing useful) → 1.0 (everything was perfect)

    Example: retrieved=[A, B, C, D, E], relevant={A, C}, k=5
      → hits = 2 (A and C), precision@5 = 2/5 = 0.40

    Good target: P@10 ≥ 0.5 for an archive system.
    Note: precision drops as k grows because the denominator (k) grows
    but correct docs stay the same.
    """
    if not retrieved_ids or k == 0:
        return 0.0
    top_k = retrieved_ids[:k]
    hits = sum(1 for rid in top_k if rid in relevant_ids)
    return hits / k


def recall_at_k(retrieved_ids: list, relevant_ids: set, k: int) -> float:
    """Fraction of all correct documents that appear in the top-k results.

    Intuition: "How many of the right answers did I manage to find?"
    Range: 0.0 (missed everything) → 1.0 (found all correct docs)

    Example: retrieved=[A, B, C, D, E], relevant={A, C, F}, k=5
      → hits = 2 (A and C found; F was missed), recall@5 = 2/3 = 0.67

    For an archive with 1 known-correct record per query, recall@k is
    basically the same as hit_rate@k.
    """
    if not relevant_ids:
        # No ground truth provided — can't compute recall
        return 0.0
    top_k = retrieved_ids[:k]
    hits = sum(1 for rid in top_k if rid in relevant_ids)
    return hits / len(relevant_ids)


def hit_rate_at_k(retrieved_ids: list, relevant_ids: set, k: int) -> float:
    """1.0 if at least one correct document appears in the top-k results, else 0.0.

    Intuition: "Did the system find anything useful at all in the top k?"
    This is a binary yes/no — useful when there's only one correct answer.

    Example: retrieved=[B, C, A, D, E], relevant={A}, k=5
      → A is at position 3, still within k=5 → hit_rate@5 = 1.0

    Hit Rate@10 ≥ 0.8 is a reasonable minimum bar for a working system.
    """
    top_k = retrieved_ids[:k]
    return 1.0 if any(rid in relevant_ids for rid in top_k) else 0.0


def mrr(retrieved_ids: list, relevant_ids: set) -> float:
    """Mean Reciprocal Rank — rewards finding the correct doc as early as possible.

    Intuition: "How quickly does the system surface the right answer?"
    Formula: 1 / rank_of_first_correct_document

    Examples:
      → correct doc at rank 1: MRR = 1/1 = 1.0  (perfect)
      → correct doc at rank 2: MRR = 1/2 = 0.5
      → correct doc at rank 5: MRR = 1/5 = 0.2
      → correct doc not found: MRR = 0.0

    For a "known item" search (user looks for one specific record), MRR is
    the single most important metric. Target: MRR ≥ 0.5.
    """
    for rank, rid in enumerate(retrieved_ids, 1):
        if rid in relevant_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_ids: list, relevant_ids: set, k: int) -> float:
    """Normalized Discounted Cumulative Gain — like MRR but considers all correct docs.

    Intuition: rewards a ranking where all correct docs appear early and
    penalizes putting them late. Assumes binary relevance (a doc is either
    correct or not — no partial credit).

    Range: 0.0 → 1.0. 1.0 means all correct docs appeared before any
    incorrect docs, in the best possible order.

    NDCG is most useful when there are multiple correct documents per query.
    For single-correct-answer queries, MRR is simpler and equivalent.
    """
    def dcg(hits: list) -> float:
        return sum(rel / math.log2(i + 2) for i, rel in enumerate(hits))

    top_k = retrieved_ids[:k]
    gains = [1 if rid in relevant_ids else 0 for rid in top_k]
    n_rel = min(k, len(relevant_ids))
    ideal = [1] * n_rel + [0] * (k - n_rel)
    actual_dcg = dcg(gains)
    ideal_dcg = dcg(ideal)
    return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0


def source_routing_accuracy(is_minutes: bool, expected_source_db: str) -> float:
    """1.0 if the system routed to the correct data source, else 0.0.

    The system has two sources: "gazete" (newspaper archive) and "minutes"
    (parliament transcripts). Routing is done by keyword matching in the query.

    A wrong routing is bad: if the user asks about a parliamentary debate
    but the system searches the newspaper archive, it will miss everything.

    is_minutes=True  → system searched parliament minutes
    is_minutes=False → system searched newspaper archive (or both)

    Expected values for expected_source_db: "gazete", "minutes", or "both".
    If "both" is expected, either routing is acceptable.
    """
    if expected_source_db == "minutes":
        return 1.0 if is_minutes else 0.0
    if expected_source_db == "gazete":
        return 1.0 if not is_minutes else 0.0
    # "both" expected — routing to either source is fine
    return 1.0


def date_filter_accuracy(parsed_dates: dict, expected_year: int | None) -> float:
    """1.0 if the system correctly parsed the expected year from the query, else 0.0.

    When a query mentions a date (e.g. "07.04.1998", "3 Mart 2000", "1998"),
    the system extracts the year and uses it to filter results. This metric
    checks that the extraction worked correctly.

    If no expected_year is provided (None), accuracy is 1.0 — the query
    doesn't contain a date so there's nothing to check.

    parsed_dates is a dict with keys:
      "years":       list of bare years found ("1998", "2001")
      "exact_dates": list of ISO date strings ("1998-04-07")
    """
    if expected_year is None:
        # Query has no date constraint — date parsing is irrelevant
        return 1.0
    years = [int(y) for y in parsed_dates.get("years", [])]
    for d in parsed_dates.get("exact_dates", []):
        try:
            years.append(int(d[:4]))
        except ValueError:
            pass
    return 1.0 if expected_year in years else 0.0


def context_coverage(relevant_ids: set, included_ids: set) -> float:
    """Fraction of relevant documents that survived the context-building filter.

    This is the BRIDGE between Layer 1 (retrieval) and Layer 3 (generation).
    A document can be retrieved successfully but then silently dropped by
    build_context() if its distance score is above the threshold — meaning
    the LLM never actually sees it.

    relevant_ids:  set of kayit_no values we expect the system to find
    included_ids:  set of kayit_no values that survived into the context
                   (computed by context_included_ids() in context.py)

    Interpretation:
      1.0 → all relevant docs reached the LLM — full pipeline working ✓
      0.0 → correct docs retrieved but all filtered out — distance threshold too strict
      0.5 → half the relevant docs reached the LLM — partial success

    If P@10 > 0 but context_coverage = 0, your distance threshold is dropping
    the right documents before the LLM can use them.
    """
    if not relevant_ids:
        return 1.0  # no ground truth — nothing to check
    hits = len(relevant_ids & included_ids)
    return hits / len(relevant_ids)
