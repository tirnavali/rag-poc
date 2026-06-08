"""Span-based retrieval metrics using character-range overlap.

Scores retrieved chunks against golden spans (not chunk_ids).
Hit semantics: any character intersection counts as a hit (threshold=0.0).
Enables chunker-agnostic ground truth (see Natural Questions).
"""
from collections import Counter


def span_overlap(a: dict, b: dict) -> float:
    """Intersection-over-Union of two spans on the same document.

    Args:
        a, b: dicts with 'document_id', 'char_start', 'char_end'

    Returns:
        float: IoU in [0, 1]. 0.0 if different documents or no overlap.
    """
    if a["document_id"] != b["document_id"]:
        return 0.0

    lo = max(a["char_start"], b["char_start"])
    hi = min(a["char_end"], b["char_end"])
    inter = max(0, hi - lo)

    if inter == 0:
        return 0.0

    union = (a["char_end"] - a["char_start"]) + (b["char_end"] - b["char_start"]) - inter
    return inter / union if union > 0 else 0.0


def is_hit(retrieved: dict, golden_list: list[dict], threshold: float = 0.0) -> bool:
    """Check if retrieved span overlaps ANY golden span beyond threshold.

    Args:
        retrieved: dict with document_id, char_start, char_end
        golden_list: list of golden span dicts
        threshold: min IoU to count as hit (default 0.0 = any intersection)

    Returns:
        bool: True if any overlap > threshold
    """
    return any(span_overlap(retrieved, g) > threshold for g in golden_list)


def precision_at_k_span(
    retrieved: list[dict], golden: list[dict], k: int, threshold: float = 0.0
) -> float:
    """Fraction of top-k retrieved spans that hit a golden span.

    Args:
        retrieved: list of retrieved span dicts
        golden: list of golden span dicts
        k: cutoff (only evaluate top-k)
        threshold: min IoU to count as hit

    Returns:
        float: precision@k in [0, 1]
    """
    if not retrieved or k == 0:
        return 0.0
    hits = sum(is_hit(r, golden, threshold) for r in retrieved[:k])
    return hits / k


def recall_at_k_span(
    retrieved: list[dict], golden: list[dict], k: int, threshold: float = 0.0
) -> float:
    """Fraction of golden spans found in top-k retrieved spans.

    Args:
        retrieved: list of retrieved span dicts
        golden: list of golden span dicts
        k: cutoff
        threshold: min IoU to count as hit

    Returns:
        float: recall@k in [0, 1]
    """
    if not golden:
        return 0.0
    found = sum(
        1
        for g in golden
        if any(span_overlap(r, g) > threshold for r in retrieved[:k])
    )
    return found / len(golden)


def mrr_span(
    retrieved: list[dict], golden: list[dict], threshold: float = 0.0
) -> float:
    """Mean reciprocal rank: 1/rank of first hit.

    Args:
        retrieved: list of retrieved span dicts
        golden: list of golden span dicts
        threshold: min IoU to count as hit

    Returns:
        float: MRR in [0, 1]. 0.0 if no hit found.
    """
    for rank, r in enumerate(retrieved, 1):
        if is_hit(r, golden, threshold):
            return 1.0 / rank
    return 0.0


def evidence_coverage_at_k(
    retrieved: list[dict], golden: list[dict], k: int, threshold: float = 0.0
) -> float:
    """Fraction of golden span characters covered by the top-k retrieved spans.

    Captures the 'tiny retrieved chunk barely touches a long golden span'
    failure mode that recall/precision miss. Uses interval union to avoid
    double-counting overlapping retrieved chunks.

    Args:
        retrieved: list of retrieved span dicts
        golden: list of golden span dicts
        k: cutoff (only evaluate top-k)
        threshold: unused (for API consistency); always uses any overlap

    Returns:
        float in [0, 1]. 0.0 if no golden spans.
    """
    if not golden:
        return 0.0

    total_golden = sum(g["char_end"] - g["char_start"] for g in golden)
    if total_golden == 0:
        return 0.0

    covered = 0
    for g in golden:
        doc_id = g["document_id"]
        # Collect overlaps from all top-k retrieved spans against this golden span
        intervals = []
        for r in retrieved[:k]:
            if r["document_id"] != doc_id:
                continue
            lo = max(r["char_start"], g["char_start"])
            hi = min(r["char_end"], g["char_end"])
            if hi > lo:
                intervals.append((lo, hi))

        # Union the intervals (handles partially-overlapping retrieved chunks)
        if intervals:
            intervals.sort()
            merged_end = -1
            for lo, hi in intervals:
                if lo > merged_end:
                    covered += hi - lo
                    merged_end = hi
                elif hi > merged_end:
                    covered += hi - merged_end
                    merged_end = hi

    return covered / total_golden


# ------------------------------------------------------------------
# Token-overlap metrikleri (Chroma yöntemi)
# ------------------------------------------------------------------

def _tokenize(text: str) -> Counter:
    """Metni küçük harfe çevirip boşluklara göre böler, token sayaç döndürür."""
    return Counter(text.lower().split())


def token_recall_at_k(retrieved_texts: list[str], gold_excerpts: list[str], k: int) -> float:
    """İlk k sonuçta altın pasaj tokenlarının ne kadarı bulundu.

    Args:
        retrieved_texts: Çekilen chunk metinleri (sıralı)
        gold_excerpts: Altın veri alıntıları (verbatim metin listesi)
        k: Değerlendirme eşiği

    Returns:
        [0, 1] aralığında token recall oranı
    """
    gold: Counter = Counter()
    for e in gold_excerpts:
        gold.update(_tokenize(e))
    retr: Counter = Counter()
    for t in retrieved_texts[:k]:
        retr.update(_tokenize(t))
    total = sum(gold.values())
    overlap = sum(min(gold[tok], retr[tok]) for tok in gold)
    return overlap / total if total > 0 else 0.0


def token_precision_at_k(retrieved_texts: list[str], gold_excerpts: list[str], k: int) -> float:
    """İlk k sonuçtaki tokenların ne kadarı altın pasajlarda geçiyor.

    Args:
        retrieved_texts: Çekilen chunk metinleri (sıralı)
        gold_excerpts: Altın veri alıntıları (verbatim metin listesi)
        k: Değerlendirme eşiği

    Returns:
        [0, 1] aralığında token precision oranı
    """
    gold: Counter = Counter()
    for e in gold_excerpts:
        gold.update(_tokenize(e))
    retr: Counter = Counter()
    for t in retrieved_texts[:k]:
        retr.update(_tokenize(t))
    total = sum(retr.values())
    overlap = sum(min(gold[tok], retr[tok]) for tok in gold)
    return overlap / total if total > 0 else 0.0


def token_iou_at_k(retrieved_texts: list[str], gold_excerpts: list[str], k: int) -> float:
    """Token düzeyinde Intersection over Union — Chroma yöntemi.

    Args:
        retrieved_texts: Çekilen chunk metinleri (sıralı)
        gold_excerpts: Altın veri alıntıları (verbatim metin listesi)
        k: Değerlendirme eşiği

    Returns:
        [0, 1] aralığında token IoU
    """
    gold: Counter = Counter()
    for e in gold_excerpts:
        gold.update(_tokenize(e))
    retr: Counter = Counter()
    for t in retrieved_texts[:k]:
        retr.update(_tokenize(t))
    inter = sum(min(gold[tok], retr[tok]) for tok in gold)
    union = sum((gold + retr).values()) - inter
    return inter / union if union > 0 else 0.0
