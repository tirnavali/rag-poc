"""Build an LLM-ready context string from ranked retrieval results."""
from __future__ import annotations

from src.common.protocols import RetrievalResult
from src.config import settings


def build_context(
    results: RetrievalResult,
    max_chars: int = settings.CONTEXT_BUILD_DEFAULT_MAX,
    distance_threshold: float = settings.DISTANCE_THRESHOLD,
    total_max_chars: int = settings.CONTEXT_BUILD_DEFAULT_TOTAL,
) -> str:
    """Concatenate ranked document chunks into a single context string.

    Chunks whose distance exceeds ``distance_threshold`` are skipped.
    Accumulation stops when ``total_max_chars`` would be exceeded.
    """
    context_list: list[str] = []
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    dists = results.get("distances", [[]])[0]

    current_total = 0
    for doc, meta, dist in zip(docs, metas, dists):
        if dist is not None and dist > distance_threshold:
            continue
        parts = []
        source = meta.get("source_name") or ""
        date = meta.get("date") or ""
        period = meta.get("period")
        author = meta.get("author") or ""
        header_parts = [p for p in [source, date, f"Dönem {period}" if period else "", author] if p]
        if header_parts:
            parts.append("[" + " | ".join(header_parts) + "]")
        parts.append(doc)
        chunk_text = "\n".join(parts)[:max_chars]
        if context_list and current_total + len(chunk_text) > total_max_chars:
            break
        context_list.append(chunk_text)
        current_total += len(chunk_text)

    return "\n\n---\n\n".join(context_list)


def build_structured_context(
    results: RetrievalResult,
    max_chars: int = settings.CONTEXT_BUILD_DEFAULT_MAX,
    distance_threshold: float = settings.DISTANCE_THRESHOLD,
    total_max_chars: int = settings.CONTEXT_BUILD_DEFAULT_TOTAL,
) -> list[dict]:
    """Mirror build_context() but emit one structured record per surviving chunk.

    Each record carries the provenance fields (source_name, document_id, date, ...)
    plus the truncated excerpt the LLM would see. MCP tools return this
    alongside the prose context so callers (Open WebUI, downstream agents) can
    render citations without parsing the concatenated string.
    """
    items: list[dict] = []
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    dists = results.get("distances", [[]])[0]

    current_total = 0
    for doc, meta, dist in zip(docs, metas, dists):
        if dist is not None and dist > distance_threshold:
            continue
        chunk_text = doc[:max_chars]
        if items and current_total + len(chunk_text) > total_max_chars:
            break
        # Strip the human-readable header line the retriever prepends; the
        # structured fields below already carry the same metadata.
        excerpt = chunk_text.split("\n", 1)[1] if "\n" in chunk_text else chunk_text
        items.append({
            "source_name": meta.get("source_name"),
            "document_id": meta.get("document_id"),
            "publication": meta.get("publication"),
            "date": meta.get("date"),
            "author": meta.get("author"),
            "title": meta.get("title"),
            "topics": meta.get("topics") or None,
            "excerpt": excerpt.strip(),
            "distance": round(dist, 4) if dist is not None else None,
        })
        current_total += len(chunk_text)

    return items


def context_included_ids(
    results: RetrievalResult,
    distance_threshold: float = settings.DISTANCE_THRESHOLD,
) -> set[int]:
    """Return the set of kayit_no values that would survive into the context.

    This mirrors the distance filter in build_context() without building the
    full string. Use it in the evaluator to check whether retrieved relevant
    documents actually reach the LLM — a document can be retrieved (Layer 1)
    but still dropped here if its distance score is too high (Layer 2 failure).

    Example: retrieved_ids=[324, 500] but context_included_ids={500} means
    KAYIT_NO 324 was retrieved but silently filtered out before the LLM saw it.
    """
    included: set[str] = set()
    metas = results.get("metadatas", [[]])[0]
    dists = results.get("distances", [[]])[0]

    for meta, dist in zip(metas, dists):
        if dist is not None and dist > distance_threshold:
            continue
        doc_id = meta.get("document_id")
        if doc_id is not None:
            included.add(str(doc_id))

    return included
