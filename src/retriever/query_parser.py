"""Source routing: decide which collections to search based on query keywords."""
from __future__ import annotations

from src.config import settings


def route_sources(query_lower: str) -> list[str]:
    """Return a list of source names to search: ["gazete"], ["minutes"], or both.

    Detection is keyword-based on the normalized (lowercased) query. When both
    sets of keywords appear, both sources are searched so nothing is missed.
    """
    is_minutes = any(kw in query_lower for kw in settings.MINUTES_KEYWORDS)
    is_publication = any(kw in query_lower for kw in settings.PUBLICATION_KEYWORDS)

    if is_minutes and not is_publication:
        return ["minutes"]
    if is_publication and not is_minutes:
        return ["gazete"]
    return ["gazete", "minutes"]
