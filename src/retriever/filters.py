"""Opt-in query-text-derived date filtering.

VectorRetriever no longer does this automatically — it is a pure vector-query
layer. Callers who want a Chroma `where` filter derived from bare years/dates
found in the query text call `auto_date_where_filter()` explicitly and pass
the result in.
"""
from __future__ import annotations

from typing import Optional

from src.common.chroma import where_year_filter
from src.common.dates import extract_dates


def auto_date_where_filter(query: str) -> tuple[Optional[dict], dict]:
    """Regex-extract years/dates from query text and build a Chroma where filter.

    Returns:
        (where_filter, parsed_dates) — where_filter is None if no years found;
        parsed_dates is {"years": [...], "exact_dates": [...]}.
    """
    parsed_dates = extract_dates(query)
    years = parsed_dates.get("years", [])
    exact_dates = parsed_dates.get("exact_dates", [])
    year_from_exact = [int(d[:4]) for d in exact_dates if d]
    all_years = list(set([int(y) for y in years] + year_from_exact))
    where_filter = where_year_filter(all_years)
    return where_filter, parsed_dates
