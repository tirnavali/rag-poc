"""Turkish date extraction and normalization."""
from __future__ import annotations

import re

_MONTHS = {
    "oca": "01", "ocak": "01",
    "şub": "02", "şubat": "02", "sub": "02", "subat": "02",
    "mar": "03", "mart": "03",
    "nis": "04", "nisan": "04",
    "may": "05", "mayıs": "05", "mayis": "05",
    "haz": "06", "haziran": "06",
    "tem": "07", "temmuz": "07",
    "ağu": "08", "ağustos": "08", "agu": "08", "agustos": "08",
    "eyl": "09", "eylül": "09", "eylul": "09",
    "eki": "10", "ekim": "10",
    "kas": "11", "kasım": "11", "kasim": "11",
    "ara": "12", "aralık": "12", "aralik": "12",
}


def extract_dates(query: str) -> dict:
    """Return {"exact_dates": [...], "years": [...]} parsed from a Turkish query.

    Supports ISO (2026-04-07), dotted/slashed (07.04.2026, 7/4/2026),
    textual Turkish ("7 Nisan 2026"), and bare 4-digit years (1900-2100).
    """
    exact_dates: list[str] = []
    years: list[str] = []
    full_date_spans: list[tuple[int, int]] = []

    for m in re.finditer(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b", query):
        year, month, day = m.groups()
        exact_dates.append(f"{year}-{int(month):02d}-{int(day):02d}")
        full_date_spans.append(m.span())

    for m in re.finditer(r"\b(\d{1,2})[./-](\d{1,2})[./-](\d{4})\b", query):
        day, month, year = m.groups()
        exact_dates.append(f"{year}-{int(month):02d}-{int(day):02d}")
        full_date_spans.append(m.span())

    month_pattern = "|".join(_MONTHS.keys())
    for m in re.finditer(
        rf"\b(\d{{1,2}})\s+({month_pattern})\s+(\d{{4}})\b",
        query,
        flags=re.IGNORECASE,
    ):
        day, month_str, year = m.groups()
        mo = _MONTHS[month_str.lower()]
        exact_dates.append(f"{year}-{mo}-{int(day):02d}")
        full_date_spans.append(m.span())

    for m in re.finditer(r"\b(\d{4})\b", query):
        year_int = int(m.group(1))
        if 1900 <= year_int <= 2100:
            span = m.span()
            if not any(s <= span[0] and span[1] <= e for s, e in full_date_spans):
                years.append(str(year_int))

    return {"exact_dates": list(set(exact_dates)), "years": list(set(years))}


def normalize_iso_date(raw) -> str:
    """Return the YYYY-MM-DD prefix of any date-like value, or ""."""
    return str(raw or "")[:10]


def extract_year(iso_date: str) -> int:
    """Return the year int from a YYYY-MM-DD string, or 0 if malformed."""
    try:
        return int(iso_date[:4]) if len(iso_date) >= 4 else 0
    except ValueError:
        return 0
