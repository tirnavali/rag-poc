import pytest
from src.common.dates import extract_dates, normalize_iso_date, extract_year


def test_iso_date():
    r = extract_dates("2026-04-07 tarihli yazı")
    assert "2026-04-07" in r["exact_dates"]
    assert r["years"] == []


def test_dotted_date():
    r = extract_dates("07.04.2026 tarihli")
    assert "2026-04-07" in r["exact_dates"]


def test_turkish_text_date():
    r = extract_dates("7 Nisan 2026 tarihli")
    assert "2026-04-07" in r["exact_dates"]


def test_bare_year():
    r = extract_dates("1997 yılındaki yazı")
    assert "1997" in r["years"]
    assert r["exact_dates"] == []


def test_year_inside_full_date_not_duplicated():
    r = extract_dates("2026-04-07 yazısı")
    assert "2026" not in r["years"]


def test_normalize_iso_date():
    assert normalize_iso_date("2026-04-07T10:00:00") == "2026-04-07"
    assert normalize_iso_date(None) == ""
    assert normalize_iso_date("") == ""


def test_extract_year():
    assert extract_year("2026-04-07") == 2026
    assert extract_year("") == 0
    assert extract_year("abcd") == 0
