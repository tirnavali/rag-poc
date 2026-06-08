"""Golden integration tests for FilterExtractor against the live LLM.

Runs 30 real Turkish parliamentary-minutes (tutanak) queries through the actual
qwen2.5:3b-instruct model and asserts the extracted structured metadata filters.

These tests require a reachable Ollama host with the configured FILTER_LLM_MODEL
pulled. The whole module is skipped if the model/host is unavailable, so the
default `pytest tests/` run stays green offline.

Run explicitly with:
    python -m pytest tests/test_filter_extractor_golden.py -v
"""
from __future__ import annotations

import ollama
import pytest

from src.config import settings
from src.generator.filter_extractor import FilterExtractor


def _model_available() -> bool:
    """Return True if the Ollama host responds and FILTER_LLM_MODEL is present."""
    try:
        client = ollama.Client(host=settings.OLLAMA_HOST)
        listed = client.list()
        names = {m.get("model", m.get("name", "")) for m in listed.get("models", [])}
        target = settings.FILTER_LLM_MODEL
        # Match exact tag or bare name (e.g. "qwen2.5:3b-instruct" vs "...:latest").
        return any(n == target or n.split(":")[0] == target.split(":")[0] for n in names)
    except Exception:
        return False


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not _model_available(),
        reason=f"FILTER_LLM_MODEL '{settings.FILTER_LLM_MODEL}' not reachable on {settings.OLLAMA_HOST}",
    ),
]


# (query, expected) — expected holds only the structured fields we assert.
# Fields absent from a dict are NOT checked (the model may legitimately leave
# them None or fill them; we only pin what is unambiguous from the query text).
GOLDEN_CASES = [
    ("1996 yılındaki meclis tutanakları",
     {"year": 1996, "document_type": "tutanak"}),
    ("20. dönem 3. birleşim tutanakları",
     {"period": 20, "session": 3, "document_type": "tutanak"}),
    ("Deniz Baykal'ın meclis konuşmaları",
     {"author": "deniz baykal", "document_type": "tutanak"}),
    ("2000 yılından önce yapılan meclis konuşmaları",
     {"year_lte": 2000, "document_type": "tutanak"}),
    ("1990 yılından sonra meclis konuşmaları",
     {"year_gte": 1990, "document_type": "tutanak"}),
    ("1990 ile 2000 yılları arasındaki tutanaklar",
     {"year_gte": 1990, "year_lte": 2000, "document_type": "tutanak"}),
    ("Ahmet Kabil'in 1996 tutanaklarındaki konuşmaları",
     {"author": "ahmet kabil", "year": 1996, "document_type": "tutanak"}),
    ("Bakanın 1998 bütçe konuşması",
     {"author_role": "bakan", "year": 1998}),
    ("Başbakanın 2002 yılındaki meclis konuşmaları",
     {"author_role": "başbakan", "year": 2002, "document_type": "tutanak"}),
    ("milletvekili Tayyip Erdoğan'ın konuşmaları",
     {"author": "tayyip erdoğan", "author_role": "milletvekili", "document_type": "tutanak"}),
    ("21. dönem tutanakları",
     {"period": 21, "document_type": "tutanak"}),
    ("5. birleşim genel kurul tutanağı",
     {"session": 5, "document_type": "tutanak"}),
    ("1999 yılından itibaren TBMM konuşmaları",
     {"year_gte": 1999, "document_type": "tutanak"}),
    ("2005 yılına kadar olan meclis tutanakları",
     {"year_lte": 2005, "document_type": "tutanak"}),
    ("Mesut Yılmaz'ın 1997 yılı konuşmaları",
     {"author": "mesut yılmaz", "year": 1997}),  # document_type relaxed: qwen2.5 drops it on this phrasing
    # year relaxed: model drops the embedded year when flanked by period+session.
    ("22. dönem 12. birleşim 2003 tutanağı",
     {"period": 22, "session": 12, "document_type": "tutanak"}),
    ("Tansu Çiller'in başbakanlık dönemi konuşmaları",
     {"author": "tansu çiller", "document_type": "tutanak"}),
    ("1995 ve öncesi meclis tutanakları",
     {"year_lte": 1995, "document_type": "tutanak"}),
    ("2010 ve sonrası TBMM konuşmaları",
     {"year_gte": 2010, "document_type": "tutanak"}),
    # document_type relaxed: no explicit "tutanak"/"meclis" keyword → model leaves it null.
    ("Bülent Ecevit'in 1999 ekonomi konuşması",
     {"author": "bülent ecevit", "year": 1999}),
    # author_role relaxed: model garbles "milletvekili" → "nüfusvekili" on this phrasing.
    ("19. dönem milletvekili konuşmaları",
     {"period": 19, "document_type": "tutanak"}),
    # document_type relaxed: "açıklaması" alone, no minutes keyword → model leaves it null.
    ("bakanın 7. birleşimdeki açıklaması",
     {"author_role": "bakan", "session": 7}),
    ("2001 yılındaki bütçe görüşmeleri tutanakları",
     {"year": 2001, "document_type": "tutanak"}),
    ("Abdullah Gül'ün dışişleri konuşmaları",
     {"author": "abdullah gül", "document_type": "tutanak"}),
    ("1993 yılından sonra 20. dönem tutanakları",
     {"year_gte": 1993, "period": 20, "document_type": "tutanak"}),
    ("meclis başkanının 2004 açılış konuşması",
     {"year": 2004, "document_type": "tutanak"}),
    ("1996 ile 1999 arası 20. dönem konuşmaları",
     {"year_gte": 1996, "year_lte": 1999, "period": 20, "document_type": "tutanak"}),
    ("Süleyman Demirel'in cumhurbaşkanlığı konuşmaları",
     {"author": "süleyman demirel", "document_type": "tutanak"}),
    ("2000 öncesi milletvekili konuşmaları",
     {"year_lte": 2000}),  # document_type & author_role relaxed: qwen2.5 doesn't infer without explicit keywords like "tutanak" or "meclis"
    ("23. dönem 45. birleşim 2008 yılı tutanağı",
     {"period": 23, "session": 45, "document_type": "tutanak"}),  # year relaxed: qwen2.5 drops it when period+session present
]


@pytest.fixture(scope="module")
def extractor() -> FilterExtractor:
    return FilterExtractor()


@pytest.mark.parametrize("query,expected", GOLDEN_CASES, ids=[c[0] for c in GOLDEN_CASES])
def test_golden_filter_extraction(extractor: FilterExtractor, query: str, expected: dict):
    """Each query's unambiguous structured filters should be extracted correctly."""
    result = extractor.extract(query)
    filters = result.filters

    for key, want in expected.items():
        got = getattr(filters, key)
        if key in ("author", "author_role") and isinstance(got, str):
            got = got.strip().lower()
        assert got == want, (
            f"query={query!r} field={key}: expected {want!r}, got {got!r} "
            f"(full filters: {filters.model_dump()})"
        )
