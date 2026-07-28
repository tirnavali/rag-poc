"""Ingestion Debug Viewer saf fonksiyon birim testleri (offline — Chroma/HTTP/fitz yok).

Analitik mantık HTTP katmanının dışında olduğu için Chroma ya da bir sunucu
kurmadan test edilir: sayfa↔chunk eşleme (çok-sayfalı '4, 5' string parse'ı dahil),
belge-içi sıralama ve proxy kalite sinyalleri.
"""
from scripts.ingestion_viewer import (
    build_overview,
    chunks_for_page,
    page_quality_signals,
    pages_of_chunk,
    _has_value,
)


def _chunk(cid, text, meta):
    return {"chunk_id": cid, "text": text, "metadata": meta}


# --------------------------------------------------------------------------- #
# pages_of_chunk — Chroma virgül-string / tekil page / fallback
# --------------------------------------------------------------------------- #
def test_pages_of_chunk_parses_comma_string():
    assert pages_of_chunk({"pages": "4, 5"}) == [4, 5]


def test_pages_of_chunk_falls_back_to_page_int():
    assert pages_of_chunk({"page": 7}) == [7]
    assert pages_of_chunk({"page": "7"}) == [7]


def test_pages_of_chunk_prefers_pages_over_page():
    assert pages_of_chunk({"pages": "4, 5", "page": 4}) == [4, 5]


def test_pages_of_chunk_empty_when_absent():
    assert pages_of_chunk({}) == []
    assert pages_of_chunk({"page": None}) == []


# --------------------------------------------------------------------------- #
# chunks_for_page — filtreleme + sıralama + çok-sayfalı chunk iki sayfada görünür
# --------------------------------------------------------------------------- #
def test_chunks_for_page_filters_and_orders_by_index():
    chunks = [
        _chunk("doc_2", "c2", {"page": 5}),
        _chunk("doc_0", "c0", {"page": 4}),
        _chunk("doc_1", "c1", {"pages": "4, 5"}),
    ]
    # sayfa 4: doc_0 ve doc_1 (index sırasıyla)
    got = chunks_for_page(chunks, 4)
    assert [c["chunk_id"] for c in got] == ["doc_0", "doc_1"]


def test_chunks_for_page_multipage_chunk_appears_on_both_pages():
    chunks = [_chunk("doc_1", "spans", {"pages": "4, 5"})]
    assert [c["chunk_id"] for c in chunks_for_page(chunks, 4)] == ["doc_1"]
    assert [c["chunk_id"] for c in chunks_for_page(chunks, 5)] == ["doc_1"]
    assert chunks_for_page(chunks, 6) == []


def test_chunks_for_page_tolerates_str_page_arg():
    chunks = [_chunk("doc_0", "c0", {"page": 3})]
    assert len(chunks_for_page(chunks, "3")) == 1
    assert chunks_for_page(chunks, "x") == []


# --------------------------------------------------------------------------- #
# _has_value — etiket dolu mu (0/''/None boş sayılır)
# --------------------------------------------------------------------------- #
def test_has_value_semantics():
    assert _has_value({"sira_sayisi": 5}, "sira_sayisi") is True
    assert _has_value({"esas_no": "2/773"}, "esas_no") is True
    assert _has_value({"sira_sayisi": 0}, "sira_sayisi") is False
    assert _has_value({"sira_sayisi": None}, "sira_sayisi") is False
    assert _has_value({"esas_no": ""}, "esas_no") is False
    assert _has_value({}, "sira_sayisi") is False


# --------------------------------------------------------------------------- #
# page_quality_signals — proxy sinyaller
# --------------------------------------------------------------------------- #
def test_signals_empty_page_flagged_bad():
    sig = page_quality_signals("", [])
    assert sig["char_count"] == 0
    assert sig["levels"]["char"] == "bad"
    assert sig["worst_level"] == "bad"
    assert any("boş" in f for f in sig["flags"])


def test_signals_clean_text_ok():
    text = (
        "Türkiye Büyük Millet Meclisi Genel Kurulu bugün toplanarak gündemindeki "
        "kanun tekliflerini görüşmeye başladı ve oturum başkanı söz verdi. " * 6
    )
    chunks = [_chunk("doc_0", text, {"page": 1})]
    sig = page_quality_signals(text, chunks)
    assert sig["levels"]["char"] == "ok"
    assert sig["chunk_count"] == 1
    assert sig["worst_level"] == "ok"
    assert sig["word_count"] > 0


def test_signals_counts_tagged_chunks():
    text = "x" * 400
    chunks = [
        _chunk("doc_0", "a", {"page": 2, "sira_sayisi": 5, "esas_no": "2/773"}),
        _chunk("doc_1", "b", {"page": 2, "sira_sayisi": 5}),
        _chunk("doc_2", "c", {"page": 2}),  # etiketsiz
    ]
    sig = page_quality_signals(text, chunks)
    assert sig["tagged_chunks"] == 2
    assert sig["esas_chunks"] == 1
    assert sig["chunk_count"] == 3


def test_signals_sparse_text_with_no_chunks_warns():
    # metin var (chunk beklenirdi) ama sayfaya değen chunk yok → chunk uyarısı
    sig = page_quality_signals("x" * 500, [])
    assert sig["levels"]["chunks"] == "warn"
    assert any("chunk yok" in f for f in sig["flags"])


# --------------------------------------------------------------------------- #
# build_overview — sayfa özetleri
# --------------------------------------------------------------------------- #
def test_build_overview_summarizes_each_page():
    pages = [
        {"sayfa_no": 1, "sayfa_markdown": "x" * 500},
        {"sayfa_no": 2, "sayfa_markdown": ""},
    ]
    chunks = [
        _chunk("doc_0", "a", {"page": 1, "sira_sayisi": 5}),
        _chunk("doc_1", "b", {"page": 1}),
    ]
    ov = build_overview(pages, chunks)
    assert [p["sayfa_no"] for p in ov] == [1, 2]
    assert ov[0]["chunk_count"] == 2
    assert ov[0]["tagged_chunks"] == 1
    assert ov[1]["char_count"] == 0
    assert ov[1]["worst_level"] == "bad"  # boş sayfa
