"""VLM tablo çıkarıcı saf-mantık testleri (Ollama/Docling gerektirmez, offline).

Kapsanan:
  1. _parse_osd — Tesseract OSD çıktısından açı/güven ayıklama
  2. _looks_like_garbage / table_is_low_quality — bozuk tablo tetikleyicisi
     (örnek belgedeki 8 bozuk tablonun gerçek metrik profillerine göre)
  3. _stitch_markdown_bands — bant markdown'larının başlık-strip + dedup ile birleşimi
  4. _split_into_bands — yükseklik eşiğine göre bantlama
"""
from __future__ import annotations

from PIL import Image

from src.common.parsing.vlm_table_extractor import (
    _is_degenerate,
    _looks_like_garbage,
    _parse_osd,
    _split_into_bands,
    _stitch_markdown_bands,
    table_is_low_quality,
)


# ---------------------------------------------------------------- OSD parse

def test_parse_osd_extracts_rotate_and_confidence():
    osd = "Page number: 0\nOrientation in degrees: 270\nRotate: 90\nOrientation confidence: 4.50\n"
    assert _parse_osd(osd) == (90, 4.50)


def test_parse_osd_handles_missing_fields():
    assert _parse_osd("garbage output, no numbers") == (None, None)


# ------------------------------------------------------- garbage / trigger

# Örnek belgedeki bozuk OCR desenleri (gerçek atom metinlerinden örneklenmiş).
_GARBAGE_SINGLECHAR = "| 1 |  |\n| --- | --- |\n| ğ | 5 1 1 1 |\n| 1 | 8 |\n|  | 1 |"
_GARBAGE_FRAGMENTED = "|  | 1 9 9 |\n| --- | --- |\n| 1 | 9 1 |\n| 1 | 1 8 1 1 |\n| J | 8 1 5 |"
_GARBAGE_MOJIBAKE = (
    "| HIZMETLERI |  |  |  |\n| --- | --- | --- | --- |\n"
    "| KMU DUZENI VE HEZMETLERI |  |  |  |\n| DINLENNE KUZTUR |  |  |  |"
)

_GOOD_TABLE = (
    "| KURUMLAR | PERSONEL GİDERLERİ | TOPLAM |\n| --- | --- | --- |\n"
    "| ÖLÇME SEÇME VE YERLEŞTİRME MERKEZİ | 474.600.000 | 717.792.000 |\n"
    "| TÜRK DİL KURUMU | 7.477.000 | 19.693.000 |\n"
    "| ATATÜRK KÜLTÜR MERKEZİ | 3.977.000 | 7.408.000 |"
)
# Meşru "0" hücreleri olan dolu tablo yanlış tetiklenmemeli.
_GOOD_ZERO_HEAVY = (
    "| A | B | C | D | TOPLAM |\n| --- | --- | --- | --- | --- |\n"
    "| KURUM BİR | 0 | 0 | 5.000.000 | 5.000.000 |\n"
    "| KURUM İKİ | 0 | 1.000.000 | 0 | 1.000.000 |\n"
    "| KURUM ÜÇ | 250.000 | 0 | 0 | 250.000 |"
)


def test_garbage_singlechar_flagged():
    assert _looks_like_garbage(_GARBAGE_SINGLECHAR, 5, 2) is True


def test_garbage_fragmented_digits_flagged():
    assert _looks_like_garbage(_GARBAGE_FRAGMENTED, 8, 2) is True


def test_garbage_mojibake_mostly_empty_flagged():
    # Büyük ızgara, çoğu hücre boş → düşük doluluk oranıyla yakalanır.
    assert _looks_like_garbage(_GARBAGE_MOJIBAKE, 58, 7) is True


def test_good_table_not_flagged():
    assert _looks_like_garbage(_GOOD_TABLE, 5, 3) is False


def test_good_zero_heavy_table_not_flagged():
    assert _looks_like_garbage(_GOOD_ZERO_HEAVY, 4, 5) is False


def test_table_is_low_quality_zero_dims():
    assert table_is_low_quality({"label": "table", "table_num_rows": 0, "table_num_cols": 0}) is True


def test_table_is_low_quality_skips_already_vlm():
    atom = {"label": "table", "extracted_by": "vlm", "table_num_rows": 0, "table_num_cols": 0}
    assert table_is_low_quality(atom) is False


def test_table_is_low_quality_ignores_non_tables():
    assert table_is_low_quality({"label": "text", "text": "| ğ |"}) is False


def test_table_is_low_quality_good_table_false():
    atom = {"label": "table", "table_num_rows": 5, "table_num_cols": 3, "text": _GOOD_TABLE}
    assert table_is_low_quality(atom) is False


# ------------------------------------------------------- degenerate output

def test_degenerate_repetition_loop_detected():
    # 7b tekrar döngüsü: bir satırda yüzlerce sütun.
    loop = "| KURUMLAR | " + "GELDERSİ | " * 300
    md = "| A | B | TOPLAM |\n| --- | --- | --- |\n" + loop + "\n| X | 1 | 2 |"
    assert _is_degenerate(md) is True


def test_degenerate_repeated_token_detected():
    md = "## " + ("BÜTÇE " * 60) + "\n| A | B |\n| --- | --- |\n| x | 1 |"
    assert _is_degenerate(md) is True


def test_clean_table_not_degenerate():
    assert _is_degenerate(_GOOD_TABLE) is False


def test_none_not_degenerate():
    assert _is_degenerate(None) is False


# --------------------------------------------------------------- stitching

def test_stitch_strips_repeated_headers_from_later_bands():
    band1 = "| A | B |\n| --- | --- |\n| x | 1 |\n| y | 2 |"
    band2 = "| A | B |\n| --- | --- |\n| z | 3 |\n| w | 4 |"
    out = _stitch_markdown_bands([band1, band2])
    # Başlık + ayraç yalnızca bir kez kalmalı.
    assert out.count("| --- | --- |") == 1
    assert out.count("| A | B |") == 1
    for cell in ("| x | 1 |", "| y | 2 |", "| z | 3 |", "| w | 4 |"):
        assert cell in out


def test_stitch_dedups_overlap_row():
    band1 = "| A | B |\n| --- | --- |\n| x | 1 |\n| y | 2 |"
    band2 = "| A | B |\n| --- | --- |\n| y | 2 |\n| z | 3 |"  # 'y | 2' örtüşmeden tekrar
    out = _stitch_markdown_bands([band1, band2])
    assert out.count("| y | 2 |") == 1


# ------------------------------------------------------------- band split

def test_split_no_tiling_when_short():
    img = Image.new("RGB", (800, 300), "white")
    assert len(_split_into_bands(img, max_h=400, overlap=50)) == 1


def test_split_tiles_tall_image_with_overlap():
    img = Image.new("RGB", (800, 1000), "white")
    bands = _split_into_bands(img, max_h=400, overlap=50)
    assert len(bands) >= 3
    assert all(b.width == 800 for b in bands)
