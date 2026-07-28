"""Offline testler — paddle_page_extractor (ağ/fitz-render gerektirmeyen saf yollar).

Gerçek :8080 yanıt fixture'ı (numpy gürültüsü temizli) üzerinden ayrıştırma; sayfa
sınıflandırma eşiği; ve monkeypatch'li tam-sayfa yönlendirme (replacement + ordering
+ başarısızlıkta veri-kaybı-yok fallback). Ağ çağrısı yapılmaz.
"""
import json
from pathlib import Path

import pytest

from src.common.parsing import paddle_page_extractor as ppe
from src.common.parsing.vlm_table_extractor import table_is_low_quality

FIXTURES = Path(__file__).parent / "fixtures"
RESP_FIXTURE = FIXTURES / "paddle_page_response_tbmm27026028_396.json"
SCANNED_PDF = FIXTURES / "tbmm27026028_page_396.pdf"


@pytest.fixture
def resp():
    return json.loads(RESP_FIXTURE.read_text(encoding="utf-8"))


# --- Ayrıştırma (parse) ---------------------------------------------------

def test_parse_blocks_labels(resp):
    blocks = ppe.parse_paddle_blocks(resp["json_denoised"])
    labels = [b["label"] for b in blocks]
    assert labels == ["aside_text", "aside_text", "figure_title", "table", "number"]


def test_parse_blocks_content_and_bbox(resp):
    blocks = ppe.parse_paddle_blocks(resp["json_denoised"])
    # sıra sayısı aside_text'te taşınıyor (metadata sinerjisi)
    assert any("Sura Sayisi" in b["content"] or "Sıra Sayısı" in b["content"] for b in blocks)
    table = next(b for b in blocks if b["label"] == "table")
    assert table["content"].startswith("<table>")
    assert "474.600.000" in table["content"]
    assert table["bbox"].startswith("[") and table["bbox"].endswith("]")


def test_parse_angle(resp):
    assert ppe.parse_angle(resp["json_denoised"]) == 270


def test_parse_blocks_empty_string():
    assert ppe.parse_paddle_blocks("") == []
    assert ppe.parse_angle("") is None


# --- Blok -> atom ---------------------------------------------------------

def test_blocks_to_atoms_drops_number_maps_labels(resp):
    blocks = ppe.parse_paddle_blocks(resp["json_denoised"])
    atoms = ppe.blocks_to_atoms(blocks, page_no=396)
    # 'number' elenir → 5 blok → 4 atom
    assert len(atoms) == 4
    labels = [a["label"] for a in atoms]
    assert "table" in labels
    assert labels.count("text") == 2          # iki aside_text
    assert "caption" in labels                # figure_title -> caption
    assert all(a["page"] == 396 and a["pages"] == [396] for a in atoms)
    assert all(a["extracted_by"] == "paddle_page" for a in atoms)


def test_blocks_to_atoms_excludes_furniture_from_body():
    """header/footer/number GÖVDEYE alınmaz (furniture'a gider)."""
    blocks = [
        {"label": "header", "bbox": "[]", "content": "üst bilgi"},
        {"label": "footer", "bbox": "[]", "content": "alt bilgi"},
        {"label": "number", "bbox": "[]", "content": "12"},
        {"label": "text", "bbox": "[]", "content": "gerçek içerik"},
    ]
    atoms = ppe.blocks_to_atoms(blocks, 5)
    assert len(atoms) == 1
    assert atoms[0]["text"] == "gerçek içerik"


def test_blocks_to_furniture_captures_header_footer_number():
    """Docling FURNITURE hizası: header→page_header; footer+number→page_footer."""
    blocks = [
        {"label": "header", "bbox": "[]", "content": "TBMM B:6 O:5"},
        {"label": "footer", "bbox": "[]", "content": "Sıra Sayısı: 129"},
        {"label": "number", "bbox": "[]", "content": "12"},
        {"label": "text", "bbox": "[]", "content": "gövde"},
    ]
    furn = ppe.blocks_to_furniture(blocks)
    assert furn["page_header"] == "TBMM B:6 O:5"
    assert "Sıra Sayısı: 129" in furn["page_footer"]
    assert "12" in furn["page_footer"]          # number → footer


def test_furniture_role_content_pattern_reclassifies_running_head():
    """Paddle running-head'i doc_title/text sansa da içerik-deseniyle furniture olur."""
    # DÖNEM/CİLT/YASAMA YILI running-head → header (etiket doc_title olsa bile)
    assert ppe._furniture_role(
        {"label": "doc_title", "content": "DÖNEM: 27 | CİLT: 1 | YASAMA YILI: 1"}
    ) == "header"
    # "B: 6 ... O: 5" → header
    assert ppe._furniture_role(
        {"label": "text", "content": "TBMM  B: 6  10.10.2018  O: 5"}
    ) == "header"
    # çıplak sayfa numarası (text sanılmış) → footer
    assert ppe._furniture_role({"label": "text", "content": "45"}) == "footer"
    # gerçek gövde (uzun) → None, 'DÖNEM' geçse bile gövdede kalır
    long_body = "Bu dönem: içinde görüşülen kanun teklifi hakkında " * 4
    assert ppe._furniture_role({"label": "text", "content": long_body}) is None
    # belirsiz/kısa gerçek başlık → None (yanlış-pozitif yok)
    assert ppe._furniture_role({"label": "doc_title", "content": "Türkiye Büyük Millet Meclisi"}) is None


def test_blocks_to_atoms_reclassifies_running_head_out_of_body():
    blocks = [
        {"label": "doc_title", "bbox": "[]", "content": "DÖNEM: 27 | CİLT: 1 | YASAMA YILI: 1"},
        {"label": "text", "bbox": "[]", "content": "Gerçek gövde metni."},
    ]
    atoms = ppe.blocks_to_atoms(blocks, 3)
    assert [a["text"] for a in atoms] == ["Gerçek gövde metni."]   # running-head gövdede değil
    furn = ppe.blocks_to_furniture(blocks)
    assert "DÖNEM: 27" in furn["page_header"]                       # → üst bilgi


def test_html_to_markdown_table_converts_and_leaves_plain_text():
    html = "<div style='text-align:center'>Başlık</div><table><tr><th>A</th><th>B</th></tr><tr><td>1</td><td>2</td></tr></table>"
    md = ppe._html_to_markdown(html)
    assert "<table" not in md and "<div" not in md      # HTML kalmadı
    assert "| A | B |" in md                            # MD pipe-tablo
    assert "| --- | --- |" in md
    assert "| 1 | 2 |" in md
    assert "Başlık" in md
    # düz metin/MD dokunulmaz
    assert ppe._html_to_markdown("düz **metin**") == "düz **metin**"
    assert ppe._html_to_markdown("") == ""


def test_blocks_to_atoms_converts_html_table_to_md():
    blocks = [{"label": "table", "bbox": "[]",
               "content": "<table><tr><td>474.600.000</td></tr></table>"}]
    atoms = ppe.blocks_to_atoms(blocks, 3)
    assert len(atoms) == 1
    assert atoms[0]["label"] == "table"
    assert "<table" not in atoms[0]["text"]
    assert "474.600.000" in atoms[0]["text"] and "|" in atoms[0]["text"]


def test_blocks_to_atoms_markdown_fallback():
    # Hiç blok yok / hepsi boş → markdown fallback'e düş
    atoms = ppe.blocks_to_atoms([], 7, markdown_fallback="# başlık\n\nmetin")
    assert len(atoms) == 1
    assert atoms[0]["label"] == "text"
    assert atoms[0]["page"] == 7
    assert atoms[0]["extracted_by"] == "paddle_page"


def test_blocks_to_atoms_no_content_no_fallback():
    assert ppe.blocks_to_atoms([{"label": "text", "bbox": "[]", "content": "  "}], 1) == []


# --- Sayfa sınıflandırma (gerçek fixture PDF, ağsız) ----------------------

def test_native_char_counts_and_threshold():
    counts = ppe.native_char_counts(str(SCANNED_PDF))
    assert set(counts.keys()) == {1}
    assert isinstance(counts[1], int)
    # dev eşik altında her sayfa "taranmış"; sıfır eşikte hiçbiri
    assert ppe.scanned_page_set(str(SCANNED_PDF), threshold=10**9) == {1}
    assert ppe.scanned_page_set(str(SCANNED_PDF), threshold=0) == set()
    assert ppe.page_count(str(SCANNED_PDF)) == 1


# --- Tam-sayfa yönlendirme (monkeypatch: ağ/render yok) -------------------

def _docling_atoms():
    """3 sayfalık sahte belge: s1 digital metin, s2 taranmış (çöp tablo), s3 digital."""
    return [
        {"text": "Sayfa1 digital paragraf", "label": "text", "page": 1, "pages": [1]},
        {"text": "| ğ | 5 1 1 1 |", "label": "table", "page": 2, "pages": [2],
         "table_num_rows": 0, "table_num_cols": 0},
        {"text": "Sayfa3 digital paragraf", "label": "text", "page": 3, "pages": [3]},
    ]


def test_process_pages_replaces_only_scanned(monkeypatch):
    monkeypatch.setattr(ppe, "scanned_page_set", lambda p, t=None: {2})
    monkeypatch.setattr(ppe, "page_count", lambda p: 3)
    monkeypatch.setattr(ppe, "render_page_png", lambda p, n, z=None: b"PNG")

    def fake_call(img, url, timeout):
        return {
            "markdown": "yedek",
            "angle": 90,
            "blocks": [
                {"label": "table", "bbox": "[0,0,1,1]",
                 "content": "<table><tr><td>474.600.000</td></tr></table>"},
                {"label": "number", "bbox": "[]", "content": "396"},
            ],
        }
    monkeypatch.setattr(ppe, "call_paddle_page", fake_call)

    atoms = _docling_atoms()
    out, stats = ppe.process_pages_with_paddle("x.pdf", atoms)

    assert stats["scanned_pages"] == [2]
    assert stats["paddle_calls"] == 1
    assert stats["failed"] == []
    assert stats["angles"] == {2: 90}
    # sıra: s1 (digital korundu) → s2 (paddle table) → s3 (digital korundu)
    assert out[0]["text"] == "Sayfa1 digital paragraf"
    assert out[-1]["text"] == "Sayfa3 digital paragraf"
    routed = [a for a in out if a.get("extracted_by") == "paddle_page"]
    assert len(routed) == 1                       # number gövdeye alınmadı (furniture)
    assert routed[0]["label"] == "table"
    assert routed[0]["page"] == 2
    # number → page-metadata (footer) olarak yakalandı (Docling FURNITURE hizası)
    assert stats["page_furniture"][2]["page_footer"] == "396"
    # digital atomlar dokunulmadı
    assert not any(a.get("extracted_by") == "paddle_page"
                   for a in out if a["page"] in (1, 3))


def test_process_pages_no_scanned_is_noop(monkeypatch):
    monkeypatch.setattr(ppe, "scanned_page_set", lambda p, t=None: set())
    monkeypatch.setattr(ppe, "page_count", lambda p: 3)
    atoms = _docling_atoms()
    out, stats = ppe.process_pages_with_paddle("x.pdf", atoms)
    assert out is atoms                            # değişmeden döner
    assert stats["paddle_calls"] == 0


def test_process_pages_empty_page_not_counted_as_failure(monkeypatch):
    """API başarılı ama içerik yok (yalnız footer → elenir) → 'empty', 'failed' değil."""
    monkeypatch.setattr(ppe, "scanned_page_set", lambda p, t=None: {2})
    monkeypatch.setattr(ppe, "page_count", lambda p: 3)
    monkeypatch.setattr(ppe, "render_page_png", lambda p, n, z=None: b"PNG")
    monkeypatch.setattr(
        ppe, "call_paddle_page",
        lambda img, url, t: {"markdown": "", "angle": 0,
                             "blocks": [{"label": "footer", "bbox": "[]", "content": "5"}]},
    )
    atoms = _docling_atoms()
    out, stats = ppe.process_pages_with_paddle("x.pdf", atoms)
    assert stats["paddle_calls"] == 1
    assert stats["empty"] == [2]
    assert stats["failed"] == []
    # sayfa 2'nin (çöp) Docling tablosu korundu — kayıp yok
    assert any(a["page"] == 2 for a in out)


def test_process_pages_failure_preserves_docling(monkeypatch):
    """Paddle çağrısı başarısızsa taranmış sayfanın Docling atomları KORUNUR (veri kaybı yok)."""
    monkeypatch.setattr(ppe, "scanned_page_set", lambda p, t=None: {2})
    monkeypatch.setattr(ppe, "page_count", lambda p: 3)
    monkeypatch.setattr(ppe, "render_page_png", lambda p, n, z=None: b"PNG")
    monkeypatch.setattr(ppe, "call_paddle_page", lambda img, url, t: None)  # başarısız

    atoms = _docling_atoms()
    out, stats = ppe.process_pages_with_paddle("x.pdf", atoms)
    assert stats["failed"] == [2]
    assert stats["paddle_calls"] == 0
    # 3 atom da korunmuş (s2 Docling çöp tablosu dahil — hiç değilse kayıp yok)
    assert len(out) == 3
    assert any(a["page"] == 2 and a["label"] == "table" for a in out)


def test_render_failure_preserves_docling(monkeypatch):
    monkeypatch.setattr(ppe, "scanned_page_set", lambda p, t=None: {2})
    monkeypatch.setattr(ppe, "page_count", lambda p: 3)
    monkeypatch.setattr(ppe, "render_page_png", lambda p, n, z=None: None)  # render patladı
    out, stats = ppe.process_pages_with_paddle("x.pdf", _docling_atoms())
    assert stats["failed"] == [2]
    assert len(out) == 3


# --- preview_page (viewer tek-sayfa önizleme sarmalayıcısı) ---------------

def test_preview_page_success(monkeypatch):
    monkeypatch.setattr(ppe, "render_page_png", lambda p, n, z=None: b"PNG")
    monkeypatch.setattr(
        ppe, "call_paddle_page",
        lambda img, url, t: {
            "markdown": "# başlık\n\nmetin",
            "angle": 270,
            "blocks": [
                {"label": "table", "bbox": "[0,0,1,1]", "content": "<table><tr><td>hücre</td></tr></table>"},
                {"label": "number", "bbox": "[]", "content": "12"},
            ],
            "annotated_image_b64": "QUJD",
        },
    )
    out = ppe.preview_page("x.pdf", 11)
    assert "error" not in out
    assert out["markdown"] == "# başlık\n\nmetin"   # ham Paddle çıktısı korunur
    assert out["angle"] == 270
    assert out["annotated_image_b64"] == "QUJD"
    assert isinstance(out["elapsed_s"], (int, float))
    # blocks ham (number dahil); atoms süzülmüş (number → furniture → yalnız table gövdede)
    assert [b["label"] for b in out["blocks"]] == ["table", "number"]
    assert [a["label"] for a in out["atoms"]] == ["table"]
    assert out["atoms"][0]["page"] == 11
    # number artık ATILMIYOR → page-metadata (footer) olarak yakalanır (Docling hizası)
    assert out["page_footer"] == "12"
    assert "body_markdown" in out


def test_preview_page_render_failure(monkeypatch):
    monkeypatch.setattr(ppe, "render_page_png", lambda p, n, z=None: None)
    out = ppe.preview_page("x.pdf", 11)
    assert "error" in out and "render" in out["error"].lower()


def test_preview_page_api_failure(monkeypatch):
    monkeypatch.setattr(ppe, "render_page_png", lambda p, n, z=None: b"PNG")
    monkeypatch.setattr(ppe, "call_paddle_page", lambda img, url, t: None)
    out = ppe.preview_page("x.pdf", 11)
    assert "error" in out


# --- Guard: paddle_page tablosu tekrar VLM'e tetiklenmez ------------------

def test_table_is_low_quality_skips_paddle_page():
    paddle_table = {
        "label": "table",
        "text": "<table><tr><td>474.600.000</td></tr></table>",
        "extracted_by": "paddle_page",
    }
    # HTML <table> normalde _looks_like_garbage'a takılırdı; guard onu keser
    assert table_is_low_quality(paddle_table) is False
