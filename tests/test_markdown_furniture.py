"""Sayfa üst/alt bilgi (FURNITURE) yakalama birim testleri — offline, model yüklemesiz.

Docling `iterate_items()` varsayılan `{BODY}` katmanını gezer; PAGE_HEADER/FOOTER ise
`FURNITURE` katmanındadır ve normalde tamamen elenir. Bu testler, ayrı FURNITURE
geçişinin (`_extract_furniture`) üst/alt bilgiyi page-metadata olarak yakaladığını ve
gövde çıkarımının (`_extract_atoms`) bunları gövdeye SOKMADIĞINI doğrular.
"""
from types import SimpleNamespace

from docling_core.types.doc import ContentLayer

from src.common.parsing.markdown_converter import MarkdownConverter


def _item(text, label, page):
    """Docling item taklidi: text + label + prov[page_no,charspan]."""
    return SimpleNamespace(
        text=text,
        label=label,
        prov=[SimpleNamespace(page_no=page, charspan=(0, len(text)))],
        data=None,
    )


class _FakeDoc:
    """Docling'in katman-filtreli iterate_items davranışını taklit eder:
    argümansız çağrı → yalnız BODY; included_content_layers verilirse ona göre."""

    def __init__(self, body=(), furniture=()):
        self._body = list(body)
        self._furn = list(furniture)

    def iterate_items(self, included_content_layers=None, **kwargs):
        layers = included_content_layers or {ContentLayer.BODY}
        if ContentLayer.BODY in layers:
            for it in self._body:
                yield it, 0
        if ContentLayer.FURNITURE in layers:
            for it in self._furn:
                yield it, 0


def _conv():
    # do_ocr=False → OCR options kurulmaz; DocumentConverter yapımı hafif (model
    # yalnız convert() çağrısında yüklenir, testte çağrılmaz).
    return MarkdownConverter(do_ocr=False)


def test_extract_furniture_groups_by_page_and_label():
    conv = _conv()
    doc = _FakeDoc(furniture=[
        _item("TBMM B:6 O:2 10.10.2018", "page_header", 1),
        _item("- 123 -", "page_footer", 1),
        _item("TBMM B:6 O:2 10.10.2018", "page_header", 2),
    ])
    furn = conv._extract_furniture(doc)
    assert furn[1]["page_header"] == "TBMM B:6 O:2 10.10.2018"
    assert furn[1]["page_footer"] == "- 123 -"
    assert furn[2]["page_header"] == "TBMM B:6 O:2 10.10.2018"
    assert furn[2]["page_footer"] == ""


def test_body_extraction_excludes_furniture():
    """_extract_atoms yalnız BODY gezer → üst bilgi gövdeye girmez."""
    conv = _conv()
    doc = _FakeDoc(
        body=[_item("Görüşmelere başlıyoruz.", "text", 1)],
        furniture=[_item("TBMM B:6 O:2", "page_header", 1)],
    )
    atoms = conv._extract_atoms(doc)
    texts = [a["text"] for a in atoms]
    assert "Görüşmelere başlıyoruz." in texts
    assert all("TBMM B:6" not in t for t in texts)  # üst bilgi gövdede YOK


def test_furniture_flows_into_pages_by_number_not_body():
    """Uçtan uca: furniture page-metadata olur, sayfa_markdown'a girmez."""
    conv = _conv()
    doc = _FakeDoc(
        body=[_item("Gündem maddesi bir.", "text", 1)],
        furniture=[_item("TBMM B:6 O:2", "page_header", 1)],
    )
    atoms = conv._extract_atoms(doc)
    furn = conv._extract_furniture(doc)
    pages = MarkdownConverter._build_pages_by_number(atoms, furn)
    p1 = next(p for p in pages if p["sayfa_no"] == 1)
    assert p1["page_header"] == "TBMM B:6 O:2"
    assert "TBMM B:6" not in p1["sayfa_markdown"]     # gövdeye sızmadı
    assert "Gündem maddesi bir." in p1["sayfa_markdown"]


def test_empty_furniture_yields_no_header_keys():
    """Furniture yoksa page_header/footer anahtarı hiç eklenmez (geriye uyum)."""
    atoms = [{"text": "x", "label": "text", "page": 1, "pages": [1]}]
    pages = MarkdownConverter._build_pages_by_number(atoms, {})
    assert pages == [{"sayfa_no": 1, "sayfa_markdown": "x"}]
