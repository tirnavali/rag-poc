"""Tek-sayfa canlı Docling+EasyOCR önizlemesi — ingestion viewer'ın "Docling (yeni ayar)"
butonu için.

paddle_page_extractor.preview_page'in kardeşidir ama Paddle yerine Docling'in KENDİ
(güncel ayarlı: DOCLING_IMAGES_SCALE, layout modeli, FURNITURE üst/alt bilgi yakalama)
çıktısını üretir; böylece kullanıcı cache'lenmiş ESKİ çıktı ile YENİ çıktıyı sayfa
sayfa kıyaslayabilir — tüm belgeyi yeniden ingest etmeden.

SALT-OKUMA sözleşmesi: seçili sayfa temp 1-sayfa PDF'e render edilir, MarkdownConverter'ın
YAPI TAŞLARI (converter + _extract_atoms/_extract_furniture/_build_pages_by_number)
doğrudan çağrılır. MarkdownConverter.convert() ÇAĞRILMAZ → hiçbir artifact/parse-cache/
Chroma yazılmaz (aksi halde temp-stem'li sahte belgeler viewer keşfini kirletirdi).
"""
from __future__ import annotations

import os
import tempfile
import time
from typing import Any, Dict

# Lazy singleton — EasyOCR + layout modelleri yalnız ilk çağrıda yüklensin, sonra
# yeniden kullanılsın (sayfa gezişi hızlı olsun).
_converter = None


def _get_converter():
    global _converter
    if _converter is None:
        from src.common.parsing.markdown_converter import MarkdownConverter
        # use_vlm=False → tablo-VLM katmanı devrede değil (hız + saf Docling çıktısı).
        # images_scale / layout modeli argüman verilmediği için settings'ten gelir.
        _converter = MarkdownConverter(use_vlm=False)
    return _converter


def preview_page_docling(pdf_path: str, page_no: int) -> Dict[str, Any]:
    """Bir PDF sayfasını güncel Docling ayarlarıyla canlı yeniden okur (salt-okuma).

    Döner: {markdown, page_header, page_footer, atoms:[{label,text}], atom_count,
    images_scale, elapsed_s} veya {error}.
    """
    if not os.path.exists(pdf_path):
        return {"error": f"PDF bulunamadı: {pdf_path}"}
    try:
        import fitz  # PyMuPDF
    except Exception as e:  # noqa: BLE001
        return {"error": f"PyMuPDF yüklenemedi: {e}"}

    t0 = time.time()
    tmp_path = None
    try:
        src = fitz.open(pdf_path)
        try:
            if page_no < 1 or page_no > len(src):
                return {"error": f"Sayfa {page_no} aralık dışı (toplam {len(src)})"}
            one = fitz.open()
            one.insert_pdf(src, from_page=page_no - 1, to_page=page_no - 1)
            fd, tmp_path = tempfile.mkstemp(suffix=".pdf", prefix=f"reparse_p{page_no}_")
            os.close(fd)
            one.save(tmp_path)
            one.close()
        finally:
            src.close()

        conv = _get_converter()
        # Ham Docling — MarkdownConverter.convert()'i BYPASS eder (artifact/cache yazmaz).
        result = conv.converter.convert(tmp_path)
        dl_doc = result.document
        atoms = conv._extract_atoms(dl_doc)
        furniture = conv._extract_furniture(dl_doc)  # temp tek-sayfa → anahtar 1
        pages = conv._build_pages_by_number(atoms, furniture)

        page = pages[0] if pages else {"sayfa_markdown": ""}
        furn1 = furniture.get(1, {})
        return {
            "markdown": page.get("sayfa_markdown", ""),
            "page_header": furn1.get("page_header", ""),
            "page_footer": furn1.get("page_footer", ""),
            "atoms": [
                {"label": a.get("label", ""), "text": a.get("text", "")} for a in atoms
            ],
            "atom_count": len(atoms),
            "images_scale": conv.images_scale,
            "elapsed_s": round(time.time() - t0, 1),
        }
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"}
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
