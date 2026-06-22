"""PaddleOCR Vision-Language tablosunu OpenAI-uyumlu API üzerinden çeker.

Aynı "bozuk tablo" tetikleyicisini (table_is_low_quality) ve kırpma/dik
çevirme altyapısını (crop_table_image, deskew_to_upright) vlm_table_extractor
ile paylaşır. Yalnızca VLM çağrısı değişir: Ollama yerine OpenAI-uyumlu
endpoint'e (LiteLLM proxy) JPEG görüntü gönderilir.

Kullanım:
    python -m src.common.parsing.markdown_converter --file belge.pdf --vlm paddleocr
"""

from __future__ import annotations

import base64
import io
import logging
import time
from typing import Any, Dict, List

import requests
from PIL import Image

from src.config import settings
from src.common.parsing.vlm_table_extractor import (
    clean_markdown_response,
    crop_table_image,
    table_is_low_quality,
)

_logger = logging.getLogger(__name__)


def _clean_paddle_response(text: str) -> str:
    """PaddleOCR ham yanıtını temizler.

    clean_markdown_response aksine `|` filtresi uygulamaz — PaddleOCR
    markdown tablo yerine düz OCR metni döndürebilir. Yalnızca backtick
    çitleri ve <|...|> chat-template artefaktlarını kaldırır.
    """
    text = text.strip()
    # Backtick çiti varsa içeriği al
    if text.startswith("```"):
        nl = text.find("\n")
        text = text[nl:].strip() if nl != -1 else text[3:].strip()
    if text.endswith("```"):
        text = text[:-3].strip()
    # <|im_start|> / <|im_end|> gibi chat-template artefaktları
    import re
    text = re.sub(r"<\|[^|>]+\|>", "", text).strip()
    return text


_PROMPT = (
    "You are given an image of a Turkish government budget table. "
    "If there is a title or caption text above or beside the table (e.g. a 'CETVEL' / "
    "law name / classification heading), output it FIRST as a markdown heading line "
    "starting with '## '. "
    "Then output the table as a precise markdown table: keep the column headers, align "
    "cells to the correct columns, preserve every numeric value exactly, and include all "
    "total/subtotal rows. "
    "Ignore page footers and page numbers. Fix only obvious OCR errors. "
    "Do not add explanations or ``` code fences. "
    "Respond with only the optional '## ' heading and the markdown table."
)


def _to_jpeg_b64(image_bytes: bytes) -> str:
    """PNG/herhangi bir görüntü byte'ını JPEG base64'e çevirir."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode()


def extract_table_with_paddle(
    image_bytes: bytes,
    base_url: str,
    model: str,
    api_key: str = "none",
    timeout: int = 120,
    start_time: float | None = None,
) -> str | None:
    """Görüntüyü PaddleOCR VL API'sine gönderir; temizlenmiş markdown döndürür."""
    if start_time is None:
        start_time = time.time()

    def _ts() -> str:
        return f"{time.time() - start_time:.1f}s"

    try:
        t0 = time.time()
        img_b64 = _to_jpeg_b64(image_bytes)
        jpeg_kb = len(img_b64) * 3 // 4 // 1024
        print(f"    [API] JPEG dönüşümü tamamlandı: ~{jpeg_kb} KB ({time.time()-t0:.1f}s)")

        url = f"{base_url.rstrip('/')}/chat/completions"
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
                        },
                        {"type": "text", "text": _PROMPT},
                    ],
                }
            ],
        }
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        print(f"    [API] POST {url}  model={model}  timeout={timeout}s")
        t1 = time.time()
        resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"]
        print(f"    [API] Yanıt alındı: {len(raw)} karakter ({time.time()-t1:.1f}s)")
        print(f"    [API] Ham yanıt (ilk 300 karakter):\n{raw[:300]}")
        cleaned = _clean_paddle_response(raw)
        lines = cleaned.count("\n") + 1 if cleaned else 0
        print(f"    [API] Temizlenmiş: {len(cleaned)} karakter, ~{lines} satır")
        return cleaned
    except Exception as e:
        _logger.error(f"PaddleOCR API hatası: {e}", exc_info=True)
        print(f"    [API] HATA: {e}")
        return None


def process_atoms_with_paddle(
    pdf_path: str,
    atoms: List[Dict[str, Any]],
    base_url: str | None = None,
    model: str | None = None,
    start_time: float | None = None,
) -> List[Dict[str, Any]]:
    """Bozuk tablo atomlarını PaddleOCR VL API ile yeniden okur."""
    if base_url is None:
        base_url = settings.PADDLE_OCR_URL
    if model is None:
        model = settings.PADDLE_OCR_MODEL
    if start_time is None:
        start_time = time.time()

    updated: List[Dict[str, Any]] = []
    table_count = 0

    def _ts() -> str:
        return f"{time.time() - start_time:.1f}s"

    for atom in atoms:
        atom_copy = dict(atom)
        if table_is_low_quality(atom_copy):
            bbox = atom_copy.get("bbox")
            page = atom_copy.get("page")
            if not (bbox and page):
                print(f"  [PADDLE] [{_ts()}] [WARN] Bozuk tablo ama bbox/sayfa yok — atlanıyor.")
            else:
                table_count += 1
                print(f"  [PADDLE] [{_ts()}] ── Tablo #{table_count} (sayfa {page}) ──────────────────")

                print(f"  [PADDLE] [{_ts()}] ADIM 1/3 — PyMuPDF bbox kırpma + yüksek çözünürlük render")
                t_crop = time.time()
                image_bytes = crop_table_image(
                    pdf_path, page, bbox,
                    zoom=None,
                    autorotate=settings.VLM_TABLE_AUTOROTATE,
                )
                print(f"  [PADDLE] [{_ts()}] Kırpma+deskew tamamlandı ({time.time()-t_crop:.1f}s)")

                if image_bytes:
                    print(f"  [PADDLE] [{_ts()}] ADIM 2/3 — JPEG encode + API isteği gönderiliyor")
                    t_api = time.time()
                    result = extract_table_with_paddle(
                        image_bytes, base_url, model,
                        api_key=settings.PADDLE_OCR_API_KEY,
                        timeout=settings.PADDLE_OCR_TIMEOUT,
                        start_time=start_time,
                    )
                    if result:
                        print(f"  [PADDLE] [{_ts()}] ADIM 3/3 — Atom metni güncellendi "
                              f"({len(result)} karakter, API süresi {time.time()-t_api:.1f}s)")
                        atom_copy["text"] = result
                        atom_copy["extracted_by"] = "paddle"
                    else:
                        print(f"  [PADDLE] [{_ts()}] [WARN] Çıkarım başarısız — orijinal OCR korunuyor.")
                else:
                    print(f"  [PADDLE] [{_ts()}] [WARN] Tablo görseli kırpılamadı — orijinal OCR korunuyor.")
        updated.append(atom_copy)

    return updated
