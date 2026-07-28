#!/usr/bin/env python3
"""Ingestion Debug Viewer — bir belgenin ingestion aşamalarını sayfa sayfa,
üç kolon yan yana gösteren bağımlılıksız (stdlib) HTTP arayüzü.

    python -m scripts.ingestion_viewer                       # http://localhost:8766
    python -m scripts.ingestion_viewer --port 9001
    python -m scripts.ingestion_viewer --collection tutanaklar_nomic_chunk256_768d

Her sayfa için üç panel:
  1) Orijinal PDF sayfa görüntüsü — data_lake/downloads altındaki yerel PDF'ten
     `fitz` (PyMuPDF) ile anında render edilir (persist edilmez).
  2) OCR sonrası metin — data_lake/pages/*_pages.json sidecar'ından (zaten persist).
  3) O sayfadan çıkarılan chunk'lar + metadata'sı — CANLI Chroma koleksiyonundan
     (`sira_sayisi`/`esas_no` backfill'i yalnızca Chroma'da olduğu için tek doğru kaynak).

Ayrıca her sayfaya, yeniden-OCR gerektirmeyen proxy kalite sinyalleri (karakter
yoğunluğu, ı/İ karışması, ünlü uyumu ihlali, chunk sayısı, etiket kapsamı) renk
kodlu rozet olarak eklenir.

Analitik mantık (chunks_for_page / page_quality_signals / build_overview) HTTP
katmanının DIŞINDA, saf ve import-edilebilir fonksiyonlardadır; tests/test_ingestion_viewer.py
bunları Chroma/HTTP olmadan offline test eder.

Belgeler data_lake/reports + pages'ten otomatik keşfedilir (golden_builder deseni).
Bu araç salt-okumadır — ingestion'a hiç dokunmaz.
"""
from __future__ import annotations

import argparse
import json
import re
from glob import glob
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from src.config import settings
from src.common.parsing.quality import turkish_ocr_signals

# golden_builder'daki belge keşfi / sayfa yükleme / Chroma-pages parse'ı yeniden kullanılır
from scripts.golden_builder import (
    _discover_documents,
    _doc_label,
    _load_pages,
    _parse_pages,
)

HTML_PATH = Path(__file__).resolve().parent / "ingestion_viewer.html"
DEFAULT_COLLECTION = "tutanaklar_nomic_chunk256_768d"  # üretim koleksiyonu (backfill'li metadata)


# --------------------------------------------------------------------------- #
# Saf analitik fonksiyonlar (HTTP/Chroma bağımsız — test edilebilir)
# --------------------------------------------------------------------------- #

# Proxy sinyal eşikleri — settings'e değil buraya konur: kolay ayar + birim testi.
CHAR_EMPTY = 1        # < bu → boş sayfa (OCR başarısız olabilir)
CHAR_SPARSE = 300     # < bu → seyrek metin (şüpheli)
I_CONF_WARN, I_CONF_BAD = 0.02, 0.05      # ı/İ karışma oranı
VOWEL_WARN, VOWEL_BAD = 0.20, 0.35        # ünlü uyumu ihlal oranı

_SEV = {"ok": 0, "warn": 1, "bad": 2}
_CHUNK_IDX_RE = re.compile(r"^(.+)_(\d+)$")


def _chunk_index(chunk_id: str) -> int:
    """chunk_id = '{document_id}_{N}' → N (sıralama için); eşleşmezse 0."""
    m = _CHUNK_IDX_RE.match(chunk_id or "")
    return int(m.group(2)) if m else 0


def pages_of_chunk(meta: dict) -> list[int]:
    """Bir chunk'ın değdiği sayfa numaraları. Önce 'pages' (Chroma'da '4, 5'
    virgül-string), yoksa tekil 'page'."""
    pages = _parse_pages((meta or {}).get("pages"))
    if pages:
        return pages
    p = (meta or {}).get("page")
    if p is None:
        return []
    try:
        return [int(p)]
    except (ValueError, TypeError):
        return []


def chunks_for_page(chunks: list[dict], page_no) -> list[dict]:
    """Verilen sayfaya değen chunk'ları, belge-içi sırayla (chunk index) döndürür.

    chunks: [{"chunk_id", "text", "metadata"}]. Çok-sayfalı bir chunk her
    değdiği sayfada görünür."""
    try:
        pn = int(page_no)
    except (ValueError, TypeError):
        return []
    sel = [c for c in chunks if pn in pages_of_chunk(c.get("metadata") or {})]
    return sorted(sel, key=lambda c: _chunk_index(c.get("chunk_id", "")))


def _has_value(meta: dict, key: str) -> bool:
    """Metadata alanının anlamlı (dolu) bir değer taşıyıp taşımadığı.
    Chroma'da yokluk = anahtarın olmaması; '0'/'None'/'' boş sayılır."""
    v = (meta or {}).get(key)
    if v is None:
        return False
    return str(v).strip() not in ("", "None", "0")


def _level(value: float, warn: float, bad: float) -> str:
    """Yüksek değer kötü: value >= bad → 'bad', >= warn → 'warn', aksi 'ok'."""
    if value >= bad:
        return "bad"
    if value >= warn:
        return "warn"
    return "ok"


def page_quality_signals(page_text: str, page_chunks: list[dict]) -> dict:
    """Bir sayfanın proxy kalite sinyalleri (yeniden-OCR yok, persist metinden).

    quality.py'nin turkish_ocr_signals'ını yeniden kullanır ki sinyaller
    ingestion kalite modülüyle tutarlı kalsın."""
    text = page_text or ""
    tr = turkish_ocr_signals(text)
    char_count = len(text)
    chunk_count = len(page_chunks)
    tagged = sum(1 for c in page_chunks if _has_value(c.get("metadata") or {}, "sira_sayisi"))
    esas = sum(1 for c in page_chunks if _has_value(c.get("metadata") or {}, "esas_no"))
    i_conf = tr["i_confusion_ratio"]
    vowel = tr["vowel_harmony_violation_ratio"]

    levels = {
        "char": "bad" if char_count < CHAR_EMPTY else ("warn" if char_count < CHAR_SPARSE else "ok"),
        "i_confusion": _level(i_conf, I_CONF_WARN, I_CONF_BAD),
        "vowel": _level(vowel, VOWEL_WARN, VOWEL_BAD),
        "chunks": "warn" if (chunk_count == 0 and char_count >= CHAR_SPARSE) else "ok",
    }
    worst = max(levels.values(), key=lambda lv: _SEV[lv])

    flags: list[str] = []
    if levels["char"] == "bad":
        flags.append("boş/çok kısa sayfa (OCR başarısız olabilir)")
    elif levels["char"] == "warn":
        flags.append("seyrek metin")
    if levels["i_confusion"] != "ok":
        flags.append(f"ı/İ karışması yüksek (%{i_conf * 100:.1f})")
    if levels["vowel"] != "ok":
        flags.append(f"ünlü uyumu ihlali yüksek (%{vowel * 100:.1f})")
    if levels["chunks"] == "warn":
        flags.append("bu sayfaya değen chunk yok (parça boşluğu?)")

    return {
        "char_count": char_count,
        "word_count": tr["word_count"],
        "chunk_count": chunk_count,
        "tagged_chunks": tagged,
        "esas_chunks": esas,
        "i_confusion_ratio": i_conf,
        "vowel_harmony_violation_ratio": vowel,
        "levels": levels,
        "worst_level": worst,
        "flags": flags,
    }


def build_overview(pages: list[dict], chunks: list[dict]) -> list[dict]:
    """Belge genel görünümü: her sayfa için hafif özet (nav + rozet şeridi için)."""
    out = []
    for p in pages:
        sayfa_no = p["sayfa_no"]
        pc = chunks_for_page(chunks, sayfa_no)
        sig = page_quality_signals(p.get("sayfa_markdown", ""), pc)
        out.append(
            {
                "sayfa_no": sayfa_no,
                "char_count": sig["char_count"],
                "chunk_count": sig["chunk_count"],
                "tagged_chunks": sig["tagged_chunks"],
                "worst_level": sig["worst_level"],
                "levels": sig["levels"],
            }
        )
    return out


# --------------------------------------------------------------------------- #
# Veri erişimi (Chroma / PDF — ağır importlar tembel)
# --------------------------------------------------------------------------- #

def find_pdf(document_id: str) -> str | None:
    """document_id için yerel PDF'i bul. Aynı PDF birden çok koleksiyon dizini
    altında indirilmiş olabilir; herhangi biri işimizi görür (aynı byte)."""
    hits = sorted(glob(str(settings.DOWNLOADS_DIR / "*" / document_id / "*.pdf")))
    return hits[0] if hits else None


def render_page_png(pdf_path: str, page_no: int, zoom: float = 2.0) -> bytes | None:
    """PDF sayfasını (1-tabanlı, fiziksel sayfa) upright PNG'ye render eder.
    crop_table_image ile aynı fitz deseni; clip yok → tam sayfa."""
    import fitz  # PyMuPDF (mevcut bağımlılık)

    try:
        doc = fitz.open(pdf_path)
        try:
            if page_no < 1 or page_no > len(doc):
                return None
            page = doc[page_no - 1]
            pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
            return pix.tobytes("png")
        finally:
            doc.close()
    except Exception:
        return None


def load_doc_chunks(collection, document_id: str) -> list[dict]:
    """Üretim koleksiyonundan bu belgeye ait tüm chunk'ları çeker (metadata dahil).
    where={document_id} → yalnız bu belge; embeddings çekilmez (hızlı)."""
    if collection is None:
        return []
    res = collection.get(
        where={"document_id": document_id},
        include=["documents", "metadatas"],
    )
    ids = res.get("ids") or []
    docs = res.get("documents") or []
    metas = res.get("metadatas") or []
    return [
        {"chunk_id": cid, "text": doc or "", "metadata": meta or {}}
        for cid, doc, meta in zip(ids, docs, metas)
    ]


# --------------------------------------------------------------------------- #
# HTTP handler
# --------------------------------------------------------------------------- #
class Handler(BaseHTTPRequestHandler):
    collection_name: str = DEFAULT_COLLECTION
    _docs_cache: list[dict] | None = None
    _docs_index: dict[str, dict] | None = None
    _collection = None
    _collection_err: str | None = None
    _doc_cache: dict[str, dict] = {}  # document_id -> {"pages": [...], "chunks": [...]}

    def log_message(self, fmt, *args):  # sessiz log
        pass

    # ----- koleksiyon (tembel) ----- #
    @classmethod
    def _get_collection(cls):
        if cls._collection is not None or cls._collection_err is not None:
            return cls._collection
        try:
            from src.config.collections import get_spec
            from src.common.chroma import open_collection

            spec = get_spec(cls.collection_name)
            _, cls._collection = open_collection(spec.db_path, spec.name)
        except Exception as exc:  # noqa: BLE001
            cls._collection_err = f"{type(exc).__name__}: {exc}"
            cls._collection = None
        return cls._collection

    # ----- belgeler ----- #
    def _docs(self) -> list[dict]:
        if Handler._docs_cache is None:
            Handler._docs_cache = _discover_documents()
            Handler._docs_index = {d["document_id"]: d for d in Handler._docs_cache}
        return Handler._docs_cache

    def _docs_by_id(self) -> dict[str, dict]:
        if Handler._docs_index is None:
            self._docs()
        return Handler._docs_index or {}

    @classmethod
    def _doc_data(cls, document_id: str) -> dict:
        """Bir belgenin (pages, chunks) verisini yükler ve önbelleğe alır."""
        cached = cls._doc_cache.get(document_id)
        if cached is not None:
            return cached
        doc = (cls._docs_index or {}).get(document_id)
        pages = _load_pages(doc) if doc else []
        chunks = load_doc_chunks(cls._get_collection(), document_id)
        data = {"pages": pages, "chunks": chunks}
        cls._doc_cache[document_id] = data
        return data

    # ----- yardımcılar ----- #
    def _send_json(self, obj, status: int = 200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self):
        body = HTML_PATH.read_text(encoding="utf-8").encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_png(self, data: bytes):
        self.send_response(200)
        self.send_header("Content-Type", "image/png")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    # ----- GET ----- #
    def do_GET(self):
        parsed = urlparse(self.path)
        route = parsed.path
        qs = parse_qs(parsed.query)
        try:
            if route in ("/", "/index.html"):
                return self._send_html()

            if route == "/api/config":
                coll = self._get_collection()
                return self._send_json(
                    {
                        "collection": self.collection_name,
                        "collection_ok": coll is not None,
                        "collection_error": self._collection_err,
                    }
                )

            if route == "/api/documents":
                out = []
                for d in self._docs():
                    pages = _load_pages(d)
                    if not pages:
                        continue
                    doc_id = d["document_id"]
                    out.append(
                        {
                            "document_id": doc_id,
                            "label": _doc_label(d),
                            "page_count": len(pages),
                            "has_pdf": find_pdf(doc_id) is not None,
                        }
                    )
                return self._send_json(out)

            if route == "/api/document":
                doc_id = (qs.get("document_id") or [""])[0]
                if doc_id not in self._docs_by_id():
                    return self._send_json({"error": "unknown document_id"}, 404)
                data = self._doc_data(doc_id)
                overview = build_overview(data["pages"], data["chunks"])
                return self._send_json(
                    {
                        "document_id": doc_id,
                        "label": _doc_label(self._docs_by_id()[doc_id]),
                        "collection": self.collection_name,
                        "collection_error": self._collection_err,
                        "has_pdf": find_pdf(doc_id) is not None,
                        "chunk_total": len(data["chunks"]),
                        "pages": overview,
                    }
                )

            if route == "/api/page":
                doc_id = (qs.get("document_id") or [""])[0]
                page_no = (qs.get("page") or [""])[0]
                if doc_id not in self._docs_by_id():
                    return self._send_json({"error": "unknown document_id"}, 404)
                data = self._doc_data(doc_id)
                page = next(
                    (p for p in data["pages"] if str(p["sayfa_no"]) == str(page_no)),
                    None,
                )
                if page is None:
                    return self._send_json({"error": "unknown page"}, 404)
                pc = chunks_for_page(data["chunks"], page["sayfa_no"])
                sig = page_quality_signals(page.get("sayfa_markdown", ""), pc)
                chunks_out = [
                    {
                        "chunk_id": c["chunk_id"],
                        "text": c["text"],
                        "pages": pages_of_chunk(c["metadata"]),
                        "metadata": c["metadata"],
                    }
                    for c in pc
                ]
                return self._send_json(
                    {
                        "sayfa_no": page["sayfa_no"],
                        "ocr_markdown": page.get("sayfa_markdown", ""),
                        "signals": sig,
                        "chunks": chunks_out,
                    }
                )

            if route == "/api/page-image":
                doc_id = (qs.get("document_id") or [""])[0]
                page_no = (qs.get("page") or [""])[0]
                try:
                    zoom = float((qs.get("zoom") or ["2.0"])[0])
                except (ValueError, TypeError):
                    zoom = 2.0
                zoom = max(0.5, min(zoom, 4.0))
                pdf = find_pdf(doc_id)
                if not pdf:
                    return self._send_json({"error": "PDF bulunamadı"}, 404)
                try:
                    pn = int(page_no)
                except (ValueError, TypeError):
                    return self._send_json({"error": "geçersiz sayfa"}, 400)
                png = render_page_png(pdf, pn, zoom=zoom)
                if png is None:
                    return self._send_json({"error": "render başarısız"}, 404)
                return self._send_png(png)

            return self._send_json({"error": "not found"}, 404)
        except Exception as exc:  # noqa: BLE001
            return self._send_json({"error": str(exc)}, 500)

    # ----- POST ----- #
    def do_POST(self):
        """Canlı sayfa önizleme POST'ları — hepsi SALT-OKUMA (hiçbir dosya/cache/Chroma yazmaz):
          /api/reocr-page   — sayfayı PaddleOCR-VL (:8080) ile yeniden oku
          /api/reparse-page — sayfayı güncel Docling+EasyOCR ayarlarıyla yeniden oku
        (golden_builder.do_POST dispatcher deseni.)
        """
        parsed = urlparse(self.path)
        try:
            length = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(length).decode("utf-8")) if length else {}
            if parsed.path == "/api/reocr-page":
                return self._handle_reocr_page(payload)
            if parsed.path == "/api/reparse-page":
                return self._handle_reparse_page(payload)
            return self._send_json({"error": "not found"}, 404)
        except Exception as exc:  # noqa: BLE001
            return self._send_json({"error": str(exc)}, 500)

    def _resolve_page_pdf(self, payload: dict):
        """Ortak: payload'dan (document_id, page, pdf_path) çözer ya da hata JSON'u yollar.
        Döner: (pdf_path, page_no) ya da None (hata zaten gönderildi)."""
        doc_id = (payload.get("document_id") or "").strip()
        if not doc_id:
            self._send_json({"error": "document_id zorunlu."}, 400)
            return None
        try:
            page_no = int(payload.get("page"))
        except (TypeError, ValueError):
            self._send_json({"error": "geçerli bir page (int) gerekli."}, 400)
            return None
        pdf = find_pdf(doc_id)
        if not pdf:
            self._send_json({"error": f"PDF bulunamadı: {doc_id}"}, 404)
            return None
        return pdf, page_no

    def _handle_reocr_page(self, payload: dict):
        resolved = self._resolve_page_pdf(payload)
        if resolved is None:
            return
        pdf, page_no = resolved
        # Ağır import gecikmeli (ilk POST'ta) — golden_builder _get_retriever deseni.
        from src.common.parsing.paddle_page_extractor import preview_page

        result = preview_page(pdf, page_no)
        status = 502 if result.get("error") else 200
        return self._send_json(result, status)

    def _handle_reparse_page(self, payload: dict):
        """Sayfayı güncel Docling ayarlarıyla (images_scale + FURNITURE üst bilgi)
        canlı yeniden okur — cache'lenmiş eski çıktı ile kıyas için. Salt-okuma."""
        resolved = self._resolve_page_pdf(payload)
        if resolved is None:
            return
        pdf, page_no = resolved
        from src.common.parsing.docling_page_preview import preview_page_docling

        result = preview_page_docling(pdf, page_no)
        status = 502 if result.get("error") else 200
        return self._send_json(result, status)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8766)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument(
        "--collection",
        default=DEFAULT_COLLECTION,
        help=f"Chunk+metadata kaynağı (üretim koleksiyonu). Vars: {DEFAULT_COLLECTION}",
    )
    args = ap.parse_args()

    Handler.collection_name = args.collection
    Handler._docs_cache = None
    Handler._docs_index = None
    Handler._collection = None
    Handler._collection_err = None
    Handler._doc_cache = {}

    docs = _discover_documents()
    print(f"Ingestion Viewer → http://{args.host}:{args.port}")
    print(f"Koleksiyon (chunk kaynağı): {args.collection} (ilk belge açılışında yüklenir)")
    print(f"Belgeler : data_lake/reports + pages otomatik keşif → {len(docs)} belge")
    print("Durdurmak için Ctrl+C")

    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nKapatılıyor…")
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
