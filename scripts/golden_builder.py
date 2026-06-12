#!/usr/bin/env python3
"""Golden Q&A Builder — sayfa içeriğini gezip golden fixture'a girdi ekleyen
bağımlılıksız (stdlib) HTTP arayüzü.

    python -m scripts.golden_builder            # http://localhost:8765
    python -m scripts.golden_builder --port 9000 --fixture path/to/golden.json

Belgeler data_lake/reports/ ve data_lake/pages/ dizinlerinden otomatik keşfedilir;
manifest dosyası artık zorunlu değildir.
Tarayıcıda soru + golden_answer yazılır, cevabın bulunduğu sayfa(lar) işaretlenir;
kayıt mevcut fixture'a merge edilir. Cevabın atıfta bulunulan sayfada gerçekten
geçip geçmediği `lint_golden.answer_in_pages` ile canlı doğrulanır (uyarı verir,
kaydı engellemez). Şema mevcut fixture ile birebir aynıdır — lint_golden /
benchmark uyumlu kalır."""
from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from glob import glob
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from scripts.lint_golden import (
    DEFAULT_MANIFEST,
    PAGES_DIR,
    REPORTS_DIR,
    ROOT,
    _norm,
    _pages_path_for,
    answer_in_pages,
)

HTML_PATH = Path(__file__).resolve().parent / "golden_builder.html"
DEFAULT_FIXTURE = ROOT / "tests" / "fixtures" / "golden_tbmm27001001.json"

# document_id örn: "tutanak-27-01-01-20180707" -> (period, leg_year, session)
DOC_ID_RE = re.compile(r"tutanak-(\d+)-(\d+)-(\d+)-")


# --------------------------------------------------------------------------- #
# Otomatik belge keşfi — reports/ + pages/ dizinlerinden
# --------------------------------------------------------------------------- #
def _discover_documents() -> list[dict]:
    """data_lake/reports/*.json ve data_lake/pages/*_pages.json dosyalarından
    tüm belgeleri keşfeder. Manifest gerektirmez."""
    docs: dict[str, dict] = {}

    # 1) reports/ dizininden: document_id + artifacts.pages doğrudan okur
    for report_file in sorted(REPORTS_DIR.glob("*.json")):
        try:
            data = json.loads(report_file.read_text(encoding="utf-8"))
        except Exception:
            continue
        doc_id = data.get("document_id", "")
        if not doc_id:
            continue
        pages_path_str = (data.get("artifacts") or {}).get("pages", "")
        if not pages_path_str:
            continue
        pages_path = Path(pages_path_str)
        if not pages_path.is_absolute():
            pages_path = ROOT / pages_path
        if not pages_path.exists():
            continue
        # oturum ve tarih bilgisini çıkar
        m = DOC_ID_RE.match(doc_id)
        session = int(m.group(3)) if m else None
        # tarih: document_id'nin son parçası (örn. 20180709)
        date_part = doc_id.rsplit("-", 1)[-1] if "-" in doc_id else ""
        date = f"{date_part[:4]}-{date_part[4:6]}-{date_part[6:]}" if len(date_part) == 8 else ""
        docs[doc_id] = {
            "document_id": doc_id,
            "document_source": "",
            "session": session,
            "document_date": date,
            "_pages_path": str(pages_path),
        }

    # 2) pages/ dizininden: reports'ta yer almayan dosyaları da ekle
    for pages_file in sorted(PAGES_DIR.glob("*_pages.json")):
        stem = pages_file.name  # örn: tbmm27001002__30ee9f62_pages.json
        # reports'tan zaten eklendiyse atla
        already = any(
            Path(d["_pages_path"]) == pages_file for d in docs.values()
        )
        if already:
            continue
        # document_id olarak dosya adının stem kısmını kullan
        doc_id = pages_file.stem.replace("_pages", "")  # tbmm27001002__30ee9f62
        docs[doc_id] = {
            "document_id": doc_id,
            "document_source": "",
            "session": None,
            "document_date": "",
            "_pages_path": str(pages_file),
        }

    # session numarasına göre sırala (tutanaklar önce, diğerleri sona)
    def sort_key(d):
        s = d.get("session")
        return (0 if s is not None else 1, s or 0, d["document_id"])

    return sorted(docs.values(), key=sort_key)


# --------------------------------------------------------------------------- #
# Veri erişimi
# --------------------------------------------------------------------------- #
def _load_manifest(manifest_path: Path) -> list[dict]:
    return json.loads(Path(manifest_path).read_text(encoding="utf-8"))["documents"]


def _load_pages(doc: dict) -> list[dict]:
    """Bir belgenin [{sayfa_no, sayfa_markdown}] dizisini döndürür.

    Önce doc içindeki '_pages_path' anahtarına bakar (otomatik keşif yolu),
    yoksa lint_golden._pages_path_for() ile çözer.
    """
    pages_path_str = doc.get("_pages_path", "")
    if pages_path_str:
        path = Path(pages_path_str)
    else:
        path = _pages_path_for(doc["document_id"], doc.get("document_source", ""))
    if path is None or not path.exists():
        return []
    entries = json.loads(Path(path).read_text(encoding="utf-8"))
    return [
        {"sayfa_no": int(e["sayfa_no"]), "sayfa_markdown": e.get("sayfa_markdown", "")}
        for e in entries
    ]


def _load_fixture(fixture_path: Path) -> list[dict]:
    if not fixture_path.exists():
        return []
    return json.loads(fixture_path.read_text(encoding="utf-8"))


def _save_fixture(fixture_path: Path, items: list[dict]) -> None:
    """Atomik yazım: temp dosyaya yaz, sonra replace."""
    fixture_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(fixture_path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(items, fh, ensure_ascii=False, indent=2)
            fh.write("\n")
        os.replace(tmp, fixture_path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def _next_id(items: list[dict], document_id: str) -> str:
    """tbmm{period}-{leg}-{session}-{NNN}: aynı oturumdaki max NNN + 1."""
    m = DOC_ID_RE.match(document_id)
    if not m:
        # bilinmeyen format -> genel artan sayaç
        prefix = "tbmm27-01-00"
        existing = [it.get("id", "") for it in items]
    else:
        period, leg, session = m.group(1), m.group(2), m.group(3)
        prefix = f"tbmm{period}-{leg}-{session}"
        existing = [it.get("id", "") for it in items]
    pat = re.compile(rf"^{re.escape(prefix)}-(\d+)$")
    nums = [int(mm.group(1)) for it in existing if (mm := pat.match(it))]
    nxt = (max(nums) + 1) if nums else 1
    return f"{prefix}-{nxt:03d}"


# --------------------------------------------------------------------------- #
# HTTP handler
# --------------------------------------------------------------------------- #
class Handler(BaseHTTPRequestHandler):
    # sınıf değişkenleri server kurulumunda atanır
    manifest_path: Path | None = None  # None = otomatik keşif (önerilen)
    fixture_path: Path = DEFAULT_FIXTURE
    _docs_cache: list[dict] | None = None

    def log_message(self, fmt, *args):  # daha sessiz log
        pass

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

    def _docs(self) -> list[dict]:
        if Handler._docs_cache is not None:
            return Handler._docs_cache
        if self.manifest_path is not None:
            Handler._docs_cache = _load_manifest(self.manifest_path)
        else:
            Handler._docs_cache = _discover_documents()
        return Handler._docs_cache

    # ----- GET ----- #
    def do_GET(self):
        parsed = urlparse(self.path)
        route = parsed.path
        qs = parse_qs(parsed.query)
        try:
            if route in ("/", "/index.html"):
                return self._send_html()

            if route == "/api/documents":
                out = []
                for d in self._docs():
                    pages = _load_pages(d)
                    if not pages:
                        continue
                    sess = d.get("session")
                    date = d.get("document_date") or ""
                    label = f"{sess}. Birleşim — {date}" if sess else d["document_id"]
                    out.append(
                        {
                            "document_id": d["document_id"],
                            "label": label,
                            "date": date,
                            "session": sess,
                            "page_count": len(pages),
                        }
                    )
                return self._send_json(out)

            if route == "/api/pages":
                doc_id = (qs.get("document_id") or [""])[0]
                doc = next(
                    (d for d in self._docs() if d["document_id"] == doc_id), None
                )
                if doc is None:
                    return self._send_json({"error": "unknown document_id"}, 404)
                return self._send_json(_load_pages(doc))

            if route == "/api/entries":
                doc_id = (qs.get("document_id") or [""])[0]
                items = _load_fixture(self.fixture_path)
                out = []
                for it in items:
                    rps = it.get("relevant_pages") or []
                    if any(rp.get("document_id") == doc_id for rp in rps):
                        pages = sorted(
                            {p for rp in rps for p in rp.get("pages", [])}
                        )
                        out.append(
                            {
                                "id": it.get("id"),
                                "query": it.get("query", ""),
                                "golden_answer": it.get("golden_answer", ""),
                                "pages": pages,
                                "tags": it.get("tags", []),
                            }
                        )
                return self._send_json(out)

            return self._send_json({"error": "not found"}, 404)
        except Exception as exc:  # noqa: BLE001
            return self._send_json({"error": str(exc)}, 500)

    # ----- POST ----- #
    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path not in ("/api/save", "/api/delete"):
            return self._send_json({"error": "not found"}, 404)
        try:
            length = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(length).decode("utf-8")) if length else {}
            if parsed.path == "/api/delete":
                return self._handle_delete(payload)
            return self._handle_save(payload)
        except Exception as exc:  # noqa: BLE001
            return self._send_json({"error": str(exc)}, 500)

    def _handle_delete(self, payload: dict):
        entry_id = (payload.get("id") or "").strip()
        if not entry_id:
            return self._send_json({"error": "id zorunlu."}, 400)
        items = _load_fixture(self.fixture_path)
        new_items = [it for it in items if it.get("id") != entry_id]
        if len(new_items) == len(items):
            return self._send_json({"error": f"Girdi bulunamadı: {entry_id}"}, 404)
        _save_fixture(self.fixture_path, new_items)
        return self._send_json({"ok": True, "id": entry_id, "total": len(new_items)})

    def _handle_save(self, payload: dict):
        document_id = (payload.get("document_id") or "").strip()
        query = (payload.get("query") or "").strip()
        answer = (payload.get("golden_answer") or "").strip()
        pages = payload.get("pages") or []
        tags = payload.get("tags") or []
        edit_id = (payload.get("edit_id") or "").strip()  # varsa ID üzerinden güncelle

        if not document_id or not query or not answer or not pages:
            return self._send_json(
                {"error": "document_id, query, golden_answer ve pages zorunlu."}, 400
            )

        docs = self._docs()
        doc = next((d for d in docs if d["document_id"] == document_id), None)
        if doc is None:
            return self._send_json({"error": "unknown document_id"}, 400)

        # sayfa metinleri (doğrulama için)
        page_map = {p["sayfa_no"]: p["sayfa_markdown"] for p in _load_pages(doc)}
        try:
            pages_int = sorted({int(p) for p in pages})
        except (ValueError, TypeError):
            return self._send_json({"error": "pages tam sayı olmalı."}, 400)
        missing_pages = [p for p in pages_int if p not in page_map]
        if missing_pages:
            return self._send_json(
                {"error": f"sayfa sidecar'da yok: {missing_pages}"}, 400
            )

        # canlı doğrulama
        page_texts = [page_map[p] for p in pages_int]
        ok, overlap, missing_spans = answer_in_pages(answer, page_texts)
        warning = None
        if not ok:
            parts = []
            if missing_spans:
                parts.append(f"sayfada bulunamayan sayısal ifade: {missing_spans}")
            parts.append(f"metin örtüşmesi düşük (%{overlap * 100:.0f})")
            warning = "Cevap atıfta bulunulan sayfada doğrulanamadı: " + "; ".join(parts)

        # otomatik etiketler
        final_tags = list(dict.fromkeys(["tbmm-minutes", "page-level", *tags]))
        if len(pages_int) > 1 and "cross-page" not in final_tags:
            final_tags.append("cross-page")

        items = _load_fixture(self.fixture_path)

        entry = {
            "query": query,
            "relevant_pages": [{"document_id": document_id, "pages": pages_int}],
            "golden_answer": answer,
            "tags": final_tags,
        }

        if edit_id:
            # ID üzerinden düzenle (sorgu değişmiş olabilir)
            existing_idx = next(
                (i for i, it in enumerate(items) if it.get("id") == edit_id), None
            )
            if existing_idx is None:
                return self._send_json({"error": f"Girdi bulunamadı: {edit_id}"}, 404)
            entry["id"] = edit_id
            items[existing_idx] = {"id": edit_id, **entry}
            action = "updated"
        else:
            # Sorguya göre dedup (yeni kayıt veya aynı sorguyu güncelle)
            nq = _norm(query)
            existing_idx = next(
                (i for i, it in enumerate(items) if _norm(it.get("query", "")) == nq), None
            )
            if existing_idx is not None:
                entry["id"] = items[existing_idx].get("id") or _next_id(items, document_id)
                items[existing_idx] = {"id": entry["id"], **entry}
                action = "updated"
            else:
                new_id = _next_id(items, document_id)
                items.append({"id": new_id, **entry})
                entry["id"] = new_id
                action = "created"

        ordered = next(it for it in items if it.get("id") == entry["id"])

        _save_fixture(self.fixture_path, items)
        return self._send_json(
            {
                "ok": True,
                "action": action,
                "id": ordered["id"],
                "total": len(items),
                "warning": warning,
                "tags": final_tags,
            }
        )



def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    ap.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Opsiyonel: manifest JSON. Belirtilmezse data_lake/reports + pages otomatik taranır.",
    )
    args = ap.parse_args()

    Handler.fixture_path = args.fixture
    Handler.manifest_path = args.manifest  # None → otomatik keşif
    Handler._docs_cache = None  # önbelleği sıfırla

    # Belge listesini başlangıçta keşfet ve göster
    docs = _discover_documents() if args.manifest is None else _load_manifest(args.manifest)
    print(f"Golden Builder → http://{args.host}:{args.port}")
    print(f"Fixture  : {args.fixture}")
    if args.manifest:
        print(f"Manifest : {args.manifest}")
    else:
        print(f"Belgeler : data_lake/reports + pages otomatik keşif → {len(docs)} belge bulundu")
        for d in docs:
            print(f"  · {d['document_id']}")
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
