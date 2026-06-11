#!/usr/bin/env python3
"""Golden Q&A Builder — sayfa içeriğini gezip golden fixture'a girdi ekleyen
bağımlılıksız (stdlib) HTTP arayüzü.

    python -m scripts.golden_builder            # http://localhost:8765
    python -m scripts.golden_builder --port 9000 --fixture path/to/golden.json

Tarayıcıda soru + golden_answer yazılır, cevabın bulunduğu sayfa(lar) işaretlenir;
kayıt mevcut fixture'a merge edilir. Cevabın atıfta bulunulan sayfada gerçekten
geçip geçmediği `lint_golden.answer_in_pages` ile canlı doğrulanır (uyarı verir,
kaydı engellemez). Şema mevcut fixture ile birebir aynıdır — lint_golden /
benchmark uyumlu kalır.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from scripts.lint_golden import (
    DEFAULT_MANIFEST,
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
# Veri erişimi
# --------------------------------------------------------------------------- #
def _load_manifest(manifest_path: Path) -> list[dict]:
    return json.loads(Path(manifest_path).read_text(encoding="utf-8"))["documents"]


def _load_pages(doc: dict) -> list[dict]:
    """Bir belgenin [{sayfa_no, sayfa_markdown}] dizisini döndürür."""
    path = _pages_path_for(doc["document_id"], doc.get("document_source", ""))
    if path is None:
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
    manifest_path: Path = DEFAULT_MANIFEST
    fixture_path: Path = DEFAULT_FIXTURE

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
        return _load_manifest(self.manifest_path)

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
        if parsed.path != "/api/save":
            return self._send_json({"error": "not found"}, 404)
        try:
            length = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(length).decode("utf-8")) if length else {}
            return self._handle_save(payload)
        except Exception as exc:  # noqa: BLE001
            return self._send_json({"error": str(exc)}, 500)

    def _handle_save(self, payload: dict):
        document_id = (payload.get("document_id") or "").strip()
        query = (payload.get("query") or "").strip()
        answer = (payload.get("golden_answer") or "").strip()
        pages = payload.get("pages") or []
        tags = payload.get("tags") or []

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

        # query'ye göre dedup (normalize)
        nq = _norm(query)
        existing_idx = next(
            (i for i, it in enumerate(items) if _norm(it.get("query", "")) == nq), None
        )

        entry = {
            "query": query,
            "relevant_pages": [{"document_id": document_id, "pages": pages_int}],
            "golden_answer": answer,
            "tags": final_tags,
        }

        if existing_idx is not None:
            entry["id"] = items[existing_idx].get("id") or _next_id(items, document_id)
            ordered = {"id": entry["id"], **entry}
            items[existing_idx] = ordered
            action = "updated"
        else:
            new_id = _next_id(items, document_id)
            ordered = {"id": new_id, **entry}
            items.append(ordered)
            action = "created"

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
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = ap.parse_args()

    Handler.fixture_path = args.fixture
    Handler.manifest_path = args.manifest

    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Golden Builder → http://{args.host}:{args.port}")
    print(f"Fixture : {args.fixture}")
    print(f"Manifest: {args.manifest}")
    print("Durdurmak için Ctrl+C")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nKapatılıyor…")
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
