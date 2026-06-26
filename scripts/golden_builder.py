#!/usr/bin/env python3
"""Golden Q&A Builder — retrieval havuzundan golden fixture'a girdi ekleyen
bağımlılıksız (stdlib) HTTP arayüzü.

    python -m scripts.golden_builder            # http://localhost:8765
    python -m scripts.golden_builder --port 9000 --fixture path/to/golden.json
    python -m scripts.golden_builder --collection tutanaklar_nomic_chunk256_768d

İki çalışma akışı vardır:

1) Retrieval ile işaretleme (birincil): bir soru yazılır, üretim koleksiyonunda
   retrieval + cross-encoder rerank çalışır, gelen ~30 sonuç gösterilir; ilgili
   olan sonuç(lar)ın bulunduğu sayfa "golden" olarak işaretlenir. Bu, IR'deki
   "pooling" yöntemidir ve `src/evaluator/benchmark.py` page_overlap puanlayıcısı
   ile birebir uyumludur (golden anahtarı `{document_id}#page_{n}`).
2) Manuel gezinme (escape-hatch): sol panelden sayfalar tek tek gezilip ilgili
   sayfa elle eklenebilir — retrieval'ın kaçırdığı ilgili sayfaları yakalamak için.

Belgeler data_lake/reports/ ve data_lake/pages/ dizinlerinden otomatik keşfedilir;
manifest dosyası gerekmez. Cevabın (golden_answer) atıfta bulunulan sayfada gerçekten
geçip geçmediği `lint_golden.answer_in_pages` ile canlı doğrulanır (uyarı verir,
kaydı engellemez). golden_answer artık opsiyoneldir. Şema mevcut fixture ile birebir
aynıdır (`relevant_pages: [{document_id, pages:[int]}]`) — lint_golden / benchmark
uyumlu kalır; sonuçlar birden çok belgeye yayıldığında relevant_pages belge başına
bir girdi taşır."""
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

    # 2) pages/ dizininden: reports'ta yer almayan dosyaları da ekle.
    # VLM açıkken artefaktlar __vlm sufiksiyle yazılır; aynı belgenin eski (VLM'siz)
    # plain sidecar'ı bayat kalıp golden_builder'da "hayalet bozuk belge" olarak
    # görünür. Bu yüzden plain ve __vlm aynı belge sayılır (base = sufiksiz ad) ve
    # __vlm (kanonik) tercih edilir; report zaten kapsayan base'ler atlanır.
    # base = tablo-çıkarıcı sufiksleri (__vlm/__tess/__paddle[-model]) ve _pages.json
    # soyulmuş ad; aynı belgenin farklı çıkarıcı varyantları tek belgede toplanır.
    def _pages_base(name: str) -> str:
        b = name[: -len("_pages.json")] if name.endswith("_pages.json") else name
        b = re.sub(r"__paddle(-[\w.\-]+)?$", "", b)
        for suf in ("__vlm", "__tess"):
            if b.endswith(suf):
                b = b[: -len(suf)]
        return b

    report_bases = {_pages_base(Path(d["_pages_path"]).name) for d in docs.values()}

    by_base: dict[str, list[Path]] = {}
    for pages_file in sorted(PAGES_DIR.glob("*_pages.json")):
        by_base.setdefault(_pages_base(pages_file.name), []).append(pages_file)

    # kanonik tercih sırası: __vlm > __tess > __paddle > plain
    def _variant_rank(f: Path) -> int:
        n = f.name
        if n.endswith("__vlm_pages.json"):
            return 0
        if n.endswith("__tess_pages.json"):
            return 1
        if "__paddle" in n:
            return 2
        return 3

    for base, files in sorted(by_base.items()):
        if base in report_bases:
            continue  # report kanonik artefaktıyla zaten ekledi
        chosen = sorted(files, key=_variant_rank)[0]
        doc_id = chosen.stem.replace("_pages", "")
        docs[doc_id] = {
            "document_id": doc_id,
            "document_source": "",
            "session": None,
            "document_date": "",
            "_pages_path": str(chosen),
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


def _parse_pages(raw) -> list[int]:
    """Chunk metadata'sındaki 'pages' değerini int listesine çevirir.

    Chroma list'leri virgülle birleşik string olarak saklar (ör. '12, 13');
    list / int / None biçimleri de tolere edilir."""
    if raw is None:
        return []
    if isinstance(raw, (list, tuple)):
        vals = raw
    elif isinstance(raw, int):
        return [raw]
    else:  # string
        vals = str(raw).split(",")
    out: list[int] = []
    for v in vals:
        try:
            out.append(int(str(v).strip()))
        except (ValueError, TypeError):
            continue
    return sorted(set(out))


def _doc_label(d: dict) -> str:
    """Belge için insan-okur etiket: '{oturum}. Birleşim — {tarih}' ya da document_id."""
    sess = d.get("session")
    date = d.get("document_date") or ""
    return f"{sess}. Birleşim — {date}" if sess else d["document_id"]


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
    collection_name: str = "tutanaklar_nomic_chunk256_768d"  # retrieval havuzu (üretim koleksiyonu)
    _docs_cache: list[dict] | None = None
    _docs_index: dict[str, dict] | None = None
    _retriever = None  # lazy: ilk /api/retrieve çağrısında kurulur
    _retriever_err: str | None = None

    def log_message(self, fmt, *args):  # daha sessiz log
        pass

    # ----- retrieval (lazy) ----- #
    @classmethod
    def _get_retriever(cls):
        """VectorRetriever'ı tembel kurar (ağır importlar yalnızca burada).

        Başarısızlık durumunda (eksik bağımlılık / koleksiyon yok) hata mesajı
        _retriever_err'e yazılır ve None döner; manuel akış etkilenmez."""
        if cls._retriever is not None or cls._retriever_err is not None:
            return cls._retriever
        try:
            from src.config.collections import get_spec
            from src.retriever.vector_retriever import VectorRetriever

            spec = get_spec(cls.collection_name)
            cls._retriever = VectorRetriever(spec)
        except Exception as exc:  # noqa: BLE001
            cls._retriever_err = f"{type(exc).__name__}: {exc}"
            cls._retriever = None
        return cls._retriever

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
        Handler._docs_index = {d["document_id"]: d for d in Handler._docs_cache}
        return Handler._docs_cache

    def _docs_by_id(self) -> dict[str, dict]:
        if Handler._docs_index is None:
            self._docs()  # _docs_index'i de doldurur
        return Handler._docs_index or {}

    # ----- GET ----- #
    def do_GET(self):
        parsed = urlparse(self.path)
        route = parsed.path
        qs = parse_qs(parsed.query)
        try:
            if route in ("/", "/index.html"):
                return self._send_html()

            if route == "/api/config":
                return self._send_json({"collection": self.collection_name})

            if route == "/api/documents":
                out = []
                for d in self._docs():
                    pages = _load_pages(d)
                    if not pages:
                        continue
                    out.append(
                        {
                            "document_id": d["document_id"],
                            "label": _doc_label(d),
                            "date": d.get("document_date") or "",
                            "session": d.get("session"),
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
                # document_id verilmezse TÜM sorular döner (sorular tüm derlemden
                # sorgulandığı için varsayılan budur). document_id verilirse o
                # belgeye değen sorulara filtrelenir (geriye dönük uyum).
                doc_id = (qs.get("document_id") or [""])[0]
                items = _load_fixture(self.fixture_path)
                idx = self._docs_by_id()
                out = []
                for it in items:
                    rps = it.get("relevant_pages") or []
                    if doc_id and not any(rp.get("document_id") == doc_id for rp in rps):
                        continue
                    rel = [
                        {
                            "document_id": rp.get("document_id", ""),
                            "label": _doc_label(idx[rp["document_id"]])
                            if rp.get("document_id") in idx
                            else rp.get("document_id", ""),
                            "pages": rp.get("pages", []),
                        }
                        for rp in rps
                    ]
                    out.append(
                        {
                            "id": it.get("id"),
                            "query": it.get("query", ""),
                            "golden_answer": it.get("golden_answer", ""),
                            "relevant_pages": rel,
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
        if parsed.path not in ("/api/save", "/api/delete", "/api/retrieve"):
            return self._send_json({"error": "not found"}, 404)
        try:
            length = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(length).decode("utf-8")) if length else {}
            if parsed.path == "/api/delete":
                return self._handle_delete(payload)
            if parsed.path == "/api/retrieve":
                return self._handle_retrieve(payload)
            return self._handle_save(payload)
        except Exception as exc:  # noqa: BLE001
            return self._send_json({"error": str(exc)}, 500)

    def _handle_retrieve(self, payload: dict):
        """Soruyu üretim koleksiyonunda retrieval + rerank ile arar; her sonucu
        sayfa numarasıyla birlikte döndürür. Sonuçlar pooling havuzunu oluşturur;
        kullanıcı ilgili olanların sayfasını golden işaretler."""
        query = (payload.get("query") or "").strip()
        if not query:
            return self._send_json({"error": "query zorunlu."}, 400)
        try:
            top_k = int(payload.get("top_k") or 30)
        except (ValueError, TypeError):
            top_k = 30
        top_k = max(1, min(top_k, 100))
        # Rerank havuz seçimini per-call kontrol eder; global settings.USE_RERANKER'a
        # dokunmaz (ThreadingHTTPServer'da global mutasyon thread-race olur). UI her
        # zaman açıkça gönderir; anahtar yoksa eski REST davranışı (açık) korunur.
        use_reranker = bool(payload.get("use_reranker", True))

        retr = self._get_retriever()
        if retr is None:
            return self._send_json(
                {
                    "error": "Retrieval kullanılamıyor: " + (self._retriever_err or "?"),
                    "hint": "Koleksiyonun indeklenmiş olduğundan ve ML bağımlılıklarının kurulu olduğundan emin olun.",
                },
                503,
            )

        # rerank havuzu top_k'nın en az 4 katı (sağlıklı cross-encoder seçimi için)
        fetch_k = max(top_k * 4, 120)
        res = retr.retrieve(query, top_k=top_k, fetch_k=fetch_k, rerank=use_reranker)
        docs = res["documents"][0]
        metas = res["metadatas"][0]
        dists = res["distances"][0]
        idx = self._docs_by_id()

        out = []
        for rank, (doc, meta, dist) in enumerate(zip(docs, metas, dists), start=1):
            doc_id = meta.get("document_id", "")
            pages = _parse_pages(meta.get("pages"))
            primary = meta.get("page")
            try:
                primary = int(primary) if primary is not None else (pages[0] if pages else None)
            except (ValueError, TypeError):
                primary = pages[0] if pages else None
            label = _doc_label(idx[doc_id]) if doc_id in idx else (doc_id or "?")
            has_pages = doc_id in idx and bool(_load_pages(idx[doc_id]))
            snippet = (doc or "").strip()
            if len(snippet) > 320:
                snippet = snippet[:320].rstrip() + "…"
            out.append(
                {
                    "rank": rank,
                    "score": round(1.0 - float(dist), 4),
                    "document_id": doc_id,
                    "label": label,
                    "page": primary,
                    "pages": pages,
                    "snippet": snippet,
                    "chunk_id": meta.get("chunk_id", ""),
                    "has_pages": has_pages,
                }
            )
        return self._send_json(
            {
                "query": query,
                "collection": self.collection_name,
                "reranker": use_reranker,
                "results": out,
            }
        )

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
        query = (payload.get("query") or "").strip()
        answer = (payload.get("golden_answer") or "").strip()  # opsiyonel
        tags = payload.get("tags") or []
        edit_id = (payload.get("edit_id") or "").strip()  # varsa ID üzerinden güncelle

        # İlgili sayfaları (document_id, page) çiftleri olarak topla. Üç giriş
        # biçimi kabul edilir:
        #   marks:          [{document_id, page}, ...]          (retrieval akışı, çok-belge)
        #   relevant_pages: [{document_id, pages:[...]}, ...]   (düzenleme yeniden gönderimi)
        #   document_id + pages                                 (eski manuel akış, tek belge)
        rel_by_doc: dict[str, set[int]] = {}
        order: list[str] = []  # belge sırası (birincil = ilk)

        def _add(doc_id: str, page) -> bool:
            doc_id = (doc_id or "").strip()
            if not doc_id:
                return True
            try:
                pg = int(page)
            except (ValueError, TypeError):
                return False
            if doc_id not in rel_by_doc:
                rel_by_doc[doc_id] = set()
                order.append(doc_id)
            rel_by_doc[doc_id].add(pg)
            return True

        ok_parse = True
        if payload.get("marks"):
            for m in payload["marks"]:
                ok_parse &= _add(m.get("document_id"), m.get("page"))
        elif payload.get("relevant_pages"):
            for rp in payload["relevant_pages"]:
                for p in rp.get("pages", []):
                    ok_parse &= _add(rp.get("document_id"), p)
        else:  # eski manuel akış
            for p in payload.get("pages") or []:
                ok_parse &= _add(payload.get("document_id"), p)

        if not ok_parse:
            return self._send_json({"error": "page tam sayı olmalı."}, 400)
        if not query or not rel_by_doc:
            return self._send_json(
                {"error": "query ve en az bir (document_id, page) işareti zorunlu."}, 400
            )

        idx = self._docs_by_id()

        # Sayfa doğrulaması: sidecar'ı olan belgeler için sayfaların gerçekten
        # var olduğunu kontrol et; sidecar'ı olmayan belgeleri uyarıyla geç.
        warnings: list[str] = []
        page_texts: list[str] = []
        for doc_id in order:
            doc = idx.get(doc_id)
            pages_int = sorted(rel_by_doc[doc_id])
            if doc is None:
                warnings.append(f"{doc_id}: belge keşfedilemedi, sayfa doğrulanmadı")
                continue
            page_map = {p["sayfa_no"]: p["sayfa_markdown"] for p in _load_pages(doc)}
            if not page_map:
                warnings.append(f"{doc_id}: sayfa sidecar'ı yok, doğrulanmadı")
                continue
            missing_pages = [p for p in pages_int if p not in page_map]
            if missing_pages:
                return self._send_json(
                    {"error": f"{doc_id}: sayfa sidecar'da yok: {missing_pages}"}, 400
                )
            page_texts.extend(page_map[p] for p in pages_int)

        # canlı cevap doğrulaması (yalnızca cevap girildiyse)
        if answer and page_texts:
            ok, overlap, missing_spans = answer_in_pages(answer, page_texts)
            if not ok:
                parts = []
                if missing_spans:
                    parts.append(f"sayfada bulunamayan sayısal ifade: {missing_spans}")
                parts.append(f"metin örtüşmesi düşük (%{overlap * 100:.0f})")
                warnings.append("Cevap atıfta bulunulan sayfada doğrulanamadı: " + "; ".join(parts))
        warning = "; ".join(warnings) if warnings else None

        # relevant_pages: belge başına bir girdi (birincil belge ilk sırada)
        relevant_pages = [
            {"document_id": doc_id, "pages": sorted(rel_by_doc[doc_id])}
            for doc_id in order
        ]
        total_pages = sum(len(v) for v in rel_by_doc.values())
        primary_doc = order[0]

        # otomatik etiketler
        final_tags = list(dict.fromkeys(["tbmm-minutes", "page-level", *tags]))
        if total_pages > 1 and "cross-page" not in final_tags:
            final_tags.append("cross-page")
        if len(order) > 1 and "cross-document" not in final_tags:
            final_tags.append("cross-document")

        items = _load_fixture(self.fixture_path)

        entry = {
            "query": query,
            "relevant_pages": relevant_pages,
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
                entry["id"] = items[existing_idx].get("id") or _next_id(items, primary_doc)
                items[existing_idx] = {"id": entry["id"], **entry}
                action = "updated"
            else:
                new_id = _next_id(items, primary_doc)
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
        "--collection",
        default="tutanaklar_nomic_chunk256_768d",
        help="Retrieval havuzu için üretim koleksiyonu (models.yaml). Vars: tutanaklar_nomic_chunk256_768d",
    )
    ap.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Opsiyonel: manifest JSON. Belirtilmezse data_lake/reports + pages otomatik taranır.",
    )
    args = ap.parse_args()

    Handler.fixture_path = args.fixture
    Handler.manifest_path = args.manifest  # None → otomatik keşif
    Handler.collection_name = args.collection
    Handler._docs_cache = None  # önbelleği sıfırla
    Handler._docs_index = None
    Handler._retriever = None
    Handler._retriever_err = None

    # Belge listesini başlangıçta keşfet ve göster
    docs = _discover_documents() if args.manifest is None else _load_manifest(args.manifest)
    print(f"Golden Builder → http://{args.host}:{args.port}")
    print(f"Fixture    : {args.fixture}")
    print(f"Koleksiyon : {args.collection} (retrieval havuzu — ilk aramada yüklenir)")
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
