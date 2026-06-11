#!/usr/bin/env python3
"""Salt-okunur QA/lint aracı: golden benchmark fixture'larını sayfa sidecar'larına
karşı doğrular. ÜRETİM motoru DEĞİLDİR — yalnızca denetler.

Kontroller (her öğe için):
  - şema: id / query / relevant_pages / golden_answer / tags alanları ve tipleri
  - document_id ingestion manifest'indeki 10 kimlikten biri mi
  - relevant_pages.pages değerleri o birleşimin sidecar'ında GERÇEKTEN var mı
    (örn. sitting 1'de 33/35 sayfaları yok → yakalanır)
  - golden_answer atfedilen sayfa(lar)ın metninde geçiyor mu
    (token-overlap + sayısal/esas-no için birebir alt-dize)
  - dairesel/meta soru ("nerede geçiyor", "kaçıncı sayfada", ...)
  - cevap sadece sayfa numarası mı ("Sayfa 202")
  - boş cevap, yinelenen id, yinelenen/yakın-yinelenen soru
  - id deseni: tbmm27-01-<SS>-<NNN>

Kullanım:
  python scripts/lint_golden.py --fixture tests/fixtures/golden_tbmm27001001.json
  python scripts/lint_golden.py --fixture X.json --list-flag answer_not_on_cited_page
  python scripts/lint_golden.py --fixture X.json --report   # exit 0 her zaman

Importable: load_doc_pages(), answer_in_pages(), lint_item(), lint_fixture().
"""
from __future__ import annotations

import argparse
import difflib
import json
import os
import re
import sys
from collections import Counter, defaultdict
from glob import glob
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = ROOT / "27_donem_01_ingestion.json"
PAGES_DIR = ROOT / "data_lake" / "pages"
REPORTS_DIR = ROOT / "data_lake" / "reports"

ID_RE = re.compile(r"^tbmm27-01-\d{2}-\d{3}$")
ESAS_RE = re.compile(r"\b\d+\s*/\s*\d+\b")            # 2/940, 6/64, 10/1959
NUM_KEY_RE = re.compile(r"\b\d[\d./:]*\d\b|\b\d\b")   # times, years, ids, plain ints
META_RE = re.compile(
    r"nerede\s+ge[çc]iyor|ka[çc][ıi]nc[ıi]\s+sayfa|hangi\s+sayfa|"
    r"sayfa\s+numaras[ıi]\s+ka[çc]|ka[çc][ıi]nc[ıi]\s+sat[ıi]r",
    re.IGNORECASE,
)
ANSWER_IS_PAGE_RE = re.compile(r"^\s*sayfa\s*[:\-]?\s*\d+\s*\.?\s*$", re.IGNORECASE)

# Kalite (içerik) kuralları — "tüm arşiv testi": soru içeriğiyle kendini
# tanımlamalı, metadata/SQL ile cevaplanmamalı, ordinal-birleşime bağlı olmamalı.
QUALITY_PATTERNS = {
    # "4. Birleşimde", "5'inci Birleşim", "2. Birleşim sonunda" gibi ordinal çıpa
    "ordinal_sitting": re.compile(
        r"\b\d+\s*['’]?\s*(?:'?(?:nci|ncı|üncü|uncu|inci|ıncı)|\.)\s*birleşim", re.I
    ),
    # metadata: "X hangi ilin milletvekilidir" (SQL/tablo lookup)
    "deputy_province": re.compile(r"hangi\s+il(?:in|)\s.*milletvekil|hangi\s+ilin\s+milletvekil", re.I),
    # metadata/usul: oturum açılış saati, toplanma günü/saati kararı
    "scheduling_time": re.compile(
        r"saat\s+ka[çc]ta\s+a[çc][ıi]l|ka[çc]ta\s+toplan|hangi\s+g[üu]n.*toplan|"
        r"toplanma\s+karar|birleşimin.*oturumu\s+saat", re.I
    ),
    # usul minutiae: kâtip üye / idare amiri kim
    "procedural_role": re.compile(r"(kâtip\s+üye|idare\s+amir).*kim|kim.*(kâtip\s+üye|idare\s+amir)", re.I),
}


def quality_flags(query: str) -> list[str]:
    return [name for name, rx in QUALITY_PATTERNS.items() if rx.search(query or "")]

# Hard-fail flag adları (final fixture'da bunların hiçbiri kalmamalı)
HARD_FLAGS = {
    "missing_field",
    "bad_relevant_pages",
    "bad_document_id",
    "page_missing_in_sidecar",
    "duplicate_id",
    "bad_id_pattern",
    "answer_is_page_number",
    "empty_answer",
    "circular_meta_query",
    "numeric_span_not_on_page",
    "answer_not_on_cited_page",
    "duplicate_query",
}


# --------------------------------------------------------------------------- #
# Sidecar / manifest yükleme
# --------------------------------------------------------------------------- #
def _pages_path_for(doc_id: str, document_source: str) -> Path | None:
    """document_id için pages sidecar yolunu çözer.

    Öncelik 1 (sağlam): reports/{document_id}.json içindeki `artifacts.pages` —
    ingestion'ın yazdığı resmi index. Kırılgan türetme yok.
    Öncelik 2 (geri uyum): manifest'in document_source PDF adından stem türetip
    pages/ altında glob (örn. .../tbmm27001001.pdf -> tbmm27001001__*_pages.json).
    """
    safe_id = doc_id.replace("/", "_").replace("\\", "_")
    report = REPORTS_DIR / f"{safe_id}.json"
    if report.exists():
        try:
            arts = (json.loads(report.read_text(encoding="utf-8")) or {}).get("artifacts")
            if arts and arts.get("pages"):
                p = Path(arts["pages"])
                if not p.is_absolute():
                    p = ROOT / p
                if p.exists():
                    return p
        except Exception:
            pass  # bozuk rapor -> türetmeye düş

    stem = os.path.basename(document_source or "").replace(".pdf", "")
    if stem:
        matches = sorted(glob(str(PAGES_DIR / f"{stem}__*_pages.json")))
        if matches:
            return Path(matches[0])
    return None


def load_doc_pages(manifest_path: Path = DEFAULT_MANIFEST) -> dict[str, dict[int, str]]:
    """{document_id: {sayfa_no: sayfa_markdown}} döndürür.

    Sayfa sidecar'ı önce reports/{document_id}.json index'inden, yoksa manifest'in
    document_source PDF adından türetilen stem glob'undan çözülür (_pages_path_for).
    Sayfa numaraları sayfa_no'ya göre anahtarlanır (liste indeksine göre DEĞİL).
    """
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    doc_pages: dict[str, dict[int, str]] = {}
    for doc in manifest["documents"]:
        doc_id = doc["document_id"]
        pages_path = _pages_path_for(doc_id, doc.get("document_source", ""))
        if pages_path is None:
            doc_pages[doc_id] = {}
            continue
        entries = json.loads(pages_path.read_text(encoding="utf-8"))
        doc_pages[doc_id] = {
            int(e["sayfa_no"]): e.get("sayfa_markdown", "") for e in entries
        }
    return doc_pages


# --------------------------------------------------------------------------- #
# Metin eşleme yardımcıları
# --------------------------------------------------------------------------- #
def _norm(text: str) -> str:
    # casefold + Türkçe i-varyantlarını tek 'i'ye indir. casefold "İ"yi i + U+0307
    # birleşik noktaya çevirir (kaldırılır); dotless "ı" da 'i'ye katlanır. Böylece
    # büyük harfli "BARTIN"/"DİYARBAKIR" başlıkları "Bartın"/"Diyarbakır" ile eşleşir.
    t = (text or "").casefold().replace("̇", "").replace("ı", "i")
    return re.sub(r"\s+", " ", t).strip()


def _tokens(text: str) -> list[str]:
    toks = _norm(text).split()
    return [t.strip(".,;:!?'\"()[]…–—") for t in toks if t.strip(".,;:!?'\"()[]…–—")]


def _despace(text: str) -> str:
    return re.sub(r"\s+", "", text or "").casefold()


def answer_in_pages(answer: str, page_texts: list[str], threshold: float = 0.5):
    """(ok, overlap, missing_numeric_spans) döndürür.

    - Sayısal/esas-no anahtar ifadeleri (esas-no, saat, yıl, id) cevabın atfedildiği
      sayfaların metninde birebir geçmeli (boşluk-bağımsız karşılaştırma).
    - Kalan tokenlar için overlap oranı >= threshold olmalı.
    """
    page_norm = " ".join(_norm(p) for p in page_texts)
    page_nospace = _despace(" ".join(page_texts))

    # 1) Sayısal/esas-no anahtar ifadeleri
    spans = set(m.group().replace(" ", "") for m in ESAS_RE.finditer(answer))
    for m in NUM_KEY_RE.finditer(answer):
        s = m.group()
        if len(re.sub(r"\D", "", s)) >= 2:  # en az 2 rakam -> anlamlı (yıl, id, saat)
            spans.add(s.replace(" ", ""))
    missing = [s for s in spans if _despace(s) not in page_nospace]

    # 2) Token overlap (sayısal olmayan içerik)
    toks = _tokens(answer)
    if not toks:
        return (False, 0.0, missing)
    hits = sum(1 for t in toks if t in page_norm)
    overlap = hits / len(toks)
    ok = (overlap >= threshold) and not missing
    return (ok, overlap, missing)


# --------------------------------------------------------------------------- #
# Öğe / fixture lint
# --------------------------------------------------------------------------- #
def lint_item(
    item: dict,
    doc_pages: dict[str, dict[int, str]],
    hard_overlap: float = 0.5,
    soft_overlap: float = 0.7,
) -> list[str]:
    flags: list[str] = []

    qid = item.get("id", "")
    query = item.get("query", "")
    answer = item.get("golden_answer", "")
    rel = item.get("relevant_pages")

    if not qid or not query or rel is None or "golden_answer" not in item or "tags" not in item:
        flags.append("missing_field")
    if qid and not ID_RE.match(qid):
        flags.append("bad_id_pattern")
    if not (answer or "").strip():
        flags.append("empty_answer")
    if ANSWER_IS_PAGE_RE.match(answer or ""):
        flags.append("answer_is_page_number")
    if META_RE.search(query or ""):
        flags.append("circular_meta_query")

    # relevant_pages yapısı + sayfa varlığı
    page_texts: list[str] = []
    if not isinstance(rel, list) or not rel:
        flags.append("bad_relevant_pages")
    else:
        for pe in rel:
            if not isinstance(pe, dict) or "document_id" not in pe or "pages" not in pe:
                flags.append("bad_relevant_pages")
                continue
            doc_id = pe["document_id"]
            if doc_id not in doc_pages:
                flags.append("bad_document_id")
                continue
            pages = pe["pages"]
            if not isinstance(pages, list) or not pages:
                flags.append("bad_relevant_pages")
                continue
            for p in pages:
                try:
                    p_int = int(float(p))
                except (ValueError, TypeError):
                    flags.append("bad_relevant_pages")
                    continue
                if p_int not in doc_pages[doc_id]:
                    flags.append("page_missing_in_sidecar")
                else:
                    page_texts.append(doc_pages[doc_id][p_int])

    # cevap sayfada mı
    if page_texts and (answer or "").strip() and "answer_is_page_number" not in flags:
        ok, overlap, missing = answer_in_pages(answer, page_texts, threshold=hard_overlap)
        if missing:
            flags.append("numeric_span_not_on_page")
        if overlap < hard_overlap:
            flags.append("answer_not_on_cited_page")
        elif overlap < soft_overlap:
            flags.append("low_overlap")  # soft

    return flags


def lint_fixture(items: list[dict], doc_pages: dict[str, dict[int, str]], **kw):
    """{qid: flags} + global yinelenen id/soru bayrakları döndürür."""
    result: dict[str, list[str]] = {}
    seen_ids: set[str] = set()
    norm_queries: dict[str, str] = {}  # norm_query -> first id

    for item in items:
        qid = item.get("id", f"<no-id:{len(result)}>")
        flags = lint_item(item, doc_pages, **kw)

        if qid in seen_ids:
            flags.append("duplicate_id")
        seen_ids.add(qid)

        nq = _norm(item.get("query", ""))
        if nq:
            if nq in norm_queries:
                flags.append("duplicate_query")
            else:
                norm_queries[nq] = qid

        result[qid] = flags

    # yakın-yinelenen sorular (soft): difflib — tüm çiftler, pencere yok
    norm_list = list(norm_queries.items())
    for i, (nq, qid) in enumerate(norm_list):
        for nq2, qid2 in norm_list[i + 1 :]:
            if difflib.SequenceMatcher(None, nq, nq2).ratio() >= 0.92:
                if "duplicate_query" not in result[qid2]:
                    result[qid2].append("near_duplicate_query")  # soft
    return result


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fixture", required=True, type=Path)
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--report", action="store_true", help="her zaman exit 0")
    ap.add_argument("--list-flag", help="bu bayrağı taşıyan öğeleri dök")
    ap.add_argument("--hard-overlap", type=float, default=0.5)
    ap.add_argument("--soft-overlap", type=float, default=0.7)
    args = ap.parse_args()

    doc_pages = load_doc_pages(args.manifest)
    items = json.loads(args.fixture.read_text(encoding="utf-8"))
    flagmap = lint_fixture(
        items, doc_pages, hard_overlap=args.hard_overlap, soft_overlap=args.soft_overlap
    )

    by_id = {it.get("id"): it for it in items}
    counts = Counter()
    hard_ids: set[str] = set()
    for qid, flags in flagmap.items():
        for f in flags:
            counts[f] += 1
            if f in HARD_FLAGS:
                hard_ids.add(qid)

    print(f"Fixture: {args.fixture}  ({len(items)} öğe)")
    print(f"Sidecar belge sayısı: {len(doc_pages)}  "
          f"(toplam sayfa: {sum(len(v) for v in doc_pages.values())})")
    # birleşim başına dağılım
    per_doc = Counter()
    per_cat = Counter()
    for it in items:
        for pe in it.get("relevant_pages", []) or []:
            per_doc[pe.get("document_id")] += 1
            break
        for t in it.get("tags", []):
            if t in {"narrative", "deputy", "bulletin"}:
                per_cat[t] += 1
        if "cross-page" in it.get("tags", []):
            per_cat["cross-page"] += 1
    print("Birleşim başına öğe:", dict(sorted(per_doc.items(), key=lambda x: str(x[0]))))
    print("Kategori dağılımı:", dict(per_cat))
    print("-" * 60)
    if not counts:
        print("Bayrak yok — temiz.")
    else:
        for f, c in counts.most_common():
            mark = "HARD" if f in HARD_FLAGS else "soft"
            print(f"  [{mark}] {f}: {c}")

    if args.list_flag:
        print("-" * 60)
        print(f"'{args.list_flag}' bayraklı öğeler:")
        for qid, flags in flagmap.items():
            if args.list_flag in flags:
                it = by_id.get(qid, {})
                pages = [p for pe in it.get("relevant_pages", []) for p in pe.get("pages", [])]
                print(f"  {qid} p{pages}: {it.get('query','')[:80]}")
                print(f"      A: {(it.get('golden_answer') or '')[:80]}")

    print("-" * 60)
    n_hard = len(hard_ids)
    print(f"HARD hata içeren öğe: {n_hard}/{len(items)}")
    if args.report:
        return 0
    return 1 if n_hard else 0


if __name__ == "__main__":
    sys.exit(main())
