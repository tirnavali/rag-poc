"""Backfill: mevcut tutanak korpusuna metadata omurgasını YENİDEN-EMBED ETMEDEN ekle:
(1) ``chunk_index`` (evrensel okuma sırası, TÜM chunk'lara, id son ekinden) — reflect
window-expand + reading-order kullanır; (2) kanun etiketleri (``sira_sayisi`` + ``esas_no``
+ ``kanun_adi``, yalnız kanun bölgelerine); (3) bölüm etiketleri (``section_type`` +
``section_ord`` + ``section_path``, HER chunk'a — ``tag_sections``). Bölüm dağılımı özeti
Docling başlık kapsamının ölçümüdür (Faz 1 karar noktası; yüksek None/diger = güvenilmez).

``collection.update(ids, metadatas)`` yalnızca metadata'ya dokunur (embeddings /
documents verilmez) → on binlerce chunk'ı yeniden embed etmeden etiketleriz. Ingest
yolu ``upsert`` eder (ve yeniden embed eder); backfill ``update`` eder — repoda ilk
``collection.update`` kullanımı. Etiketleme çekirdeği ingest ile PAYLAŞILIR
(``tag_law_regions``) → backfill ve gelecek ingest'ler aynı sonucu verir.

Neden gerekli: TBMM açık-oylama roll-call tabloları kanunu adıyla da esas no'suyla da
anmaz; yalnız "S.S. N" işareti taşır → semantik arama ulaşamaz. sira_sayisi metadata'sı
reflect Hop-2'yi kesin ``where={'sira_sayisi': N}`` filtresine çevirir.

Kullanım:
  python -m scripts.backfill_law_metadata --dry-run                    # kapsam istatistiği
  python -m scripts.backfill_law_metadata                              # yaz (production tutanak)
  python -m scripts.backfill_law_metadata --collection <key>           # başka koleksiyon
  python -m scripts.backfill_law_metadata --document <id> --dry-run    # nokta kontrol

Idempotent: iki kez çalıştır → aynı sonuç (tag deterministik; ``{**eski, **yeni}`` aynı
değerlerle üzerine yazar).
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import Counter

from src.common.chroma import open_or_create_collection
from src.config.collections import get_spec
from src.trainer.ingestion.law_region_tagger import tag_law_regions, tag_sections

DEFAULT_COLLECTION = "tutanaklar_nomic_chunk256_768d"  # production_ready tutanak (models.yaml)

_ORDER_RE = re.compile(r"^(.+)_(\d+)$")  # span_resolver.py:46 deseni


def _order_key(chunk_id: str) -> int:
    """Chunk id sonekine göre sıralama anahtarı (``{document_id}_{chunk_index}``)."""
    m = _ORDER_RE.match(chunk_id or "")
    return int(m.group(2)) if m else -1


def _sanitize(val):
    """Pipeline metadata sanitizasyonu (pipeline.py:378-393) aynası — ChromaDB yalnız
    str/int/float/bool kabul eder. Tag değerleri zaten int (sira_sayisi) | str (esas_no,
    kanun_adi) | None; None'lar çağrıdan önce elenir."""
    if isinstance(val, (str, int, float, bool)):
        return val
    return str(val)


def _distinct_document_ids(col) -> list[str]:
    """Koleksiyondaki tüm distinct document_id'leri topla (yalnız metadata çeker)."""
    got = col.get(include=["metadatas"])
    ids = {
        (m or {}).get("document_id")
        for m in (got.get("metadatas") or [])
        if m and m.get("document_id")
    }
    return sorted(i for i in ids if i)


def backfill(collection_key: str, *, dry_run: bool, only_document: str | None) -> int:
    spec = get_spec(collection_key)
    _client, col = open_or_create_collection(spec.db_path, spec.name)

    doc_ids = _distinct_document_ids(col)
    if only_document:
        doc_ids = [d for d in doc_ids if d == only_document]
        if not doc_ids:
            print(f"[backfill] belge bulunamadı: {only_document!r} (koleksiyon {collection_key})")
            return 1

    total_docs = len(doc_ids)
    docs_with_labels = 0
    labeled_chunks = 0
    indexed_chunks = 0   # chunk_index (okuma sırası) yazılan chunk sayısı (~tümü)
    total_chunks = 0
    esas_dist: Counter = Counter()
    section_dist: Counter = Counter()  # section_type dağılımı (None = etiketsiz)
    region_count = 0  # distinct sira_sayisi bölgeleri (belge×sıra sayısı)
    unmatched_docs: list[str] = []

    mode = "DRY-RUN" if dry_run else "YAZ"
    print(f"[backfill] koleksiyon={collection_key} db={spec.db_path} mod={mode} belge={total_docs}")

    for did in doc_ids:
        d = col.get(where={"document_id": did}, include=["documents", "metadatas"])
        rows = sorted(
            zip(d["ids"], d["documents"], d["metadatas"]),
            key=lambda r: _order_key(r[0]),
        )
        if not rows:
            continue
        ids = [r[0] for r in rows]
        texts = [r[1] or "" for r in rows]
        metas = [r[2] or {} for r in rows]
        total_chunks += len(rows)

        tags = tag_law_regions(texts)
        section_tags = tag_sections(texts)

        upd_ids: list[str] = []
        upd_metas: list[dict] = []
        doc_siras: set[int] = set()
        doc_labeled = 0
        for cid, old, tag, stag in zip(ids, metas, tags, section_tags):
            add: dict = {}
            idx = _order_key(cid)
            if idx >= 0:
                add["chunk_index"] = idx  # evrensel okuma sırası (id son ekinden; int)
            law_add = {k: _sanitize(v) for k, v in tag.items() if v is not None}
            add.update(law_add)
            add.update({k: _sanitize(v) for k, v in stag.items() if v is not None})
            section_dist[stag.get("section_type")] += 1
            if not add:
                continue  # yalnız bozuk-id + etiketsiz (nadir) → güncelleme yok
            merged = {**old, **add}
            upd_ids.append(cid)
            upd_metas.append(merged)
            if law_add:
                doc_labeled += 1
            if tag.get("sira_sayisi") is not None:
                doc_siras.add(tag["sira_sayisi"])
            if tag.get("esas_no"):
                esas_dist[tag["esas_no"]] += 1

        indexed_chunks += len(upd_ids)
        labeled_chunks += doc_labeled
        if doc_labeled:
            docs_with_labels += 1
            region_count += len(doc_siras)
        else:
            unmatched_docs.append(did)  # kanun etiketi yok (chunk_index yine yazıldı)
        if upd_ids and not dry_run:
            col.update(ids=upd_ids, metadatas=upd_metas)

        if only_document:
            _print_document_detail(did, ids, texts, tags, section_tags)

    # ── Kapsam özeti ────────────────────────────────────────────────
    pct = (100.0 * labeled_chunks / total_chunks) if total_chunks else 0.0
    ipct = (100.0 * indexed_chunks / total_chunks) if total_chunks else 0.0
    print("\n[backfill] ÖZET")
    print(f"  chunk_index yazıldı: {indexed_chunks}/{total_chunks} (%{ipct:.1f}) — evrensel okuma sırası")
    print(f"  belge: {docs_with_labels}/{total_docs} kanun etiketli bölge içeriyor")
    print(f"  kanun chunk: {labeled_chunks}/{total_chunks} etiketlendi (%{pct:.1f})")
    print(f"  bölge (belge×sıra sayısı): {region_count}")
    print(f"  esas_no dağılımı (en sık 10): {dict(esas_dist.most_common(10))}")
    # Bölüm (section_type) histogramı — Docling başlık kapsamı ölçümü. Yüksek
    # None/diger oranı = güvenilmez başlık ya da eksik enum deseni (Faz 1 karar noktası).
    sec_none = section_dist.get(None, 0)
    sec_labeled = total_chunks - sec_none
    spct = (100.0 * sec_labeled / total_chunks) if total_chunks else 0.0
    dpct = (100.0 * section_dist.get("diger", 0) / total_chunks) if total_chunks else 0.0
    print(f"  section_type: {sec_labeled}/{total_chunks} etiketli (%{spct:.1f}), "
          f"None=%{100.0 - spct:.1f}, diger=%{dpct:.1f}")
    sec_items = sorted(
        ((k or "None", v) for k, v in section_dist.items()),
        key=lambda kv: kv[1], reverse=True,
    )
    print(f"  section_type dağılımı: {dict(sec_items)}")
    if unmatched_docs:
        print(f"  eşleşmeyen belge: {len(unmatched_docs)} (ör. {unmatched_docs[:5]})")
    if dry_run:
        print("  [DRY-RUN — hiçbir şey yazılmadı; gerçek yazım için --dry-run'ı kaldırın]")
    return 0


def _print_document_detail(did: str, ids, texts, tags, section_tags) -> None:
    """--document nokta kontrolü: etiketli chunk aralıklarını + roll-call örneğini yazdır."""
    labeled = [
        (_order_key(cid), t["sira_sayisi"], t["esas_no"])
        for cid, t in zip(ids, tags) if t["sira_sayisi"] is not None
    ]
    print(f"\n  --- {did}: {len(labeled)}/{len(ids)} chunk kanun-etiketli ---")
    # Bölüm (section_type) omurgası: idx-sıralı bölüm aralıkları (Docling başlık ölçümü).
    sec_rows = sorted(
        ((_order_key(cid), s.get("section_type")) for cid, s in zip(ids, section_tags)),
        key=lambda r: r[0],
    )
    by_sec: Counter = Counter(st for _, st in sec_rows)
    print(f"    section_type dağılımı: {dict(by_sec)}")
    # Ardışık bölüm aralıklarını sıkıştır → belgenin bölüm iskeletini göster.
    runs: list[tuple[str, int, int]] = []
    for idx, st in sec_rows:
        label = st or "None"
        if runs and runs[-1][0] == label:
            runs[-1] = (label, runs[-1][1], idx)
        else:
            runs.append((label, idx, idx))
    print("    bölüm iskeleti (idx aralığı → section_type):")
    for label, lo, hi in runs:
        span = f"_{lo}" if lo == hi else f"_{lo}.._{hi}"
        print(f"      {span}: {label}")
    if not labeled:
        print("    (kanun etiketi yok)")
        return
    by_sira: Counter = Counter(s for _, s, _ in labeled)
    print(f"    sira_sayisi dağılımı: {dict(by_sira)}")
    idxs = sorted(i for i, _, _ in labeled)
    print(f"    kanun-etiketli idx aralığı: {idxs[0]}..{idxs[-1]}")
    # roll-call bölgesi örneği (varsa)
    sample = [(i, s, e) for i, s, e in sorted(labeled) if i >= idxs[0]][:6]
    for i, s, e in sample:
        print(f"    _{i}: sira_sayisi={s} esas_no={e}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Kanun metadata omurgası backfill (re-embed YOK)")
    ap.add_argument("--collection", default=DEFAULT_COLLECTION,
                    help=f"models.yaml koleksiyon anahtarı (varsayılan {DEFAULT_COLLECTION})")
    ap.add_argument("--dry-run", action="store_true",
                    help="yazmadan kapsam istatistiği üret")
    ap.add_argument("--document", default=None,
                    help="yalnız bu document_id'yi işle (nokta kontrol)")
    args = ap.parse_args()
    return backfill(args.collection, dry_run=args.dry_run, only_document=args.document)


if __name__ == "__main__":
    sys.exit(main())
