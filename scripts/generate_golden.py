"""ChromaDB chunk'larından sıfırdan sentetik golden dataset üretir.

FARK: `generate_excerpts.py` mevcut bir veri setindeki sorulara sadece cevap (excerpt) eklerken, 
bu script herhangi bir başlangıç verisine ihtiyaç duymadan, rastgele metinler üzerinden 
hem araştırma sorusunu hem de cevabını sıfırdan üretir.

Her chunk için LLM ile hem soru hem verbatim excerpt otomatik üretilir.
Çıktı benchmark.py ile doğrudan kullanılabilir (excerpts matcher).

Kullanım:
    python scripts/generate_golden.py \\
        --collection tbmm_tutanaklar_docling_jina_v3_4k \\
        --n 20 \\
        --output tests/fixtures/golden_tutanak_auto.json

    # 3 örnek ekrana bas, dosyaya yazma:
    python scripts/generate_golden.py \\
        --collection tbmm_tutanaklar_docling_jina_v3_4k \\
        --n 3 \\
        --dry-run
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import chromadb
import ollama

from src.config import settings
from src.config.collections import get_spec

_PROMPT = """\
Sen TBMM tutanak arşivini araştıran bir gazeteci ya da akademisyensin.
Aşağıdaki tutanak bölümünü değerlendir.

ADIM 1 — İçerik kalite kontrolü:
Metin yalnızca şunlardan oluşuyorsa null döndür (başka hiçbir şey yazma):
- Gündem geçişi ("... kısmına geçiyoruz", "birleşim açıldı" vb.)
- Yoklama, oylama sayımı, usul işlemleri
- İçindekiler tablosu veya madde listeleri
- Çok kısa veya anlamsız OCR kalıntısı

ADIM 2 — Gerçek araştırma sorusu yaz:
Bu metni okumamış biri tarafından sorulabilecek GERÇEK bir araştırma sorusu yaz.

İyi soru kalıpları:
- "[Konu] hakkında mecliste ne tür eleştiriler dile getirildi?"
- "[Kurum / sektör] ile ilgili meclisteki tartışma ne yönde gelişti?"
- "Hangi sorunlar / talepler meclis gündemine taşındı?"
- "[X] konusundaki hükümet / muhalefet tutumu neydi?"

Yasaklar:
- Metindeki spesifik rakam, tarih, kişi adı SORU İÇİNDE KULLANMA
- "Konuşmacıya göre" / "metne göre" / "belgede" YASAK
- Gündem sırası / oturum numarası soran sorular YASAK

ADIM 3 — Verbatim pasaj çıkar:
Soruyu cevaplayan metindeki pasajı kelimesi kelimesine yaz.

Yanıt formatı — sadece JSON, başka hiçbir şey:
{{"query": "araştırma sorusu", "excerpts": ["verbatim pasaj"]}}

İçerik yetersizse:
null

Metin:
{chunk_text}
"""


def _sample_chunks(
    collection: chromadb.Collection,
    n: int,
    min_chars: int,
    seed: int | None,
) -> list[dict]:
    """Koleksiyondan rastgele n chunk çeker, kısa olanları filtreler."""
    total = collection.count()
    if total == 0:
        return []

    rng = random.Random(seed)
    offset = rng.randint(0, max(0, total - n * 3))
    fetch_n = min(n * 4, total)

    result = collection.get(
        limit=fetch_n,
        offset=offset,
        include=["documents", "metadatas"],
    )

    chunks = []
    for cid, doc, meta in zip(result["ids"], result["documents"], result["metadatas"]):
        if doc and len(doc.strip()) >= min_chars:
            chunks.append({"id": cid, "text": doc, "meta": meta})

    rng.shuffle(chunks)
    return chunks[:n]


def _normalize_ws(text: str) -> str:
    """OCR kaynaklı çoklu boşlukları normalize eder."""
    return re.sub(r"\s+", " ", text).strip()


def _excerpt_overlap(excerpt: str, chunk_text: str, threshold: float = 0.70) -> bool:
    """Excerpt tokenlarının en az threshold oranı chunk'ta geçiyorsa True döner.

    Exact string match yerine token overlap kullanır; LLM'in küçük
    parafraz veya büyük/küçük harf farkı üretmesi durumunda da çalışır.
    """
    exc_tokens = _normalize_ws(excerpt).lower().split()
    chunk_lower = _normalize_ws(chunk_text).lower()
    if not exc_tokens:
        return False
    hits = sum(1 for t in exc_tokens if t in chunk_lower)
    return hits / len(exc_tokens) >= threshold


def _parse_response(raw: str, chunk_text: str) -> dict | None:
    """LLM yanıtından query ve excerpts parse eder; hallucination kontrolü yapar."""
    raw = raw.strip()
    # LLM "null" döndürdüyse (prosedürel chunk reddi)
    if raw.lower().strip("` \n") == "null":
        return None
    # Markdown code block sarmalını kaldır
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    raw = raw.strip()

    match = re.search(r"\{.*?\}", raw, re.DOTALL)
    if not match:
        return None
    try:
        parsed = json.loads(match.group())
    except json.JSONDecodeError:
        return None

    query = parsed.get("query", "").strip()
    excerpts = parsed.get("excerpts", [])

    if not query or not excerpts:
        return None
    if not isinstance(excerpts, list):
        return None

    # Hallucination koruması: token overlap ≥ 70%
    valid = [
        e for e in excerpts
        if isinstance(e, str) and e.strip() and _excerpt_overlap(e, chunk_text)
    ]
    if not valid:
        return None

    return {"query": query, "excerpts": valid}


def generate_golden(
    collection_name: str,
    n: int,
    output_path: Path | None,
    model: str,
    min_chars: int,
    seed: int | None,
    dry_run: bool,
) -> None:
    """Koleksiyondan n chunk sample alır, LLM ile soru+excerpt üretir.

    Args:
        collection_name: collections.py'deki koleksiyon anahtarı
        n: Üretilmek istenen entry sayısı
        output_path: Çıktı JSON dosyası; None ise tests/fixtures/ altına yazar
        model: Ollama model adı
        min_chars: Chunk minimum karakter sayısı filtresi
        seed: Rastgele tohum (tekrarlanabilirlik için)
        dry_run: True ise dosyaya yazmaz, ekrana basar
    """
    spec = get_spec(collection_name)
    client = chromadb.PersistentClient(path=str(spec.db_path))
    collection = client.get_collection(name=spec.name)
    ollama_client = ollama.Client(host=settings.OLLAMA_HOST)

    print(f"Koleksiyon: {spec.name} ({collection.count()} chunk)")
    chunks = _sample_chunks(collection, n=n * 2, min_chars=min_chars, seed=seed)
    print(f"Filtrelenmiş sample: {len(chunks)} chunk (hedef: {n})")

    entries: list[dict] = []
    failed = 0

    for i, chunk in enumerate(chunks):
        if len(entries) >= n:
            break

        qid = chunk["id"]
        print(f"[{i+1}/{len(chunks)}] {qid} ...", end=" ", flush=True)

        prompt = _PROMPT.format(chunk_text=chunk["text"])
        try:
            res = ollama_client.chat(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.3},
            )
            raw = res.message.content
        except Exception as e:
            print(f"HATA: {e}")
            failed += 1
            continue

        parsed = _parse_response(raw, chunk["text"])
        if parsed is None:
            print("atlandı (parse/hallucination)")
            failed += 1
            continue

        entry = {
            "id": f"auto-{collection_name}-{len(entries)+1:03d}",
            "query": parsed["query"],
            "excerpts": parsed["excerpts"],
            "relevant_chunk_ids": [qid],
            "source_collection": collection_name,
        }

        # Metadata'dan tarih ve dönem bilgisi ekle
        meta = chunk["meta"]
        if meta.get("date"):
            entry["expected_year"] = meta.get("year")
        if meta.get("document_id"):
            entry["document_id"] = meta["document_id"]

        entries.append(entry)

        if dry_run:
            print(f"OK\n  Sorgu: {parsed['query']}")
            for j, ex in enumerate(parsed["excerpts"], 1):
                print(f"  Excerpt {j}: {ex[:150]}")
        else:
            print(f"OK — '{parsed['query'][:60]}...'")

    print(f"\nSonuç: {len(entries)} entry üretildi, {failed} başarısız")

    if dry_run:
        return

    dest = output_path or (
        Path("tests/fixtures") / f"golden_{collection_name}_auto.json"
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Yazıldı → {dest}")


def main() -> None:
    parser = argparse.ArgumentParser(description="ChromaDB chunk'larından golden dataset üret")
    parser.add_argument("--collection", required=True,
                        help="Koleksiyon anahtarı (collections.py'de tanımlı, örn: tbmm_tutanaklar_docling_jina_v3_4k)")
    parser.add_argument("--n", type=int, default=20,
                        help="Üretilecek entry sayısı (varsayılan: 20)")
    parser.add_argument("--output", default=None,
                        help="Çıktı JSON dosyası (varsayılan: tests/fixtures/golden_<koleksiyon>_auto.json)")
    parser.add_argument("--model", default=settings.LLM_MODEL,
                        help=f"Ollama model adı (varsayılan: {settings.LLM_MODEL})")
    parser.add_argument("--min-chars", type=int, default=300, dest="min_chars",
                        help="Minimum chunk karakter sayısı (varsayılan: 300)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Rastgele tohum (varsayılan: None — her çalıştırmada farklı)")
    parser.add_argument("--dry-run", action="store_true", dest="dry_run",
                        help="Dosyaya yazmadan ekrana bas")
    args = parser.parse_args()

    generate_golden(
        collection_name=args.collection,
        n=args.n,
        output_path=Path(args.output) if args.output else None,
        model=args.model,
        min_chars=args.min_chars,
        seed=args.seed,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
