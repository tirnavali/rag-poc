"""LLM ile mevcut altın veri (golden data) fixture'larına verbatim excerpt ekler.

FARK: `generate_golden.py` sıfırdan sentetik sorular ve cevaplar üretirken, 
bu script önceden hazırlanmış (örneğin insan eliyle yazılmış kaliteli sorular içeren) 
ancak cevap metinleri (excerpts) eksik olan veri setlerini zenginleştirmek için kullanılır.

Her sorgu için relevant_chunk_ids'ten chunk metinlerini çeker,
Ollama'ya prompt gönderir ve dönen pasajları `excerpts` alanı olarak yazar.

Kullanım:
    python scripts/generate_excerpts.py \\
        --fixture tests/fixtures/eval_queries_docling_d20.json \\
        --collection minutes_jina_v3 \\
        --output tests/fixtures/eval_queries_docling_d20_excerpts.json

    # Sadece ilk 2 sorguyu ekrana yaz, dosyaya yazma:
    python scripts/generate_excerpts.py \\
        --fixture tests/fixtures/eval_queries_docling_d20.json \\
        --collection minutes_jina_v3 \\
        --dry-run
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ollama

from src.config import settings
from src.config.collections import get_spec
from src.retriever.vector_search import VectorSearch

_PROMPT_TEMPLATE = """\
Sorgu: {query}

Aşağıdaki metin parçalarından yalnızca sorgunun cevabını doğrudan içeren \
verbatim pasajları çıkar. Metinde gerçekten geçen ifadeleri birebir aktar; \
ekleme veya parafraz yapma. Cevap bulunamıyorsa boş liste döndür.

Yanıtı yalnızca JSON listesi olarak ver, başka hiçbir şey yazma:
["pasaj1", "pasaj2"]

Metin:
{context}
"""


def _fetch_chunk_texts(chunk_ids: list[str], search: VectorSearch) -> list[str]:
    """Chunk ID listesinden metin içeriklerini ChromaDB'den çeker."""
    if not chunk_ids:
        return []
    try:
        result = search.collection.get(ids=chunk_ids, include=["documents"])
        return result.get("documents") or []
    except Exception as e:
        print(f"  [WARN] chunk fetch failed: {e}", file=sys.stderr)
        return []


def _parse_excerpts(raw: str) -> list[str]:
    """LLM çıktısından JSON listesi parse eder; hata durumunda boş liste döner."""
    raw = raw.strip()
    match = re.search(r"\[.*?\]", raw, re.DOTALL)
    if not match:
        return []
    try:
        parsed = json.loads(match.group())
        return [s for s in parsed if isinstance(s, str) and s.strip()]
    except json.JSONDecodeError:
        return []


def generate_excerpts(
    fixture_path: Path,
    collection_name: str,
    output_path: Path | None,
    model: str,
    fetch_k: int,
    dry_run: bool,
) -> None:
    """Fixture dosyasındaki her sorguya LLM ile excerpt ekler.

    Args:
        fixture_path: Girdi golden JSON dosyası
        collection_name: Chunk metinlerinin çekileceği koleksiyon adı
        output_path: Çıktı dosyası; None ise fixture_path üzerine yazar
        model: Ollama model adı
        fetch_k: Her sorgu için chunk sayısı (relevant_chunk_ids yoksa kullanılır)
        dry_run: True ise dosyaya yazmaz, ilk 2 sorguyu ekrana basar
    """
    queries: list[dict] = json.loads(fixture_path.read_text(encoding="utf-8"))
    spec = get_spec(collection_name)
    search = VectorSearch(spec)
    client = ollama.Client(host=settings.OLLAMA_HOST)

    limit = 2 if dry_run else len(queries)
    updated = 0

    for item in queries[:limit]:
        qid = item.get("id", "?")
        query = item.get("query", "")

        # Mevcut excerpts varsa atla
        if item.get("excerpts"):
            print(f"[SKIP] {qid} — excerpts zaten mevcut")
            continue

        # Chunk metinlerini çek
        chunk_ids = item.get("relevant_chunk_ids", [])
        if chunk_ids:
            texts = _fetch_chunk_texts(chunk_ids, search)
        else:
            # Chunk ID yoksa vector search ile doldur
            raw = search.search(query, top_k=fetch_k, fetch_k=fetch_k * 4)
            texts = [r["doc"] for r in raw]

        if not texts:
            print(f"[WARN] {qid} — chunk metni bulunamadı, atlanıyor")
            continue

        context = "\n\n---\n\n".join(texts)
        prompt = _PROMPT_TEMPLATE.format(query=query, context=context)

        try:
            res = client.chat(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.0},
            )
            raw_reply = res.message.content
        except Exception as e:
            print(f"[ERROR] {qid} — Ollama hatası: {e}", file=sys.stderr)
            continue

        excerpts = _parse_excerpts(raw_reply)
        item["excerpts"] = excerpts
        updated += 1

        if dry_run:
            print(f"\n=== {qid} ===")
            print(f"Sorgu: {query}")
            print(f"Excerpts ({len(excerpts)}):")
            for i, e in enumerate(excerpts, 1):
                print(f"  {i}. {e[:200]}")
        else:
            print(f"[OK] {qid} — {len(excerpts)} excerpt")

    if not dry_run:
        dest = output_path or fixture_path
        dest.write_text(json.dumps(queries, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n{updated} sorguya excerpt eklendi → {dest}")


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM ile fixture'lara verbatim excerpt ekle")
    parser.add_argument("--fixture", required=True, help="Girdi golden JSON dosyası")
    parser.add_argument("--collection", required=True, help="Koleksiyon adı (collections.py'de tanımlı)")
    parser.add_argument("--output", default=None, help="Çıktı dosyası (varsayılan: fixture üzerine yazar)")
    parser.add_argument("--model", default=settings.LLM_MODEL, help="Ollama model adı")
    parser.add_argument("--fetch-k", type=int, default=5, dest="fetch_k",
                        help="relevant_chunk_ids yoksa kaç chunk çekilsin (varsayılan: 5)")
    parser.add_argument("--dry-run", action="store_true", dest="dry_run",
                        help="Dosyaya yazmadan ilk 2 sorguyu ekrana bas")
    args = parser.parse_args()

    generate_excerpts(
        fixture_path=Path(args.fixture),
        collection_name=args.collection,
        output_path=Path(args.output) if args.output else None,
        model=args.model,
        fetch_k=args.fetch_k,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
