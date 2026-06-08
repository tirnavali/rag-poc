"""Embed and index newspaper press clips into ChromaDB."""
from __future__ import annotations

import sqlite3
from pathlib import Path

from tqdm import tqdm

from src.common.chroma import open_or_create_collection
from src.common.chunking import build_text_splitter
from src.common.dates import extract_year, normalize_iso_date
from src.common.embeddings import build_embedder, ensure_ollama_model
from src.config import settings


def embed_and_index(
    db_file: Path = settings.PRESS_SQLITE,
    chroma_path: Path = settings.PRESS_CHROMA,
    collection_name: str = settings.PRESS_COLLECTION,
) -> None:
    ensure_ollama_model(settings.EMBED_MODEL)
    chroma_path.mkdir(parents=True, exist_ok=True)

    print("--- ChromaDB istemcisi başlatılıyor ---")
    _, collection = open_or_create_collection(chroma_path, collection_name)

    print(f"--- {db_file} verileri okunuyor ---")
    conn = sqlite3.connect(str(db_file))
    cursor = conn.cursor()

    # Stream rows via fetchmany so memory stays flat at 30M+ rows.
    # (Earlier the whole table was fetchall'd which OOM'd large archives.)
    total_rows = cursor.execute("SELECT COUNT(*) FROM kupurler").fetchone()[0]
    cursor.execute(
        "SELECT KAYIT_NO, GAZETE_ADI, TARIH, BASLIK, YAZARLAR, DOKUMAN_METNI, METADATA_KONULAR "
        "FROM kupurler"
    )

    embedder = build_embedder()
    splitter = build_text_splitter(settings.PRESS_CHUNK_SIZE, settings.PRESS_CHUNK_OVERLAP)

    print(f"--- {total_rows} kayıt için embedding üretiliyor ---")
    pbar = tqdm(total=total_rows, unit="row")
    while True:
        batch = cursor.fetchmany(settings.EMBED_BATCH_SIZE)
        if not batch:
            break

        all_chunks: list[str] = []
        all_metas: list[dict] = []
        all_ids: list[str] = []

        for row in batch:
            k_no, g_adi, tarih, baslik, yazar, metin, konular = row
            tarih_clean = normalize_iso_date(tarih)
            tarih_year = extract_year(tarih_clean)
            prefix = f"Gazete: {g_adi} | Tarih: {tarih_clean} | Yazar: {yazar} | Başlık: {baslik}\n"
            chunks = splitter.split_text(metin)
            for j, chunk in enumerate(chunks):
                all_chunks.append(prefix + chunk)
                all_metas.append({
                    "gazete": g_adi,
                    "tarih": tarih_clean,
                    "tarih_year": tarih_year,
                    "yazar": yazar or "Bilinmiyor",
                    "baslik": baslik,
                    "konular": konular,
                    "chunk_index": j,
                })
                all_ids.append(f"{k_no}_{j}")

        if all_chunks:
            vecs = embedder.embed_documents(all_chunks)
            collection.upsert(
                embeddings=vecs,
                documents=all_chunks,
                metadatas=all_metas,
                ids=all_ids,
            )
        pbar.update(len(batch))

    pbar.close()
    print(f"--- İndeksleme Tamamlandı. Toplam Belge: {collection.count()} ---")
    conn.close()


if __name__ == "__main__":
    embed_and_index()
