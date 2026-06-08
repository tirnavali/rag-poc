"""
Docling Pipeline Deneme Scripti
------------------------------
Bu script, 'tutanak/raw' klasöründeki PDF'leri tarar,
klasör yapısından metadata'yı okur ve ChromaDB'ye (docling_test koleksiyonuna) kaydeder.
"""

import os
import sys
from pathlib import Path

# Proje kök dizinini yola ekle
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.config import settings
from src.config.collections import CollectionSpec
from src.config.document_types import DocumentType
from src.trainer.ingestion.pipeline import IngestionPipeline


def main():
    # PDF'lerin olduğu ana klasör
    source_dir = PROJECT_ROOT / "tutanak" / "raw"
    
    if not source_dir.exists():
        print(f"Hata: {source_dir} klasörü bulunamadı. Lütfen PDF'leri yerleştirin.")
        return

    # Deneme için ayrı bir koleksiyon adı kullanıyoruz
    # Eğer gerçek koleksiyona yazmak istersen 'tbmm_minutes' olarak değiştirebilirsin.
    collection_name = "tbmm_minutes_docling_test"
    
    print(f"\n--- Ingestion Başlatılıyor ---")
    print(f"Kaynak: {source_dir}")
    print(f"Koleksiyon: {collection_name}\n")
    
    spec = CollectionSpec(
        name=collection_name,
        db_path=settings.MINUTES_CHROMA,
        embed_model="jinaai/jina-embeddings-v3",
        doc_type=DocumentType.TUTANAK,
    )
    pipeline = IngestionPipeline(spec=spec)
    
    # Batch işlemi başlat
    # Pipeline otomatik olarak alt klasörlerdeki .pdf dosyalarını bulacak
    # ve klasör isimlerinden (D20, Y1, B1_...) metadata çıkaracaktır.
    total_chunks = pipeline.run_batch(str(source_dir), extensions=[".pdf"])
    
    print(f"\n--- İşlem Tamamlandı ---")
    print(f"Toplam parça sayısı: {total_chunks}")
    print(f"Vektör veritabanı: {pipeline.db_path}")


if __name__ == "__main__":
    main()
