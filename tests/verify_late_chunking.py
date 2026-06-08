import sys
import os
from pathlib import Path

# Proje kök dizinini path'e ekleyelim
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import torch
import numpy as np
from src.trainer.ingestion.embedder import LocalLateChunkingEmbedder
from src.config import settings

def cosine_similarity(v1, v2):
    v1 = np.array(v1)
    v2 = np.array(v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0
    return np.dot(v1, v2) / (norm1 * norm2)

def main():
    # Test için ayarları manuel geçersiz kılalım
    os.environ["USE_LOCAL_LATE_CHUNKING"] = "1"
    
    try:
        embedder = LocalLateChunkingEmbedder()
    except Exception as e:
        print(f"Hata: Model yüklenemedi. 'einops' ve 'transformers' kurulu mu? \n{e}")
        return

    # Test metni (Bağlam içeren bir paragraf)
    text = (
        "Türkiye Büyük Millet Meclisi (TBMM), 23 Nisan 1920'de Ankara'da kurulmuştur. "
        "Milli mücadelenin merkezi olan bu kurum, Türkiye Cumhuriyeti'nin temel taşıdır. "
        "Bugün TBMM, yasama yetkisini Türk milleti adına kullanan en üst düzey organdır. "
        "Demokrasinin kalesi olan meclis, halkın iradesini temsil eder."
    )
    
    # Hedef cümlemiz (Bu cümlenin vektörünü karşılaştıracağız)
    target_sentence = "Demokrasinin kalesi olan meclis, halkın iradesini temsil eder."
    
    print("\n" + "="*60)
    print("--- LATE CHUNKING DOĞRULAMA TESTİ (Jina v3) ---")
    print("="*60)
    
    # 1. NAIVE EMBEDDING: Cümleyi tamamen izole (tek başına) embed ediyoruz.
    # Bu yöntemde model cümlenin TBMM ile ilgili olduğunu sadece cümle içindeki kelimelerden anlar.
    print(f"\n[1] Naive (Geleneksel) Vektör oluşturuluyor...")
    naive_vec = embedder.embed_query(target_sentence)
    
    # 2. LATE CHUNKING: Cümleyi tüm paragraf içinde embed ediyoruz.
    # Model önce tüm paragrafı okur, sonra hedef cümlenin olduğu yerdeki token vektörlerini havuzlar.
    print(f"[2] Late Chunking (Bağlamsal) Vektör oluşturuluyor...")
    start_char = text.find(target_sentence)
    end_char = start_char + len(target_sentence)
    spans = [(start_char, end_char)]
    
    late_vecs = embedder.embed_with_late_chunking(text, spans)
    late_vec = late_vecs[0]
    
    # 3. KONTROL: Paragrafın genel anlamı (Global Context)
    full_text_vec = embedder.embed_query(text)
    
    # Skorları hesapla
    # Hedef cümlenin vektörü, paragrafın genel vektörüne ne kadar benziyor?
    sim_naive_to_full = cosine_similarity(naive_vec, full_text_vec)
    sim_late_to_full = cosine_similarity(late_vec, full_text_vec)
    
    print("\n" + "-"*40)
    print(f"Hedef Cümle: '{target_sentence}'")
    print("-"*40)
    print(f"Naive Vektör  <-> Tüm Paragraf Benzerliği: %{sim_naive_to_full*100:.2f}")
    print(f"Late Vektör   <-> Tüm Paragraf Benzerliği: %{sim_late_to_full*100:.2f}")
    print("-"*40)
    
    if sim_late_to_full > sim_naive_to_full:
        gain = ((sim_late_to_full - sim_naive_to_full) / sim_naive_to_full) * 100
        print(f"SONUÇ: BAŞARILI! ✅")
        print(f"Late Chunking, cümlenin bağlamını %{gain:.2f} daha iyi korudu.")
        print("\nAçıklama: Late chunking vektörü, paragrafın bütününe daha yakın.")
        print("Çünkü 'meclis' kelimesinin TBMM olduğunu tüm metinden öğrenmiş oldu.")
    else:
        print("\nSONUÇ: Fark tespit edilemedi veya model bu kısa metinde yeterli ayrımı yapamadı.")

if __name__ == "__main__":
    main()
