import sys
import os
from pathlib import Path

# Proje kök dizinini ekleyelim
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.common.parsing.packer import greedy_pack

def test_greedy_packer_merges_small_atoms():
    # Test verisi: Meclis tutanaklarını simüle edelim
    atoms = [
        "BAŞKAN - Sayın milletvekilleri, bugün gündemimizde anayasa değişikliği teklifi bulunmaktadır.", # Uzun
        "AHMET BEY - Efendim, bir saniye...", # Çok kısa (atom)
        "BAŞKAN - Buyurun Ahmet Bey.", # Çok kısa (atom)
        "AHMET BEY - Bu teklifin üçüncü maddesi üzerinde bazı tereddütlerimiz var, detaylandırmanızı rica ediyorum.", # Orta
        "MEHMET BEY - Ben de katılıyorum!" # Çok kısa
    ]
    
    # Ayarlar: Minimum 200 karakter olsun istiyoruz
    min_chars = 200
    max_chars = 1000
    
    packed = greedy_pack(atoms, min_chars=min_chars, max_chars=max_chars)
    
    print("\n--- PACKER TEST SONUÇLARI ---")
    print(f"Girdi atom sayısı: {len(atoms)}")
    print(f"Çıktı chunk sayısı: {len(packed)}")
    
    for i, chunk in enumerate(packed):
        print(f"\nCHUNK {i+1} (Uzunluk: {len(chunk)}):")
        print("-" * 20)
        print(chunk)
        print("-" * 20)
        
    # Doğrulamalar
    # Toplam atom sayısı 5 idi, ama min_chars 200 olduğu için hepsi muhtemelen birleşecek
    # Çünkü toplam uzunluk yaklaşık 300-350 karakter.
    assert len(packed) < len(atoms), "Küçük parçalar birleştirilmeliydi."
    
    # Eğer toplam metin min_chars'tan büyükse ama max_chars'ı aşmıyorsa tek chunk olmalı
    total_len = sum(len(a) for a in atoms) + (len("\n\n") * (len(atoms)-1))
    if total_len <= max_chars:
        assert len(packed) == 1, f"Tüm metin {total_len} karakter, tek parça olmalıydı."

    print("\n✅ TEST BAŞARILI: Küçük parçalar başarıyla paketlendi.")

if __name__ == "__main__":
    test_greedy_packer_merges_small_atoms()
