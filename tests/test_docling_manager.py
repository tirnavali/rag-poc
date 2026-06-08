import sys
import os
from pathlib import Path

import pytest

# Proje kök dizinini ekleyelim
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.common.parsing.docling_manager import DoclingManager


@pytest.fixture
def pdf_path():
    pytest.skip("Entegrasyon testi — manuel çalıştırın: python tests/test_docling_manager.py <pdf>")

def print_side_by_side(text1: str, text2: str, width: int = 50):
    """İki metni yan yana kolonlar halinde ekrana basar."""
    import textwrap
    lines1 = textwrap.wrap(text1, width=width)
    lines2 = textwrap.wrap(text2, width=width)
    
    max_lines = max(len(lines1), len(lines2))
    
    # Başlıkları bas
    header = f"{'SOL CHUNK':<{width}} | {'SAĞ CHUNK':<{width}}"
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    
    for i in range(max_lines):
        l1 = lines1[i] if i < len(lines1) else ""
        l2 = lines2[i] if i < len(lines2) else ""
        print(f"{l1:<{width}} | {l2:<{width}}")
    print("=" * len(header))

def test_docling_conversion_and_packing(pdf_path: str, verbose: bool = False, compare: bool = False, do_pack: bool = True):
    """
    Belirtilen doküman üzerinde Docling dönüşümü ve paketleme işlemini test eder.
    """
    if not os.path.exists(pdf_path):
        print(f"HATA: Dosya bulunamadı ({pdf_path}).")
        return

    print(f"\n--- DOCLING MANAGER TESTİ BAŞLIYOR ({'PAKETLİ' if do_pack else 'HAM ATOMLAR'}): {os.path.basename(pdf_path)} ---")
    
    manager = DoclingManager()
    
    # Dönüştürme ve Paketleme
    min_chars = 400
    max_chars = 1500
    full_text, chunks = manager.convert_and_pack(
        pdf_path, 
        min_chars=min_chars, 
        max_chars=max_chars,
        do_pack=do_pack
    )
    
    print(f"Toplam karakter sayısı: {len(full_text)}")
    print(f"Oluşturulan parça sayısı: {len(chunks)}")
    
    # 1. Ofset Doğruluğunu Kontrol Et (Sessizce)
    for i, chunk in enumerate(chunks):
        start, end = chunk["span"]
        assert full_text[start:end] == chunk["text"], f"Ofset hatası: Chunk {i} metni eşleşmiyor!"

    # 2. Görüntüleme Mantığı
    if compare:
        # İkişerli kıyaslama modu
        print(f"\n--- KIYASLAMA MODU (2'şerli) - {'PAKETLEME AÇIK' if do_pack else 'PAKETLEME KAPALI'} ---")
        for i in range(0, len(chunks), 2):
            c1_data = f"[{chunks[i]['metadata']['type']}]\n{chunks[i]['text']}"
            if i+1 < len(chunks):
                c2_data = f"[{chunks[i+1]['metadata']['type']}]\n{chunks[i+1]['text']}"
            else:
                c2_data = "(Son parça tek kaldı)"
            
            print(f"\nKıyaslanan: Parça {i+1} ve Parça {i+2 if i+1 < len(chunks) else 'N/A'}")
            print_side_by_side(c1_data, c2_data)
    else:
        # Standart veya Verbose mod
        display_chunks = chunks if verbose else chunks[:3]
        for i, chunk in enumerate(display_chunks):
            chunk_type = chunk['metadata']['type']
            print(f"\nParça {i+1} [{chunk_type}] doğrulandı (Koordinatlar: {chunk['span'][0]}-{chunk['span'][1]})")
            if verbose:
                print("-" * 30)
                print(chunk['text'])
                print("-" * 30)
            else:
                print(f"Önizleme: {chunk['text'][:100]}...")

        if not verbose and len(chunks) > 3:
            print(f"\n(Diğer {len(chunks)-3} parça gizlendi. Tümünü görmek için --verbose kullanın.)")

    print(f"\n✅ TEST BAŞARILI: Docling okuması{' ve paketleme' if do_pack else ''} kusursuz.")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Docling Manager Test Aracı")
    parser.add_argument("path", nargs="?", default="rag-poc-2gun-rehberi.pdf", 
                        help="Test edilecek dosyanın yolu (varsayılan: rag-poc-2gun-rehberi.pdf)")
    parser.add_argument("-v", "--verbose", action="store_true", 
                        help="Tüm chunk içeriklerini ekrana basar")
    parser.add_argument("-c", "--compare", action="store_true", 
                        help="Chunk'ları ikişerli yan yana basar")
    parser.add_argument("--nopack", action="store_true", 
                        help="Paketleme (Packer) adımını atlar, ham Docling atomlarını gösterir")
    
    args = parser.parse_args()
    
    # Eğer path tam yol değilse proje kökünden ara
    target_path = args.path
    if not os.path.isabs(target_path) and not os.path.exists(target_path):
        target_path = os.path.join(PROJECT_ROOT, target_path)
        
    test_docling_conversion_and_packing(
        target_path, 
        verbose=args.verbose, 
        compare=args.compare,
        do_pack=not args.nopack
    )
