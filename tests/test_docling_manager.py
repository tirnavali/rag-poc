import sys
import os
from pathlib import Path

import pytest

# Proje kök dizinini ekleyelim
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.common.parsing.docling_manager import (
    DoclingManager,
    _atom_char_spans,
    token_pack_atoms,
)
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.datamodel.base_models import InputFormat


# ---------------------------------------------------------------------------
# Atom-tabanlı token paketleme (birincil span yolu) — hızlı birim testleri.
# Gerçek PDF/model gerektirmez; full_text = "\n\n".join(atom.text) değişmezini
# ve OCR-bağışık span üretimini doğrular.
# ---------------------------------------------------------------------------

# Kelime-tabanlı sahte token sayacı (HF modeli yüklemeden).
_word_tokens = lambda t: len(t.split())


def _atoms_to_full_text(atoms):
    return "\n\n".join(a["text"] for a in atoms)


def test_atom_spans_are_exact_against_full_text():
    """Her atom full_text'te birebir bulunur; span aritmetiği doğru."""
    atoms = [
        {"text": "Birinci paragraf.", "pages": [1]},
        {"text": "Birinci paragraf.", "pages": [1]},  # tekrar — cursor sırayı korumalı
        {"text": "Üçüncü farklı içerik.", "pages": [2]},
    ]
    full_text = _atoms_to_full_text(atoms)
    spans = _atom_char_spans(atoms, full_text)
    assert all(s is not None for s in spans)
    # Tekrarlayan atomlar farklı (ilerleyen) ofsetlere oturmalı
    assert spans[0][0] < spans[1][0]
    for a, s in zip(atoms, spans):
        assert full_text[s[0]:s[1]] == a["text"]


def test_token_pack_spans_match_text_and_no_charspan_needed():
    """OCR-bozuk senaryo: atomlarda hiç charspan yok; yine de %100 span üretilir
    ve her chunk için full_text[span] == chunk.text."""
    atoms = [
        {"text": "Alfa bir iki uc.", "pages": [1]},
        {"text": "Beta dort bes.", "pages": [1]},
        {"text": "Gama alti yedi sekiz dokuz on.", "pages": [2]},
    ]
    full_text = _atoms_to_full_text(atoms)
    chunks = token_pack_atoms(
        atoms, full_text, count_tokens=_word_tokens, max_tokens=8, min_tokens=2
    )
    assert chunks, "Hiç chunk üretilmedi"
    for c in chunks:
        assert c["span"] is not None
        assert full_text[c["span"][0]:c["span"][1]] == c["text"]


def test_token_pack_keeps_table_atom_whole():
    """max_chunk_tokens'ı aşmayan tablo atomu tek chunk içinde bütün kalır
    (ortadan bölünmez)."""
    table = "| Ad | İl |\n| Levent Gök | Ankara |\n| Celal Adan | İstanbul |"
    atoms = [
        {"text": "Giriş paragrafı kısa.", "pages": [1]},
        {"text": table, "pages": [1]},
        {"text": "Tablodan sonra gelen açıklama metni.", "pages": [1]},
    ]
    full_text = _atoms_to_full_text(atoms)
    chunks = token_pack_atoms(
        atoms, full_text, count_tokens=_word_tokens, max_tokens=12, min_tokens=2
    )
    # Tablo metni tek bir chunk içinde tam geçmeli
    assert any(table in c["text"] for c in chunks), "Tablo atomu ortadan bölündü!"


def test_token_pack_oversize_atom_becomes_own_chunk():
    """Tek başına max_tokens'ı aşan atom kendi chunk'ı olur, bölünmez."""
    big = " ".join(f"kelime{i}" for i in range(40))  # 40 token
    atoms = [
        {"text": "Kısa giriş.", "pages": [1]},
        {"text": big, "pages": [1]},
    ]
    full_text = _atoms_to_full_text(atoms)
    chunks = token_pack_atoms(
        atoms, full_text, count_tokens=_word_tokens, max_tokens=10, min_tokens=2
    )
    assert any(c["text"] == big for c in chunks), "Büyük atom bütün bir chunk olmadı"
    for c in chunks:
        assert full_text[c["span"][0]:c["span"][1]] == c["text"]


def test_docling_manager_uses_pypdfium_backend():
    """DoclingManager'ın PDF formatı için PyPdfium backend'i kullandığını doğrular.

    Varsayılan Docling backend'i (DoclingParse) bazı gömülü fontları yanlış çözüp
    glyph-substitution bozulması üretir ("Adalet" → "AGaOeW"); PyPdfium bunu giderir.
    """
    manager = DoclingManager(do_ocr=False)
    format_options = manager._converter.converter.format_to_options
    assert InputFormat.PDF in format_options
    pdf_option = format_options[InputFormat.PDF]
    assert pdf_option.backend == PyPdfiumDocumentBackend


@pytest.mark.slow
def test_docling_turkish_encoding_and_layout():
    """test_docling_turkish_encoding.pdf dosyasının doğru şekilde okunduğunu,
    Türkçe karakterlerin bozulmadığını ve layoutun düzgün çıktığını doğrular.

    PDF'in native metin katmanı bozuk font encoding içerdiğinden OCR gereklidir.
    """
    pdf_path = str(PROJECT_ROOT / "tests" / "fixtures" / "test_docling_turkish_encoding.pdf")
    assert os.path.exists(pdf_path), f"Test fixture bulunamadı: {pdf_path}"

    manager = DoclingManager(do_ocr=True)
    full_text, chunks = manager.convert_and_pack(pdf_path, do_pack=False)

    # 1. Türkçe karakter ve bozuk kelime kontrolleri
    assert "Adalet ve Kalkınma" in full_text
    assert "Bülent Turan" in full_text
    assert "İçişleri Bakanlığı" in full_text
    
    # Bozuk kelimelerin çıkmadığından emin ol
    assert "AGaOeW" not in full_text
    assert "KaONıQma" not in full_text

    # 2. Layout ve Offset (Span) Doğruluğu Kontrolü
    assert len(chunks) > 0, "Hiç chunk üretilmedi!"
    
    for i, chunk in enumerate(chunks):
        assert "span" in chunk and chunk["span"] is not None, f"Chunk {i}'de span bilgisi eksik!"
        start, end = chunk["span"]
        assert full_text[start:end] == chunk["text"], f"Span eşleşme hatası: Chunk {i}"
        
        # Metadata kontrolleri
        meta = chunk.get("metadata", {})
        assert "type" in meta
        assert meta["source"] == "test_docling_turkish_encoding.pdf"


def test_docling_table_extraction():
    """docling_table_extraction_test.pdf dosyasındaki tabloların doğru şekilde parse edildiğini,
    tablo yapısının korunduğunu ve Türkçe karakterlerin bozulmadığını doğrular.
    """
    pdf_path = str(PROJECT_ROOT / "tests" / "fixtures" / "docling_table_extraction_test.pdf")
    assert os.path.exists(pdf_path), f"Test fixture bulunamadı: {pdf_path}"

    manager = DoclingManager(do_ocr=False)
    full_text, chunks = manager.convert_and_pack(pdf_path, do_pack=False)

    # 1. Tablo varlığı kontrolü
    table_chunks = [c for c in chunks if c.get("metadata", {}).get("type") == "table"]
    assert len(table_chunks) >= 1, "Hiç tablo chunk'ı bulunamadı!"

    table_text = table_chunks[0]["text"]
    
    # 2. Tablo kolon ve satır yapısı doğrulaması
    assert "|" in table_text, "Tablo markdown formatında değil!"
    assert "Adı-Soyadı" in table_text
    assert "Seçim Çevresi" in table_text
    assert "Siyasi Parti Grubu" in table_text
    
    # 3. Tablo içerisindeki isimlerin ve Türkçe karakterlerin doğrulanması
    assert "Levent Gök" in table_text
    assert "Celal Adan" in table_text
    assert "Mithat Sancar" in table_text
    assert "Mustafa Şentop" in table_text
    assert "Kâtip Üyelikler" in table_text
    assert "Burcu Köksal" in table_text
    assert "İsmail Ok" in table_text
    assert "Bayram Özçelik" in table_text
    assert "Rümeysa Kadak" in table_text
    assert "Fatma Kaplan Hürriyet" in table_text
    assert "Şeyhmus Dinçel" in table_text

    # 4. Span / Offset eşleşme doğrulaması
    for i, chunk in enumerate(chunks):
        assert "span" in chunk and chunk["span"] is not None
        start, end = chunk["span"]
        assert full_text[start:end] == chunk["text"], f"Span eşleşme hatası: Chunk {i}"
        
        meta = chunk.get("metadata", {})
        assert meta["source"] == "docling_table_extraction_test.pdf"


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
    
    # Dönüştürme ve Paketleme (token-tabanlı; tokenizer'sız manager → fallback)
    full_text, chunks = manager.convert_and_pack(
        pdf_path,
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
