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


# ---------------------------------------------------------------------------
# Chunk overlap (atom-sınırlı ~%10) — token_pack_atoms(overlap_tokens=...)
# ---------------------------------------------------------------------------

def _four_word_atoms(n, pages=None):
    return [
        {"text": f"atom{i} kelime{i}a kelime{i}b kelime{i}c", "pages": pages or [1]}
        for i in range(n)
    ]


def test_token_pack_overlap_zero_unchanged():
    """overlap_tokens=0 ve parametresiz çağrı birebir aynı çıktıyı vermeli."""
    atoms = _four_word_atoms(6)
    full_text = _atoms_to_full_text(atoms)
    base = token_pack_atoms(
        atoms, full_text, count_tokens=_word_tokens, max_tokens=8, min_tokens=5
    )
    explicit = token_pack_atoms(
        atoms, full_text, count_tokens=_word_tokens, max_tokens=8, min_tokens=5,
        overlap_tokens=0,
    )
    assert base == explicit
    # overlap kapalıyken chunk'lar örtüşmez
    for prev, nxt in zip(base, base[1:]):
        assert nxt["span"][0] >= prev["span"][1]


def test_token_pack_overlap_budget_respected():
    """Yeni chunk önceki chunk'ın kuyruk atomuyla başlar; örtüşme bütçeyi aşmaz."""
    atoms = _four_word_atoms(4)
    full_text = _atoms_to_full_text(atoms)
    chunks = token_pack_atoms(
        atoms, full_text, count_tokens=_word_tokens, max_tokens=8, min_tokens=5,
        overlap_tokens=4,
    )
    assert len(chunks) >= 2
    for prev, nxt in zip(chunks, chunks[1:]):
        # Gerçek örtüşme: sonraki chunk öncekinin bitişinden ÖNCE başlar,
        # ama başlangıçlar kesin artan (kopya chunk yok).
        assert nxt["span"][0] < prev["span"][1]
        assert nxt["span"][0] > prev["span"][0]
        # Örtüşen bölgenin token sayısı bütçeyi aşmaz
        overlap_text = full_text[nxt["span"][0]:prev["span"][1]]
        assert _word_tokens(overlap_text) <= 4
        # Örtüşme öncekinin kuyruğu = sonrakinin başı
        assert nxt["text"].startswith(overlap_text)
        assert prev["text"].endswith(overlap_text)
    for c in chunks:
        assert full_text[c["span"][0]:c["span"][1]] == c["text"]


def test_token_pack_overlap_pages_union():
    """Overlap atomunun sayfası yeni chunk'ın pages birleşimine dahil olmalı."""
    atoms = _four_word_atoms(2, pages=[1]) + _four_word_atoms(2, pages=[2])[0:2]
    # atom0,1 → sayfa 1; atom2,3 → sayfa 2 (metinler farklı olsun diye yeniden adlandır)
    atoms[2]["text"] = "beta0 kelimeb0 kelimeb1 kelimeb2"
    atoms[3]["text"] = "beta1 kelimeb3 kelimeb4 kelimeb5"
    full_text = _atoms_to_full_text(atoms)
    chunks = token_pack_atoms(
        atoms, full_text, count_tokens=_word_tokens, max_tokens=8, min_tokens=5,
        overlap_tokens=4,
    )
    # chunk0 = [atom0, atom1] (sayfa 1); chunk1 = [atom1(ovl, s.1), atom2(s.2)] → [1, 2]
    assert len(chunks) >= 2
    assert chunks[1]["pages"] == [1, 2]
    assert chunks[1]["page"] == 1


def test_token_pack_overlap_span_with_repeated_text():
    """Tekrarlayan atom metinleri overlap açıkken de doğru span üretmeli."""
    atoms = [
        {"text": "Tekrar eden paragraf metni burada.", "pages": [1]},
        {"text": "Tekrar eden paragraf metni burada.", "pages": [1]},
        {"text": "Tekrar eden paragraf metni burada.", "pages": [2]},
        {"text": "Farklı kapanış cümlesi geliyor şimdi.", "pages": [2]},
    ]
    full_text = _atoms_to_full_text(atoms)
    chunks = token_pack_atoms(
        atoms, full_text, count_tokens=_word_tokens, max_tokens=10, min_tokens=6,
        overlap_tokens=5,
    )
    starts = [c["span"][0] for c in chunks]
    assert starts == sorted(starts) and len(set(starts)) == len(starts)
    for c in chunks:
        assert full_text[c["span"][0]:c["span"][1]] == c["text"]


def test_token_pack_overlap_suffix_fallback():
    """Kuyruk atomu bütçeye sığmıyorsa son-eki kopyalanır: sonraki chunk atom
    İÇİNDEN başlar, örtüşme bütçeyi aşmaz, atom kendi chunk'ında bütün kalır."""
    atoms = _four_word_atoms(4)  # her atom 4 token, bütçe 2 → atom bütün sığmaz
    full_text = _atoms_to_full_text(atoms)
    chunks = token_pack_atoms(
        atoms, full_text, count_tokens=_word_tokens, max_tokens=8, min_tokens=5,
        overlap_tokens=2,
    )
    assert len(chunks) >= 2
    found_suffix = False
    for prev, nxt in zip(chunks, chunks[1:]):
        assert nxt["span"][0] < prev["span"][1], "Son-ek fallback örtüşme üretmedi"
        overlap_text = full_text[nxt["span"][0]:prev["span"][1]]
        assert _word_tokens(overlap_text) <= 2
        # Atom-içi başlangıç: örtüşme tam atom değil, atomun son-eki
        if 0 < _word_tokens(overlap_text) < 4:
            found_suffix = True
    assert found_suffix
    for c in chunks:
        assert full_text[c["span"][0]:c["span"][1]] == c["text"]
    # Önceki chunk'ın kuyruk atomunun sayfası yeni chunk'a taşınmalı
    assert chunks[1]["pages"] and chunks[0]["pages"][-1] in chunks[1]["pages"]


def test_token_pack_overlap_table_atom_not_split():
    """Tablo etiketli kuyruk atomundan son-ek alınmaz — tablo parçalanmaz."""
    table = " ".join(f"hucre{i}" for i in range(6))
    atoms = [
        {"text": "Giriş cümlesi tam dört kelime.", "pages": [1]},
        {"text": table, "pages": [1], "label": "table"},
        {"text": "Devam eden açıklama metni burada beş.", "pages": [2]},
        {"text": "Son paragraf da beş kelime içerir.", "pages": [2]},
    ]
    full_text = _atoms_to_full_text(atoms)
    chunks = token_pack_atoms(
        atoms, full_text, count_tokens=_word_tokens, max_tokens=11, min_tokens=8,
        overlap_tokens=4,
    )
    # Tablo ile biten chunk'tan sonra gelen chunk tablo içinden BAŞLAYAMAZ
    for prev, nxt in zip(chunks, chunks[1:]):
        if prev["text"].endswith(table):
            assert nxt["span"][0] >= prev["span"][1], "Tablo atomu son-ek için bölündü"
    for c in chunks:
        assert full_text[c["span"][0]:c["span"][1]] == c["text"]


def test_token_pack_overlap_oversized_atom_own_chunk():
    """Max'ı aşan atom kendi chunk'ı olur; önüne tohum girmez, kopya chunk oluşmaz."""
    big = " ".join(f"buyuk{i}" for i in range(12))
    atoms = [
        {"text": "Kısa giriş cümlesi burada dört.", "pages": [1]},
        {"text": big, "pages": [1]},
        {"text": "Kapanış cümlesi de dört kelime.", "pages": [2]},
    ]
    full_text = _atoms_to_full_text(atoms)
    chunks = token_pack_atoms(
        atoms, full_text, count_tokens=_word_tokens, max_tokens=8, min_tokens=3,
        overlap_tokens=4,
    )
    assert any(c["text"] == big for c in chunks), "Oversize atom kendi chunk'ı olmadı"
    # Oversize atomun ÖNÜNE tohum girmez (bütçe = max - n < 0)
    big_chunk = next(c for c in chunks if c["text"] == big)
    prev_chunk = chunks[chunks.index(big_chunk) - 1]
    assert big_chunk["span"][0] >= prev_chunk["span"][1]
    texts = [c["text"] for c in chunks]
    assert len(texts) == len(set(texts)), "Kopya chunk üretildi"


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
