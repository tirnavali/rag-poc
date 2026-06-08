"""Text-splitting helper wrapping LangChain's RecursiveCharacterTextSplitter."""
from __future__ import annotations

from langchain_text_splitters import RecursiveCharacterTextSplitter


from typing import List, Tuple


def build_text_splitter(chunk_size: int, chunk_overlap: int) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


def split_with_offsets(text: str, chunk_size: int, chunk_overlap: int) -> List[Tuple[str, Tuple[int, int]]]:
    """
    Metni parçalara ayırır ve her bir parçanın orijinal metindeki karakter ofsetlerini hesaplar.
    
    Bu metod, standart LangChain splitter'larından farklı olarak sadece metin parçalarını değil, 
    bu parçaların hangi karakter indisleri (başlangıç, bitiş) arasında olduğunu da döndürür.
    
    Neden Gerekli? (Late Chunking):
    Late Chunking tekniğinde, embedding modeli tüm metni tek seferde işler. Belirli bir parçanın
    (chunk) vektörünü oluşturmak için, modelin o parçaya karşılık gelen token vektörlerini
    havuzlaması (mean pooling) gerekir. Bu fonksiyonun sağladığı (start_char, end_char) indisleri,
    modelin tüm metin içinden tam olarak hangi bölgeyi vektörleştireceğini bilmesini sağlar.

    Args:
        text: Parçalanacak tam metin içeriği.
        chunk_size: Hedef parça boyutu (karakter sayısı).
        chunk_overlap: Parçalar arası çakışma miktarı.

    Returns:
        Liste içerisinde her eleman: (parça_metni, (başlangıç_indisi, bitiş_indisi))
    """
    splitter = build_text_splitter(chunk_size, chunk_overlap)
    chunks = splitter.split_text(text)
    
    results = []
    start_search = 0
    for chunk in chunks:
        # RecursiveCharacterTextSplitter might strip whitespace, so we try to find the match
        # in the original text while allowing for slight variations if needed.
        # But usually, it just slices.
        start_idx = text.find(chunk, start_search)
        if start_idx == -1:
            # Fallback to search from beginning if not found from start_search
            start_idx = text.find(chunk)
            
        if start_idx != -1:
            end_idx = start_idx + len(chunk)
            results.append((chunk, (start_idx, end_idx)))
            # Move search pointer forward, but stay behind the end_idx to handle overlaps
            # We move it just past the start_idx to ensure we find the next chunk even if identical
            start_search = start_idx + 1
            
    return results
