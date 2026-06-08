"""
Doküman işleme sürecinde anlamsal parçaları (atomları) akıllıca birleştiren modül.
"""

from typing import List


def greedy_pack(
    atoms: List[str], 
    min_chars: int, 
    max_chars: int, 
    join_str: str = "\n\n"
) -> List[str]:
    """
    Küçük metin parçalarını (atomları) belirli bir eşik değere ulaşana kadar birleştirir.
    
    Bu metod, özellikle meclis tutanaklarındaki kısa araya girmeler veya Docling'den 
    gelen çok kısa paragraflar gibi 'mikro-chunk'ların oluşmasını engellemek için kullanılır.
    Amacı, her bir parçanın (chunk) LLM için yeterli bağlamsal yoğunluğa sahip olmasını sağlamaktır.

    Mantık:
    - Atomları sırayla döner.
    - Mevcut parça (current_chunk) 'min_chars' değerinden küçükse bir sonraki atomu eklemeye devam eder.
    - Eğer 'min_chars' değerine ulaşılmışsa ve bir sonraki atomu eklemek 'max_chars' değerini 
      aşacaksa, mevcut parçayı kapatır ve yeni bir parçaya başlar.

    Argümanlar:
        atoms: Birleştirilecek olan metin parçalarının (paragraf, cümle vb.) listesi.
        min_chars: Bir parçanın sahip olması gereken minimum karakter sayısı.
        max_chars: Bir parçanın (mümkünse) aşmaması gereken maksimum karakter sayısı.
        join_str: Parçalar birleştirilirken araya konulacak karakter (varsayılan: çift satır başı).

    Döndürür:
        Birleştirilmiş metin bloklarının (chunk) listesi.
    """
    packed_chunks = []
    current_chunk = ""
    
    for atom in atoms:
        # Eğer mevcut parça boşsa, direkt atomu ata
        if not current_chunk:
            current_chunk = atom
            continue
            
        # Mevcut parça ile yeni atom birleşirse uzunluk ne olur?
        proposed_len = len(current_chunk) + len(join_str) + len(atom)
        
        # Eğer mevcut parça yeterince büyükse (min_chars aşılmışsa)
        # VE yeni atomu eklemek maksimum sınırı (max_chars) aşıyorsa
        # O zaman mevcut parçayı "paketle" ve yeni bir parçaya geç.
        if len(current_chunk) >= min_chars and proposed_len > max_chars:
            packed_chunks.append(current_chunk)
            current_chunk = atom
        else:
            # Aksi takdirde (yeterli büyüklüğe ulaşılmadıysa veya hala yer varsa) birleştir
            current_chunk += join_str + atom
            
    # Döngü sonunda kalan son parçayı ekle
    if current_chunk:
        packed_chunks.append(current_chunk)
        
    return packed_chunks
