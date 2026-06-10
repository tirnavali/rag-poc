"""Döküman veri alma (ingestion) adaptör sistemi için temel türler.

DocumentInput: TÜM döküman tipleri için birleştirilmiş giriş sözleşmesi.
DocumentAdapter: Tipe özel ayrıştırıcılar (parser) için soyut temel sınıf.
ManifestRecord: Döküman manifest SQLite tablosundaki bir satır.
IngestResult: Bir dökümanın işlenme sonucu.

Key tasarım prensibi: Üstveri (metadata) anahtarları İNGİLİZCE, değerleri TÜRKÇE'dir.
LLM anahtarları asla görmez — parça metnini ve üstveri DEĞERLERİNİ görür.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional
from abc import ABC, abstractmethod


@dataclass
class DocumentInput:
    """TÜM döküman tipleri için birleştirilmiş giriş sözleşmesi.

    Veri alma hattına giren her döküman, PDF, JSON, CSV veya düz metin
    fark etmeksizin bir DocumentInput nesnesi olarak karşılanır.

    Üstveri anahtarları evrensel API uyumluluğu için İNGİLİZCE'dir.
    Üstveri DEĞERLERİ ise TÜRKÇE'dir — bu bir Türkçe dil arşividir.
    """

    # ─── Kimlik Bilgileri (zorunlu, dışarıdan sağlanır) ────
    document_id: str
    """Belirleyici benzersiz tanımlayıcı. Dışarıdan sağlanır, asla otomatik üretilmez.
    
    Format: {tip}-{donem}-{yil}-{birlesim}-{tarih} (tutanaklar için)
    Örnek: "tbmm-20-1-1-19960108"
    """

    document_type: str
    """Adaptör seçici. Hangi DocumentAdapter'ın kullanılacağını belirler.
    
    ADAPTER_REGISTRY içinde tanımlı olmalıdır.
    Örnekler: "tutanak", "press_clip", "pdf_report"
    """

    collection_name: str
    """Hedef CollectionSpec adı. COLLECTIONS kaydında mevcut olmalıdır.
    
    Örnek: "tbmm_minutes_docling_jina_v4", "gazete_arsivi"
    """

    # ─── İçerik Kaynağı ────────────────────────────────
    document_source: Optional[str] = None
    """Kaynak dosyaya giden yol veya URL. None ise içerik satır içidir (örn. gazete kupürü metni)."""

    # ─── Zaman Bilgileri ────────────────────────────────
    document_date: Optional[str] = None
    """Döküman içeriğinin ISO formatında tarihi. Örnek: 1996-01-08"""

    year: Optional[int] = None
    """Tarih aralığı sorguları için document_date'den çıkarılan yıl bilgisi."""

    period: Optional[int] = None
    """TBMM Yasama Dönemi. Örnek: 20, 26, 28."""

    # ─── Yasama Bilgileri (TBMM dışı dökümanlar için boştur) ────
    legislative_year: Optional[int] = None
    """Dönem içindeki TBMM Yasama Yılı. Örnek: 1, 4."""

    session: Optional[int] = None
    """TBMM birleşim (oturum) numarası. Örnek: 1, 70."""

    # ─── Yazarlık ve Yayın Bilgileri ──────────────────────
    author: Optional[str] = None
    """Birincil yazar veya konuşmacı. Türkçe değerler: "Mustafa Kalemli", "Ahmet Hakan"."""

    author_role: Optional[str] = None
    """Yazarın rolü veya ünvanı. Örnekler: "başkan", "bakan", "köşe yazarı"."""

    source_name: Optional[str] = None
    """Yayın kaynağının adı. Örnekler: "Hürriyet", "TBMM Tutanakları", "Milliyet"."""

    title: Optional[str] = None
    """Döküman başlığı veya manşeti. Örnekler: "6. Birleşim", "Seçim öncesi..."."""

    topics: Optional[str] = None
    """Virgülle ayrılmış konu etiketleri. Örnek: "siyaset, seçim, dışişleri"."""

    # ─── Diğer Üstveriler (Sorgulama dışı, tipe özel) ───
    metadata: Optional[dict] = None
    """JSON olarak saklanan, genişletilebilir dökümana özel üstveriler.
    
    BURAYA SADECE asla filtreleme veya sorgulama yapılmayacak alanları koyun.
    Eğer bir alan üzerinden sıkça sorgulama yapılıyorsa, onu yukarıdaki standart alanlardan birine taşıyın.
    
    Örnekler:
      tutanak:  {acilma_saati, katip_uyeler, is_full_minutes}
      press:    {kayit_no, sayfa_no}
      report:   {bolum, dil}
    """

    # ─── İşleme Seçenekleri ────────────────────────────
    ocr: bool = True
    """Dijital üretilmiş (metin katmanlı) PDF'ler için False yapın. OCR'ı atlar, işlemi hızlandırır."""

    min_chunk_chars: Optional[int] = None
    """Koleksiyon düzeyindeki minimum parça boyutunu (karakter) geçersiz kılar."""

    max_chunk_chars: Optional[int] = None
    """Koleksiyon düzeyindeki maksimum parça boyutunu (karakter) geçersiz kılar."""

    # ─── Sistem Bilgileri (Otomatik yönetilir) ─────────
    content_hash: Optional[str] = None
    """Değişiklik tespiti için hızlı içerik hash'i. Belirtilmezse otomatik hesaplanır.
    
    Strateji: Dosya kaynakları için SHA-256(ilk 1MB + dosya boyutu).
    Satır içi içerik için: Metnin kendi SHA-256 hash'i.
    """

    def to_dict(self) -> dict[str, Any]:
        """JSON/JSONB depolama için sözlüğe (dict) seri hale getirir."""
        d = {
            "document_id": self.document_id,
            "document_type": self.document_type,
            "collection_name": self.collection_name,
            "document_source": self.document_source,
            "document_date": self.document_date,
            "year": self.year,
            "period": self.period,
            "legislative_year": self.legislative_year,
            "session": self.session,
            "author": self.author,
            "author_role": self.author_role,
            "source_name": self.source_name,
            "title": self.title,
            "topics": self.topics,
            "metadata": self.metadata,
            "ocr": self.ocr,
            "min_chunk_chars": self.min_chunk_chars,
            "max_chunk_chars": self.max_chunk_chars,
            "content_hash": self.content_hash,
        }
        return {k: v for k, v in d.items() if v is not None}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "DocumentInput":
        """Sözlükten (örn. manifest dosyasından veya veritabanından) nesne oluşturur."""
        return cls(
            document_id=d["document_id"],
            document_type=d["document_type"],
            collection_name=d["collection_name"],
            document_source=d.get("document_source"),
            document_date=d.get("document_date"),
            year=d.get("year"),
            period=d.get("period"),
            legislative_year=d.get("legislative_year"),
            session=d.get("session"),
            author=d.get("author"),
            author_role=d.get("author_role"),
            source_name=d.get("source_name"),
            title=d.get("title"),
            topics=d.get("topics"),
            metadata=d.get("metadata"),
            ocr=d.get("ocr", True),
            min_chunk_chars=d.get("min_chunk_chars"),
            max_chunk_chars=d.get("max_chunk_chars"),
            content_hash=d.get("content_hash"),
        )


@dataclass
class ManifestRecord:
    """Döküman manifest SQLite tablosundaki bir satırı temsil eder."""

    document_id: str
    document_source: Optional[str]
    document_type: str
    collection_name: str
    content_hash: str
    status: str
    chunk_count: int
    document_date: Optional[str]
    year: Optional[int]
    period: Optional[int]
    legislative_year: Optional[int]
    session: Optional[int]
    author: Optional[str]
    author_role: Optional[str]
    source_name: Optional[str]
    title: Optional[str]
    topics: Optional[str]
    ingest_time: str
    last_modified: str
    error_message: Optional[str]
    metadata_json: Optional[str]
    ocr: bool = True
    source_etag: Optional[str] = None
    source_last_modified: Optional[str] = None
    quality_json: Optional[str] = None
    """Tier-1 OCR kalite özeti (JSON). Örnek: {"ocr_flagged": true}"""
    perf_json: Optional[str] = None
    """Aşama zamanlamaları ve chunk istatistikleri (JSON).
    Örnek: {"total_ms": 7400, "parse_ms": 4250, "span_coverage_pct": 100.0}"""
    """Aşama zamanlamaları ve chunk istatistikleri (JSON). Örnek: {"total_ms": 7400, "span_coverage_pct": 100.0}"""


@dataclass
class IngestResult:
    """Bir dökümanın veri alma hattından geçirilme sonucudur."""

    document_id: str
    status: str  # "done", "skipped", "failed"
    chunk_count: int = 0
    reason: Optional[str] = None  # örn. "already_ingested", "content_hash_changed"


class DocumentAdapter(ABC):
    """Döküman tipi adaptörleri için soyut temel sınıf.

    Her adaptör şunları bilir:
    1. Bir DocumentInput kaynağını (tam_metin, üstverili_parçalar) şeklinde ayrıştırmak.
    2. Değişiklik tespiti için hızlı bir içerik hash'i hesaplamak.

    Adaptör, tam olarak doldurulmuş bir DocumentInput nesnesi alır (JSON manifestinden).
    DocumentInput nesnelerini oluşturmak adaptörün değil, dış sistemin sorumluluğundadır.
    """

    @abstractmethod
    def parse(self, doc: DocumentInput) -> tuple[str, list[dict]]:
        """Dökümanı tam metne ve parçalara (chunks) ayrıştırır.

        Args:
            doc: Üstverileri doldurulmuş bir DocumentInput nesnesi.

        Returns:
            full_text: Dökümanın tam metni. Gömme modeli (embedder) destekliyorsa
                       'late chunking' için kullanılır.
            chunks: Sözlük listesi, her biri şu anahtarları içerir:
                - "text": str — parçanın metni
                - "span": tuple[int, int] | None — tam metin içindeki karakter ofsetleri
                - "metadata": dict — parça düzeyindeki üstveriler

        Adaptör, DocumentInput'tan gelen döküman düzeyindeki üstverileri her bir
        parçanın üstverisiyle birleştirmelidir. Parça üstveri anahtarları standart
        İngilizce şemayı (author, source_name, date, vb.) kullanır.
        """
        ...

    @abstractmethod
    def compute_content_hash(self, doc: DocumentInput) -> str:
        """Değişiklik tespiti için hızlı bir içerik hash'i hesaplar.

        Strateji:
        - Dosya kaynakları: SHA-256(ilk 1MB) + dosya boyutu (bayt)
        - Satır içi içerik: Metin dizisinin SHA-256 hash'i

        Hash, çok büyük PDF'lerde tüm dosyayı okumaya gerek kalmadan hızlı
        hesaplanmalı, ancak yanlış negatiflere karşı dirençli olmalıdır.
        """
        ...
