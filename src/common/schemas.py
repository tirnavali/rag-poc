from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field

class RAGBaseModel(BaseModel):
    """Base Pydantic model for the RAG project with strict validation."""
    model_config = ConfigDict(
        extra="forbid",  # Tanımlanmamış ekstra alanlara izin verme
        validate_assignment=True,  # Atama sırasında doğrulamayı çalıştır
        populate_by_name=True,  # Field name/alias ile veri kabul et
    )

class Message(RAGBaseModel):
    """Standardized message schema for LLM conversation arrays."""
    role: Literal["system", "user", "assistant", "tool"]
    content: str
    name: Optional[str] = None
    tool_call_id: Optional[str] = None

class ToolCallFunction(RAGBaseModel):
    name: str
    arguments: str

class ToolCall(RAGBaseModel):
    """Standard representation of an LLM tool call."""
    id: str
    type: Literal["function"] = "function"
    function: ToolCallFunction

class SearchResult(RAGBaseModel):
    """Typed schema representing vector database hits."""
    document: str = Field(..., description="The chunk text retrieved from the DB")
    metadata: Dict[str, Any] = Field(default_factory=dict)
    distance: float = Field(..., description="The vector distance/similarity score")
    
class RoutingDecision(RAGBaseModel):
    """Example schema: Structured output to decide which collection to search."""
    intents: List[str] = Field(..., description="List of source types to search, e.g. ['minutes', 'gazete']")
    reasoning: str = Field(..., description="Why these sources were chosen")

class FilterCriteria(RAGBaseModel):
    """Doğal dil sorgularından çıkarılan metadata filtre kriterlerini doğrulayan validasyon kontrol sınıfı.

    Bu Pydantic şeması, sorgudan çıkarılan yıl, yazar, rol, kaynak, meclis dönemi ve birleşimi gibi
    tüm metadata alanlarının veri tiplerini ve değer kısıtlamalarını kontrol eder, geçersiz tipleri engeller.
    """
    year: Optional[int] = Field(None, description="Filtrelenecek tam yıl bilgisi, kesin eşleşme için (örn. 1996)")
    year_lte: Optional[int] = Field(None, description="Bu yıl DAHİL ÖNCE filtresi ($lte) — 'X yılından önce/itibaren', 'X yılına kadar' ifadeleri (örn. '2000 yılından önce' → year_lte=2000)")
    year_gte: Optional[int] = Field(None, description="Bu yıl DAHİL SONRA filtresi ($gte) — 'X yılından sonra/itibaren', 'X yılı ve sonrası' ifadeleri (örn. '1990 yılından sonra' → year_gte=1990)")
    author: Optional[str] = Field(None, description="Belirli bir yazar veya konuşmacının adı (örn. 'Deniz Baykal', 'Ahmet Kabil')")
    author_role: Optional[str] = Field(None, description="Yazarın rolü veya ünvanı (örn. 'bakan', 'başkan')")
    source_name: Optional[str] = Field(None, description="Yayın kaynağı adı (örn. 'Hürriyet', 'TBMM Tutanakları', 'Sabah')")
    period: Optional[int] = Field(None, description="TBMM yasama dönemi (örn. 20)")
    session: Optional[int] = Field(None, description="TBMM birleşim numarası (örn. 7)")
    document_type: Optional[Literal["tutanak", "press_clip", "pdf_report", "kanun_teklifi"]] = Field(
        None, description="Belgenin türü"
    )

class ExtractedFilterResponse(RAGBaseModel):
    """LLM filtre çıkarma ve sorgu sadeleştirme yanıtının tamamını doğrulayan ana validasyon kontrol sınıfı.

    Bu şema, LLM'den gelen yapısal JSON yanıtının hem temizlenmiş arama sorgusunu (refined_query)
    hem de çıkarılan filtre kriterlerini (filters) içerdiğini garanti altına alır, veri bütünlüğünü denetler.
    """
    refined_query: str = Field(..., description="Filtre kelimelerinden arındırılmış temiz arama sorgusu (örn. 'Deniz Baykal 1996 Kardak' -> 'Kardak')")
    filters: FilterCriteria = Field(default_factory=FilterCriteria, description="Çıkarılan filtre kriterleri")
    removed_words: list[str] = Field(default_factory=list, description="Filtreye dönüştürülen veya temizlenen kelimeler (örn. ['1996', 'Deniz Baykal'])")

