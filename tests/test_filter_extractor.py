"""Unit tests for the FilterExtractor engine and its integrations.

Covers rule-based pre-processor bypass, schema parsing, ChromaDB conversion,
and integrations with VectorRetriever and RAGService.
"""
from unittest.mock import MagicMock, patch, ANY
import pytest

from src.common.schemas import ExtractedFilterResponse, FilterCriteria
from src.generator.filter_extractor import FilterExtractor
from src.retriever.vector_retriever import VectorRetriever
from src.generator.service import RAGService
from src.config.collections import CollectionSpec
from src.config.document_types import DocumentType


class TestFilterExtractor:
    """Test suite for FilterExtractor engine."""

    def test_default_model_is_filter_model(self):
        """FilterExtractor should default to the dedicated FILTER_LLM_MODEL, not LLM_MODEL."""
        from src.config import settings
        fe = FilterExtractor()
        assert fe.model == settings.FILTER_LLM_MODEL

    def test_has_filter_hints_years(self):
        """Should detect 4-digit years in query."""
        fe = FilterExtractor()
        assert fe.has_filter_hints("1996 yılında ne oldu?") is True
        assert fe.has_filter_hints("TBMM 2003 tutanakları") is True
        assert fe.has_filter_hints("Kardak krizi tarihi") is False

    def test_has_filter_hints_keywords(self):
        """Should detect TBMM and newspaper keywords in query."""
        fe = FilterExtractor()
        assert fe.has_filter_hints("meclis tutanakları nerede?") is True
        assert fe.has_filter_hints("Hürriyet gazetesi haberleri") is True
        assert fe.has_filter_hints("önerge sahibi kim?") is True
        assert fe.has_filter_hints("kardak kayalıkları") is False

    def test_has_filter_hints_proper_nouns(self):
        """Should detect capitalized proper nouns in the middle of query."""
        fe = FilterExtractor()
        assert fe.has_filter_hints("Ahmet Kabil konuşmaları") is True
        assert fe.has_filter_hints("Sorgumuzda Deniz Baykal geçiyor") is True
        # First word capitalized should be ignored if it's the only capitalized word
        assert fe.has_filter_hints("Kardak krizi nedir") is False

    def test_to_chroma_filter_single(self):
        """Should convert single filter to $eq condition."""
        filters = FilterCriteria(year=1996)
        chroma_filter = FilterExtractor.to_chroma_filter(filters)
        assert chroma_filter == {"year": {"$eq": 1996}}

    def test_to_chroma_filter_multiple(self):
        """Should convert multiple filters to $and structure with Turkish-aware author casing."""
        filters = FilterCriteria(year=1996, author="Deniz Baykal", document_type="tutanak")
        chroma_filter = FilterExtractor.to_chroma_filter(filters)
        assert chroma_filter == {
            "$and": [
                {"year": {"$eq": 1996}},
                {
                    "$or": [
                        {"author": {"$eq": "Deniz Baykal"}},
                        {"author": {"$eq": "DENİZ BAYKAL"}},
                    ]
                },
                {"document_type": {"$eq": "tutanak"}}
            ]
        }

    def test_to_chroma_filter_empty(self):
        """Should return None if all filters are empty."""
        filters = FilterCriteria()
        chroma_filter = FilterExtractor.to_chroma_filter(filters)
        assert chroma_filter is None

    def test_to_chroma_filter_year_lte(self):
        """year_lte should produce a $lte condition (inclusive upper bound)."""
        filters = FilterCriteria(year_lte=2000)
        chroma_filter = FilterExtractor.to_chroma_filter(filters)
        assert chroma_filter == {"year": {"$lte": 2000}}

    def test_to_chroma_filter_year_gte(self):
        """year_gte should produce a $gte condition (inclusive lower bound)."""
        filters = FilterCriteria(year_gte=1990)
        chroma_filter = FilterExtractor.to_chroma_filter(filters)
        assert chroma_filter == {"year": {"$gte": 1990}}

    def test_to_chroma_filter_year_range(self):
        """year_lte + year_gte together should produce $and with $lte and $gte."""
        filters = FilterCriteria(year_lte=2000, year_gte=1990)
        chroma_filter = FilterExtractor.to_chroma_filter(filters)
        assert chroma_filter == {
            "$and": [
                {"year": {"$lte": 2000}},
                {"year": {"$gte": 1990}},
            ]
        }


    def test_extract_with_bypass(self):
        """Queries without filter hints should bypass LLM call and return empty filters."""
        fe = FilterExtractor()
        with patch.object(fe.client, "chat") as mock_chat:
            res = fe.extract("kardak krizi")
            mock_chat.assert_not_called()
            assert res.refined_query == "kardak krizi"
            assert res.filters.year is None
            assert res.filters.author is None

    def test_extract_with_llm_success(self):
        """Queries with filter hints should call LLM and parse structured output."""
        fe = FilterExtractor()
        mock_response = MagicMock()
        mock_response.message.content = (
            '{"refined_query": "Kardak adaları", "filters": '
            '{"year": 1996, "author": "Deniz Baykal", "document_type": "tutanak"}}'
        )
        with patch.object(fe.client, "chat", return_value=mock_response) as mock_chat:
            res = fe.extract("Deniz Baykal 1996 Kardak adaları konuşması")
            mock_chat.assert_called_once()
            assert res.refined_query == "Kardak adaları"
            assert res.filters.year == 1996
            assert res.filters.author == "Deniz Baykal"
            assert res.filters.document_type == "tutanak"

    def test_extract_with_llm_failure_fallback(self):
        """Should gracefully fall back to original query and empty filters on LLM or parse failure."""
        fe = FilterExtractor()
        with patch.object(fe.client, "chat", side_effect=Exception("Ollama offline")):
            res = fe.extract("Deniz Baykal 1996 Kardak")
            assert res.refined_query == "Deniz Baykal 1996 Kardak"
            assert res.filters.year is None
            assert res.filters.author is None


class TestIntegration:
    """Test suite for FilterExtractor integrations with retriever and service."""

    def test_vector_retriever_uses_explicit_where_filter(self):
        """Passing explicit where_filter to VectorRetriever.retrieve should bypass year extraction."""
        spec = MagicMock(spec=CollectionSpec)
        spec.supports_late_chunking = False
        spec.doc_type = DocumentType.GAZETE

        with patch("src.retriever.vector_retriever.VectorSearch") as mock_vs_class:
            mock_vs = MagicMock()
            mock_vs.search.return_value = []
            mock_vs_class.return_value = mock_vs

            retriever = VectorRetriever(spec)
            custom_filter = {"year": {"$eq": 2005}}
            retriever.retrieve("1996 yılında ne oldu?", where_filter=custom_filter)

            # Verify that our explicit filter is forwarded, completely ignoring "1996" date extraction
            call_kwargs = mock_vs.search.call_args[1]
            assert call_kwargs["where_filter"] == custom_filter

    def test_rag_service_auto_extracts_filters(self):
        """RAGService.retrieve should dynamically extract filters and cascade on zero hits."""
        with patch("src.generator.service.FilterExtractor") as mock_fe_class, \
             patch("src.generator.service.VectorRetriever") as mock_vr_class:

            mock_fe = MagicMock()
            mock_fe.extract.return_value = ExtractedFilterResponse(
                refined_query="Kardak",
                filters=FilterCriteria(year=1996)
            )
            mock_fe.to_chroma_filter.return_value = {"year": {"$eq": 1996}}
            mock_fe.fallback_chain.return_value = [
                (None, {"year": {"$eq": 1996}}),
                ("semantic_only", None),
            ]
            mock_fe_class.return_value = mock_fe

            mock_vr = MagicMock()
            # Return empty on first call (full filter), non-empty on second (semantic)
            mock_vr.retrieve.side_effect = [
                {
                    "documents": [[]],
                    "metadatas": [[]],
                    "distances": [[]],
                    "is_minutes": False,
                    "parsed_dates": {},
                    "expanded_query": None,
                    "fallback_level": None,
                },
                {
                    "documents": [["content1", "content2"]],
                    "metadatas": [{"id": 1}, {"id": 2}],
                    "distances": [[0.1, 0.2]],
                    "is_minutes": False,
                    "parsed_dates": {},
                    "expanded_query": None,
                    "fallback_level": None,
                },
            ]
            mock_vr_class.return_value = mock_vr

            service = RAGService()
            result = service.retrieve("1996 yılındaki Kardak")

            mock_fe.extract.assert_called_once_with("1996 yılındaki Kardak")
            mock_fe.fallback_chain.assert_called_once()
            # Should call retriever twice: once with full filter (empty), once with None (non-empty)
            assert mock_vr.retrieve.call_count == 2
            assert result["fallback_level"] == "semantic_only"


class TestFilterHintsKnownLimitations:
    """Belgelenmiş bilinen sınırlamalar — has_filter_hints bilerek maliyet-gate olarak
    tutuluyor (precision kritik). Çıplak soyad / iyelik eki vakaları tespit edilemiyor.

    Bu kabul edilebilir çünkü mimaride Planner sorguyu önce refine eder (çıplak soyadı
    tam isme çevirir: 'Baykal' → 'Deniz Baykal'), dolayısıyla FilterExtractor'a hep
    temiz/tam-isimli sorgu ulaşır ve has_filter_hints zaten geçer. Bkz. plan:
    Planner ↔ FilterExtractor entegrasyonu.
    """

    def test_has_filter_hints_surname_first_not_detected(self):
        """KNOWN LIMITATION: tek çıplak soyad cümle başındayken (words[0]) tespit edilmez.

        Mantık sadece words[1:]'i kontrol eder; words[0] her cümle büyük harfle
        başladığı için kasıtlı olarak atlanır (yoksa 'Kardak nedir' false-positive olur).
        Planner sorguyu tam isme refine ettiği için pratikte sorun değil.
        """
        fe = FilterExtractor()
        assert fe.has_filter_hints("Baykal konuşmaları") is False
        assert fe.has_filter_hints("Ecevit açıklaması") is False

    def test_has_filter_hints_genitive_suffix_not_detected(self):
        """KNOWN LIMITATION: cümle başındaki iyelik ekli özel isim tek başına tespit edilmez.

        words[0]="Erdoğan'ın" → atlanır; words[1]="konuşmaları" → küçük harf → False.
        Ancak başka bir ipucu (yıl, keyword, ikinci özel isim) varsa kurtarır.
        """
        fe = FilterExtractor()
        assert fe.has_filter_hints("Erdoğan'ın konuşmaları") is False
        # Yıl ipucu varsa kurtarır (soyadı değil, 1996 tetikler):
        assert fe.has_filter_hints("Baykal'ın 1996 konuşması") is True
        # 'meclis' keyword'ü varsa kurtarır:
        assert fe.has_filter_hints("Çiller'in meclis tutanakları") is True


class TestFilterTranslatorEdgeCases:
    """Eksik kapsam — ChromaFilterTranslator kenar durumları."""

    def test_to_chroma_filter_author_only(self):
        """Yazar-tek filtre: top-level $or üretilmeli (tek koşul olduğunda $and yok)."""
        filters = FilterCriteria(author="Deniz Baykal")
        chroma_filter = FilterExtractor.to_chroma_filter(filters)
        assert chroma_filter == {
            "$or": [
                {"author": {"$eq": "Deniz Baykal"}},
                {"author": {"$eq": "DENİZ BAYKAL"}},
            ]
        }

    def test_to_chroma_filter_source_name_not_normalized(self):
        """source_name büyük/küçük harf normalizasyonu yapılmıyor — bilinen sınırlama.

        LLM genellikle küçük harf döndürür ("hürriyet"), metadata "Hürriyet" saklar.
        Bu exact-match sessizce başarısız olur. author için yapılan $or trick'i burada yok.
        """
        filters = FilterCriteria(source_name="hürriyet")
        chroma_filter = FilterExtractor.to_chroma_filter(filters)
        assert chroma_filter == {"source_name": {"$eq": "hürriyet"}}

    def test_to_chroma_filter_year_and_year_lte(self):
        """year + year_lte aynı anda: $and ile iki koşul üretilmeli."""
        filters = FilterCriteria(year=1996, year_lte=2000)
        chroma_filter = FilterExtractor.to_chroma_filter(filters)
        assert chroma_filter == {
            "$and": [
                {"year": {"$eq": 1996}},
                {"year": {"$lte": 2000}},
            ]
        }

    def test_to_chroma_filter_period_only(self):
        """period tek başına: single-condition dönem filtresi."""
        filters = FilterCriteria(period=20)
        chroma_filter = FilterExtractor.to_chroma_filter(filters)
        assert chroma_filter == {"period": {"$eq": 20}}

    def test_to_chroma_filter_session_only(self):
        """session tek başına: single-condition birleşim filtresi."""
        filters = FilterCriteria(session=7)
        chroma_filter = FilterExtractor.to_chroma_filter(filters)
        assert chroma_filter == {"session": {"$eq": 7}}


class TestFallbackChain:
    """Test suite for filter relaxation cascade logic."""

    def test_fallback_chain_full_filter(self):
        """Should return full filter as first candidate."""
        fe = FilterExtractor()
        criteria = FilterCriteria(year=1996, author="Deniz Baykal")
        chain = fe.fallback_chain(criteria)
        assert chain[0][0] is None  # level is None for full filter
        assert chain[0][1] is not None  # has where_filter

    def test_fallback_chain_with_author(self):
        """Should skip author_dropped tier if author wasn't in original filter."""
        fe = FilterExtractor()
        criteria = FilterCriteria(year=1996, author=None)
        chain = fe.fallback_chain(criteria)
        # full + semantic_only (no author to drop)
        assert len(chain) == 2
        assert chain[0][0] is None
        assert chain[1][0] == "semantic_only"

    def test_fallback_chain_with_author_and_role(self):
        """Should include author_dropped tier when author is present."""
        fe = FilterExtractor()
        criteria = FilterCriteria(
            year=1996, author="Deniz Baykal", author_role="bakan"
        )
        chain = fe.fallback_chain(criteria)
        # full + author_dropped + semantic_only
        assert len(chain) == 3
        assert chain[0][0] is None
        assert chain[1][0] == "author_dropped"
        assert chain[2][0] == "semantic_only"

    def test_fallback_chain_author_role_only(self):
        """author_role var, author yok: relaxed author_role'ü de düşürür → 3 kademe."""
        fe = FilterExtractor()
        criteria = FilterCriteria(year=1996, author_role="bakan")
        chain = fe.fallback_chain(criteria)
        assert len(chain) == 3
        assert chain[0][0] is None
        assert chain[0][1] == {"$and": [{"year": {"$eq": 1996}}, {"author_role": {"$eq": "bakan"}}]}
        assert chain[1][0] == "author_dropped"
        assert chain[1][1] == {"year": {"$eq": 1996}}
        assert chain[2][0] == "semantic_only"

    def test_fallback_chain_author_only_no_intermediate(self):
        """Sadece author varsa: relaxed=None → author_dropped tier eklenmiyor."""
        fe = FilterExtractor()
        criteria = FilterCriteria(author="Deniz Baykal")
        chain = fe.fallback_chain(criteria)
        # relaxed author'ı düşürünce hiç koşul kalmıyor (None) → intermediate yok
        assert len(chain) == 2
        assert chain[0][0] is None
        assert chain[0][1] is not None
        assert chain[1][0] == "semantic_only"

    def test_fallback_chain_semantic_only_is_always_last(self):
        """semantic_only should always be the last candidate with where_filter=None."""
        fe = FilterExtractor()
        criteria = FilterCriteria(
            year=1996,
            author="Deniz Baykal",
            source_name="Hürriyet",
            period=20,
        )
        chain = fe.fallback_chain(criteria)
        assert chain[-1][0] == "semantic_only"
        assert chain[-1][1] is None
