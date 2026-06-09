"""Unit tests for PlanningAgent orchestration logic (offline — no LLM/Chroma).

Covers the pure-Python decision logic and locks in two fixes:
  - _execute_single must call SearchTool.search(collection_key=...) (param name).
  - _execute_single must translate FilterCriteria into a Chroma where dict
    ($and/$eq), not pass a raw model_dump() dict.
"""
import pytest

from unittest.mock import MagicMock

from src.agent.planner import PlanningAgent
from src.agent.schemas import CollectionSearchPlan, SearchPlan, SearchQueryDraft
from src.agent.tracer import PipelineTracer
from src.common.llm_client_pool import LLMClientPool
from src.common.schemas import ExtractedFilterResponse, FilterCriteria
from src.config.pipeline_loader import load_pipeline_config


def _agent(filter_extractor=None) -> PlanningAgent:
    cfg = load_pipeline_config()
    pool = LLMClientPool.from_config(cfg)
    return PlanningAgent(cfg, pool, filter_extractor)


def _two_collection_plan() -> SearchPlan:
    """Plan with 2 collections × 2 drafts each, all filters initially None."""
    return SearchPlan(
        intent="factual",
        resources=[
            CollectionSearchPlan(
                collection="tbmm_minutes",
                query_drafts=[SearchQueryDraft(text="q1a"), SearchQueryDraft(text="q1b")],
            ),
            CollectionSearchPlan(
                collection="gazete_arsivi",
                query_drafts=[SearchQueryDraft(text="q2a"), SearchQueryDraft(text="q2b")],
            ),
        ],
        reasoning="r",
    )


def test_needs_reretrieval_true_when_below_threshold():
    agent = _agent()
    assert agent._needs_reretrieval([{"documents": ["only one"]}]) is True


def test_needs_reretrieval_false_when_enough():
    agent = _agent()
    assert agent._needs_reretrieval([{"documents": ["a", "b", "c", "d"]}]) is False


def test_merge_results_deduplicates_by_chunk_id():
    agent = _agent()
    original = [{
        "documents": ["a"], "metadatas": [{"chunk_id": "1"}], "distances": [0.1],
    }]
    new = [{
        "documents": ["a-dup", "b"],
        "metadatas": [{"chunk_id": "1"}, {"chunk_id": "2"}],
        "distances": [0.2, 0.3],
    }]
    merged = agent._merge_results(original, new)
    all_ids = [m["chunk_id"] for r in merged for m in r.get("metadatas", [])]
    assert all_ids.count("1") == 1
    assert "2" in all_ids


def test_fallback_plan_fills_query_template():
    agent = _agent()
    plan = agent._fallback_plan("Kardak krizi")
    assert plan.intent == "unknown"
    assert plan.resources
    drafts = plan.resources[0].query_drafts
    assert any("Kardak" in d.text for d in drafts)


def test_new_planner_class_returns_search_plan(monkeypatch):
    """The Planner class wraps PlanningAgent._generate_plan into a public method."""
    from src.agent.planner import Planner

    cfg = load_pipeline_config()
    pool = LLMClientPool.from_config(cfg)
    planner = Planner(cfg, pool)

    def _no_plan(*a, **kw):
        return None
    monkeypatch.setattr(planner._inner, "_generate_plan", _no_plan)

    plan = planner.plan("test query")
    assert plan is not None
    assert plan.intent == "unknown"
    assert plan.resources


def test_execute_single_passes_collection_key_and_translates_filter(monkeypatch):
    agent = _agent()
    captured = {}

    def fake_search(*, collection_key, query_text, filters, top_k):
        captured["collection_key"] = collection_key
        captured["filters"] = filters
        captured["top_k"] = top_k
        return {"documents": [], "metadatas": [], "distances": []}

    monkeypatch.setattr(agent._search_tool, "search", fake_search)

    draft = SearchQueryDraft(
        text="ege adalari",
        filters=FilterCriteria(year=1996, author="Deniz Baykal"),
        top_k=7,
    )
    agent._execute_single("tbmm_minutes", draft, PipelineTracer())

    assert captured["collection_key"] == "tbmm_minutes"
    assert captured["top_k"] == 7
    # Translated to Chroma where dict (multi-field → $and), not a raw dict.
    assert "$and" in captured["filters"]


def test_execute_single_no_filter_passes_none(monkeypatch):
    agent = _agent()
    captured = {}

    def fake_search(*, collection_key, query_text, filters, top_k):
        captured["filters"] = filters
        return {"documents": [], "metadatas": [], "distances": []}

    monkeypatch.setattr(agent._search_tool, "search", fake_search)
    draft = SearchQueryDraft(text="serbest sorgu")
    agent._execute_single("tbmm_minutes", draft, PipelineTracer())
    assert captured["filters"] is None


def test_execute_plan_parallel_runs_all_drafts(monkeypatch):
    agent = _agent()
    seen = []

    def fake_search(*, collection_key, query_text, filters, top_k):
        seen.append(query_text)
        return {"documents": ["d"], "metadatas": [{"chunk_id": query_text}], "distances": [0.1]}

    monkeypatch.setattr(agent._search_tool, "search", fake_search)
    plan = SearchPlan(
        intent="factual",
        resources=[CollectionSearchPlan(
            collection="tbmm_minutes",
            mode="parallel",
            query_drafts=[SearchQueryDraft(text="q1"), SearchQueryDraft(text="q2")],
        )],
        reasoning="r",
    )
    results = agent._execute_plan(plan, PipelineTracer())
    assert len(results) == 2
    assert set(seen) == {"q1", "q2"}


def test_execute_plan_tags_re_retrieval_phase(monkeypatch):
    agent = _agent()
    monkeypatch.setattr(
        agent._search_tool, "search",
        lambda **k: {"documents": [], "metadatas": [], "distances": []},
    )
    tracer = PipelineTracer()
    plan = SearchPlan(
        intent="factual",
        resources=[CollectionSearchPlan(
            collection="tbmm_minutes",
            query_drafts=[SearchQueryDraft(text="q")],
        )],
        reasoning="r",
    )
    agent._execute_plan(plan, tracer, phase="re_retrieval")
    assert tracer.events
    assert all(e.phase == "re_retrieval" for e in tracer.events)


# --- Session koleksiyon kapsamı (legacy yol) ---

def _plan(*collections: str) -> SearchPlan:
    return SearchPlan(
        intent="factual",
        resources=[
            CollectionSearchPlan(
                collection=c,
                query_drafts=[SearchQueryDraft(text="q")],
            )
            for c in collections
        ],
        reasoning="r",
    )


def test_enforce_session_collections_drops_out_of_scope():
    """Planner kapsam dışı koleksiyon seçerse düşürülür, seçili olan kalır."""
    agent = _agent()
    plan = _plan("gazete_arsivi", "tbmm_minutes")
    out = agent._enforce_session_collections(
        "q", plan, {"gazete_arsivi"}, PipelineTracer()
    )
    assert [r.collection for r in out.resources] == ["gazete_arsivi"]


def test_enforce_session_collections_emits_policy_trace():
    agent = _agent()
    plan = _plan("gazete_arsivi", "tbmm_minutes")
    tracer = PipelineTracer()
    agent._enforce_session_collections("q", plan, {"gazete_arsivi"}, tracer)
    policy_events = [e for e in tracer.events if e.phase == "policy"]
    assert policy_events
    details = policy_events[0].details
    assert details["kept"] == ["gazete_arsivi"]
    assert details["dropped"] == ["tbmm_minutes"]


def test_enforce_session_collections_rebuilds_when_all_dropped():
    """Planner tamamen yanlış koleksiyon seçerse seçili koleksiyona fallback kurulur."""
    agent = _agent()
    plan = _plan("tbmm_minutes")  # kullanıcı bunu seçmedi
    out = agent._enforce_session_collections(
        "ege adaları", plan, {"gazete_arsivi"}, PipelineTracer()
    )
    # Boş bırakmaz; seçili koleksiyonu arar.
    assert [r.collection for r in out.resources] == ["gazete_arsivi"]
    assert out.resources[0].query_drafts


def test_fallback_plan_scoped_to_allowed_keys():
    """allowed_keys verilince fallback tam olarak seçili koleksiyonları arar."""
    agent = _agent()
    plan = agent._fallback_plan("Kardak", allowed_keys={"gazete_arsivi"})
    assert [r.collection for r in plan.resources] == ["gazete_arsivi"]


def test_run_restricts_to_session_collections(monkeypatch):
    """run(session_collections=...) → sadece seçili koleksiyonda arama yapılır."""
    agent = _agent()
    # Planner tüm katalogdan iki koleksiyon seçti; kullanıcı sadece birini seçmişti.
    monkeypatch.setattr(
        agent, "_generate_plan",
        lambda q, tracer, allowed_keys=None: _plan("gazete_arsivi", "tbmm_minutes"),
    )

    searched = []

    def fake_search(*, collection_key, query_text, filters, top_k):
        searched.append(collection_key)
        return {
            "documents": ["a", "b", "c", "d", "e"],
            "metadatas": [{"chunk_id": f"{collection_key}-{i}"} for i in range(5)],
            "distances": [0.1] * 5,
        }

    monkeypatch.setattr(agent._search_tool, "search", fake_search)
    monkeypatch.setattr(
        agent._answer_tool, "generate",
        lambda query, context, mufettis_mode=False: ("t", "ok"),
    )
    monkeypatch.setattr(agent._sanitizer, "sanitize", lambda *a, **kw: "ok")
    agent._bad_words = None
    agent._classifier = None

    agent.run("soru", session_collections=["gazete_arsivi"])

    # Sadece seçili koleksiyon arandı; tbmm_minutes düşürüldü.
    assert set(searched) == {"gazete_arsivi"}


def test_reretrieval_stays_within_session_collections(monkeypatch):
    """Re-retrieval (broader plan) session seçiminin DIŞINA çıkmamalı.

    İlk arama yetersiz sonuç döndürür → broader plan üretilir; broader plan
    başka koleksiyon önerse bile session kısıtı uygulanır, sadece seçili
    koleksiyon aranır (regresyon koruması: aksi halde diğer koleksiyonların
    embedder'ları yüklenir)."""
    agent = _agent()
    # İlk plan: seçili koleksiyon. Broader plan: kapsam-dışı koleksiyon önerir.
    monkeypatch.setattr(
        agent, "_generate_plan",
        lambda q, tracer, allowed_keys=None: _plan("test"),
    )
    monkeypatch.setattr(
        agent, "_generate_broader_plan",
        lambda q, prev, results, tracer, allowed_keys=None: _plan("gazete_arsivi", "tbmm_minutes"),
    )

    searched = []

    def fake_search(*, collection_key, query_text, filters, top_k):
        searched.append(collection_key)
        # İlk turda az sonuç → re-retrieval tetiklensin.
        return {
            "documents": ["a"],
            "metadatas": [{"chunk_id": f"{collection_key}-0"}],
            "distances": [0.1],
        }

    monkeypatch.setattr(agent._search_tool, "search", fake_search)
    monkeypatch.setattr(
        agent._answer_tool, "generate",
        lambda query, context, mufettis_mode=False: ("t", "ok"),
    )
    monkeypatch.setattr(agent._sanitizer, "sanitize", lambda *a, **kw: "ok")
    agent._bad_words = None
    agent._classifier = None

    agent.run("soru", session_collections=["test"])

    # Broader plan gazete/tbmm_minutes önerse de hiçbiri aranmadı; sadece 'test'.
    assert set(searched) == {"test"}


# --- FilterExtractor entegrasyonu: filtrelerin tek kaynağı FilterExtractor ---

def test_apply_filter_extractor_populates_all_drafts():
    """FilterExtractor BİR KEZ çağrılır, çıkan FilterCriteria tüm draft'lara uygulanır."""
    mock_fe = MagicMock()
    mock_fe.model = "test-model"
    mock_fe.extract.return_value = ExtractedFilterResponse(
        refined_query="Kardak", filters=FilterCriteria(year=1996)
    )
    agent = _agent(filter_extractor=mock_fe)
    plan = _two_collection_plan()

    agent._apply_filter_extractor("1996 Kardak", plan, PipelineTracer())

    # Tek-çağrı garantisi: filtreler ifade biçiminin değil kullanıcı niyetinin özelliği.
    mock_fe.extract.assert_called_once_with("1996 Kardak")
    drafts = [d for r in plan.resources for d in r.query_drafts]
    assert len(drafts) == 4
    assert all(d.filters is not None and d.filters.year == 1996 for d in drafts)
    # Aliasing yok: her draft ayrı kopya.
    assert len({id(d.filters) for d in drafts}) == 4


def test_apply_filter_extractor_empty_filters_sets_none():
    """İpucu/filtre yoksa draft.filters None olur (filtresiz arama)."""
    mock_fe = MagicMock()
    mock_fe.model = "test-model"
    mock_fe.extract.return_value = ExtractedFilterResponse(
        refined_query="kardak krizi", filters=FilterCriteria()
    )
    agent = _agent(filter_extractor=mock_fe)
    plan = _two_collection_plan()

    agent._apply_filter_extractor("kardak krizi", plan, PipelineTracer())

    drafts = [d for r in plan.resources for d in r.query_drafts]
    assert all(d.filters is None for d in drafts)


def test_apply_filter_extractor_masks_filters_per_collection():
    """Filtreler her koleksiyona, o türün indekslediği alanlarla maskelenerek uygulanır.

    Çapraz-tür plan (tutanak + gazete): source_name (basın alanı) tutanağa,
    period (parlamento alanı) gazeteye sızmamalı.
    """
    mock_fe = MagicMock()
    mock_fe.model = "test-model"
    mock_fe.extract.return_value = ExtractedFilterResponse(
        refined_query="Kardak",
        filters=FilterCriteria(year=1996, source_name="Hürriyet", period=20),
    )
    agent = _agent(filter_extractor=mock_fe)
    plan = _two_collection_plan()  # tbmm_minutes (tutanak), gazete_arsivi (gazete)

    agent._apply_filter_extractor("1996 Hürriyet Kardak", plan, PipelineTracer())

    by_coll = {r.collection: r.query_drafts[0].filters for r in plan.resources}
    tutanak_f = by_coll["tbmm_minutes"]
    gazete_f = by_coll["gazete_arsivi"]

    # year her ikisinde de geçerli.
    assert tutanak_f.year == 1996 and gazete_f.year == 1996
    # source_name: sadece gazetede; tutanakta düşürüldü.
    assert gazete_f.source_name == "Hürriyet"
    assert tutanak_f.source_name is None
    # period: sadece tutanakta; gazetede düşürüldü.
    assert tutanak_f.period == 20
    assert gazete_f.period is None


def test_apply_filter_extractor_keeps_author_for_tutanak():
    """Konuşmacı adı tutanakta da KORUNUR (maskeleme düşürmez).

    Tutanakta `author` tam ünvan olarak indekslidir ("...DENİZ BAYKAL"), ama 0-sonuç
    tuzağı maskeleme ile değil, çeviri aşamasında build_chroma_where tarafından
    çözülür: ad koleksiyonun gerçek etiketlerine eşlenip `author $in [...]`'a çevrilir
    (bkz. test_filter_translators). Bu yüzden author hem tutanak hem gazetede
    FilterCriteria üzerinde kalır.
    """
    mock_fe = MagicMock()
    mock_fe.model = "test-model"
    mock_fe.extract.return_value = ExtractedFilterResponse(
        refined_query="Kardak kayalıkları",
        filters=FilterCriteria(author="Deniz Baykal", year=1996),
    )
    agent = _agent(filter_extractor=mock_fe)
    plan = _two_collection_plan()  # tbmm_minutes (tutanak), gazete_arsivi (gazete)

    agent._apply_filter_extractor("Deniz Baykal 1996 Kardak", plan, PipelineTracer())

    by_coll = {r.collection: r.query_drafts[0].filters for r in plan.resources}
    # author her iki türde de korunur; $in çözümü çeviri katmanında yapılır.
    assert by_coll["tbmm_minutes"].author == "Deniz Baykal"
    assert by_coll["tbmm_minutes"].year == 1996
    assert by_coll["gazete_arsivi"].author == "Deniz Baykal"


def test_apply_filter_extractor_populates_refined_query():
    """refined_query plan'a taşınır (orchestrator retrieval için)."""
    mock_fe = MagicMock()
    mock_fe.model = "test-model"
    mock_fe.extract.return_value = ExtractedFilterResponse(
        refined_query="Kardak krizi", filters=FilterCriteria(year=1996)
    )
    agent = _agent(filter_extractor=mock_fe)
    plan = _two_collection_plan()

    agent._apply_filter_extractor("1996 Kardak krizi", plan, PipelineTracer())

    assert plan.refined_query == "Kardak krizi"


def test_apply_filter_extractor_noop_without_extractor():
    """filter_extractor enjekte edilmemişse no-op; draft.filters dokunulmaz, hata yok."""
    agent = _agent(filter_extractor=None)
    plan = _two_collection_plan()

    returned = agent._apply_filter_extractor("herhangi sorgu", plan, PipelineTracer())

    drafts = [d for r in returned.resources for d in r.query_drafts]
    assert all(d.filters is None for d in drafts)


def test_legacy_run_propagates_extracted_filters_to_search(monkeypatch):
    """Legacy PlanningAgent.run yolu: FE → draft.filters → _execute_single çevirisi → search."""
    mock_fe = MagicMock()
    mock_fe.model = "test-model"
    mock_fe.extract.return_value = ExtractedFilterResponse(
        refined_query="ege adaları", filters=FilterCriteria(year=1996)
    )
    agent = _agent(filter_extractor=mock_fe)

    # Planner LLM'i atla: sabit tek-koleksiyonlu plan döndür.
    fixed_plan = SearchPlan(
        intent="factual",
        resources=[CollectionSearchPlan(
            collection="tbmm_minutes",
            query_drafts=[SearchQueryDraft(text="ege adaları")],
        )],
        reasoning="r",
    )
    monkeypatch.setattr(agent, "_generate_plan", lambda q, t, allowed_keys=None: fixed_plan)

    calls = []

    def fake_search(*, collection_key, query_text, filters, top_k):
        calls.append(filters)
        # Yeterli sonuç döndür ki re-retrieval tetiklenmesin (gerçek LLM'e gitmesin).
        return {
            "documents": ["a", "b", "c", "d", "e"],
            "metadatas": [{"chunk_id": str(i)} for i in range(5)],
            "distances": [0.1] * 5,
        }

    monkeypatch.setattr(agent._search_tool, "search", fake_search)
    # Cevap üretimi ve sanitizer'ı atla (gerçek LLM'e gitmesin).
    monkeypatch.setattr(
        agent._answer_tool, "generate",
        lambda query, context, mufettis_mode=False: ("t", "ok"),
    )
    monkeypatch.setattr(agent._sanitizer, "sanitize", lambda *a, **kw: "ok")
    # Gate'leri sadeleştir.
    agent._bad_words = None
    agent._classifier = None

    agent.run("1996 ege adaları")

    # İlk arama: FE year=1996 → ChromaFilterTranslator → tek koşul $eq.
    assert calls[0] == {"year": {"$eq": 1996}}


# --- _parse_plan robustness: LLM şema ihlallerine tolerans ---

def test_parse_plan_coerces_invalid_intent():
    """LLM intent alanına cümle yazarsa 'unknown'a düşülür, ValidationError atılmaz."""
    agent = _agent()
    plan = agent._parse_plan({
        "intent": "Deniz Baykal'ın konuşmasını detaylı bulmak.",  # geçersiz, cümle
        "resources": [{"collection": "test", "query_drafts": [{"text": "q"}]}],
        "reasoning": "r",
    })
    assert plan.intent == "unknown"


def test_parse_plan_coerces_invalid_query_type():
    """Geçersiz query_type 'fact'e düşülür."""
    agent = _agent()
    plan = agent._parse_plan({
        "intent": "factual",
        "query_type": "bilgi",  # geçersiz
        "resources": [{"collection": "test", "query_drafts": [{"text": "q"}]}],
        "reasoning": "r",
    })
    assert plan.query_type == "fact"


def test_parse_plan_drops_llm_filters():
    """LLM filtre kusarsa (geçersiz document_type dahil) parse patlamaz; filters None olur.

    Filtrelerin tek kaynağı FilterExtractor; planner LLM'in ürettiği filters yok sayılır.
    """
    agent = _agent()
    plan = agent._parse_plan({
        "intent": "factual",
        "resources": [{
            "collection": "test",
            "query_drafts": [
                {"text": "q", "filters": {"document_type": "gazete"}},  # geçersiz enum
            ],
        }],
        "reasoning": "r",
    })
    # ValidationError atılmadı ve LLM filtresi düşürüldü.
    assert plan.resources[0].query_drafts[0].filters is None


def test_parse_plan_skips_resource_without_collection():
    """'collection' eksik/boş kaynak KeyError atmadan atlanır; geçerliler korunur."""
    agent = _agent()
    plan = agent._parse_plan({
        "intent": "factual",
        "resources": [
            {"query_drafts": [{"text": "q"}]},          # collection yok → atla
            {"collection": "", "query_drafts": [{"text": "q"}]},  # boş → atla
            {"collection": "good", "query_drafts": [{"text": "q"}]},
        ],
        "reasoning": "r",
    })
    assert [r.collection for r in plan.resources] == ["good"]


def test_parse_plan_skips_draft_without_text_and_empty_resource():
    """'text' eksik draft atlanır; hiç geçerli draft kalmayan kaynak da atlanır."""
    agent = _agent()
    plan = agent._parse_plan({
        "intent": "factual",
        "resources": [
            {"collection": "c1", "query_drafts": [{"top_k": 5}, {"text": "  "}]},  # geçerli draft yok → atla
            {"collection": "c2", "query_drafts": [{"text": "ok"}, {"foo": "bar"}]},
        ],
        "reasoning": "r",
    })
    assert [r.collection for r in plan.resources] == ["c2"]
    assert [d.text for d in plan.resources[0].query_drafts] == ["ok"]


def test_parse_plan_tolerates_non_dict_entries():
    """resources/query_drafts içinde dict olmayan öğeler parse'ı patlatmaz."""
    agent = _agent()
    plan = agent._parse_plan({
        "intent": "factual",
        "resources": [
            "garbage",  # dict değil → atla
            {"collection": "c1", "query_drafts": ["nope", {"text": "ok"}]},
        ],
        "reasoning": "r",
    })
    assert [r.collection for r in plan.resources] == ["c1"]
    assert [d.text for d in plan.resources[0].query_drafts] == ["ok"]
