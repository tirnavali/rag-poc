"""Unit tests for the grounded clarification stage (offline; no LLM)."""
from __future__ import annotations

from src.agent.clarifier import AmbiguityGate, FacetMiner, QueryRefiner
from src.agent.schemas import FacetSet, FacetValue
from src.agent.tracer import PipelineTracer
from src.config.pipeline_loader import ClarificationConfig


def _results():
    return [{
        "metadatas": [
            {"year": 1997, "topics": ["bütçe", "ekonomi"], "author": "A", "_source_collection": "col1"},
            {"year": 2001, "topics": "bütçe", "author": "B", "_source_collection": "col1"},
            {"year": 2010, "topics": ["dış politika"], "author": "A", "_source_collection": "col2"},
        ]
    }]


def test_facet_miner_counts_metadata():
    fs = FacetMiner().mine(_results())
    assert fs.total == 3
    assert {f.value for f in fs.years} == {"1997", "2001", "2010"}
    assert fs.topics[0].value == "bütçe" and fs.topics[0].count == 2
    assert {f.value for f in fs.collections} == {"col1", "col2"}


def test_facet_miner_handles_string_and_list_topics():
    res = [{"metadatas": [{"topics": "a, b; c"}, {"topics": ["a", "d"]}]}]
    fs = FacetMiner().mine(res)
    values = {f.value for f in fs.topics}
    assert {"a", "b", "c", "d"} <= values


def test_ambiguity_gate_triggers_on_many_years():
    cfg = ClarificationConfig({"ambiguity": {"min_distinct_years": 3, "dominance_ratio": 0.6}})
    fs = FacetMiner().mine(_results())  # 3 distinct years
    assert AmbiguityGate(cfg).is_ambiguous(fs) is True


def test_ambiguity_gate_skips_when_dominant_and_few_years():
    cfg = ClarificationConfig({"ambiguity": {"min_distinct_years": 3, "dominance_ratio": 0.6}})
    fs = FacetSet(
        years=[FacetValue(value="1997", count=9), FacetValue(value="1998", count=1)],
        collections=[FacetValue(value="col1", count=10)],
        total=10,
    )
    assert AmbiguityGate(cfg).is_ambiguous(fs) is False


def test_ambiguity_gate_false_on_empty():
    cfg = ClarificationConfig({})
    assert AmbiguityGate(cfg).is_ambiguous(FacetSet(total=0)) is False


def _ohal_facets():
    """Mirrors the live "ohal" probe: results cluster in 2 years, one collection."""
    return FacetSet(
        years=[FacetValue(value="1980", count=9), FacetValue(value="1981", count=1)],
        collections=[FacetValue(value="tutanaklar", count=10)],
        total=10,
    )


def test_ambiguity_gate_triggers_on_vague_query():
    """Broad/unscoped query ("...hakkında bilgi") clarifies even when facets cluster."""
    cfg = ClarificationConfig({"ambiguity": {"min_distinct_years": 3, "dominance_ratio": 0.6}})
    fs = _ohal_facets()  # would be False on facet-diversity alone
    assert AmbiguityGate(cfg).is_ambiguous(fs) is False  # no query → not vague
    assert AmbiguityGate(cfg).is_ambiguous(fs, "ohal konusunda bilgi istiyorum") is True


def test_ambiguity_gate_specific_query_with_year_not_vague():
    """An explicit year means the user already scoped it → no clarification."""
    cfg = ClarificationConfig({"ambiguity": {"min_distinct_years": 3, "dominance_ratio": 0.6}})
    fs = _ohal_facets()
    assert AmbiguityGate(cfg).is_ambiguous(fs, "1980 ohal hakkında bilgi") is False


def test_ambiguity_gate_vague_query_needs_askable_facet():
    """Vague query but nothing to ask about (single year/collection) → no clarification."""
    cfg = ClarificationConfig({"ambiguity": {"min_distinct_years": 3, "dominance_ratio": 0.6}})
    fs = FacetSet(
        years=[FacetValue(value="1980", count=10)],
        collections=[FacetValue(value="tutanaklar", count=10)],
        total=10,
    )
    assert AmbiguityGate(cfg).is_ambiguous(fs, "ohal hakkında bilgi") is False


def test_ambiguity_gate_vague_clarify_disabled():
    cfg = ClarificationConfig({"ambiguity": {"vague_query_clarify": False}})
    assert AmbiguityGate(cfg).is_ambiguous(_ohal_facets(), "ohal hakkında bilgi") is False


def test_query_refiner_build_questions_grounded_options():
    # pool=None → LLM phrasing fails gracefully to default texts; options stay grounded.
    cfg = ClarificationConfig({"question_count": 3})

    class _Cfg:
        clarification = cfg
    refiner = QueryRefiner(None, _Cfg())
    fs = FacetMiner().mine(_results())
    qs = refiner.build_questions("meclis ne konuştu", fs, PipelineTracer())
    axes = {q.axis for q in qs}
    assert "year" in axes and "topic" in axes
    year_q = next(q for q in qs if q.axis == "year")
    assert set(year_q.options) <= {"1997", "2001", "2010"}
    assert year_q.text  # non-empty default text


def test_query_refiner_resolve_maps_answers_to_constraints():
    cfg = ClarificationConfig({})

    class _Cfg:
        clarification = cfg
    refiner = QueryRefiner(None, _Cfg())
    fs = FacetMiner().mine(_results())
    qs = refiner.build_questions("q", fs, PipelineTracer())
    constraints = refiner.resolve(qs, {"year": "2001", "topic": "bütçe"})
    assert constraints["year"] == 2001
    assert constraints["topic"] == "bütçe"


def test_query_refiner_resolve_skips_blank_and_unknown_axes():
    cfg = ClarificationConfig({})

    class _Cfg:
        clarification = cfg
    refiner = QueryRefiner(None, _Cfg())
    fs = FacetMiner().mine(_results())
    qs = refiner.build_questions("q", fs, PipelineTracer())
    # blank year (skip) + unknown axis ignored
    constraints = refiner.resolve(qs, {"year": "", "bogus": "x"})
    assert "year" not in constraints


def test_query_refiner_auto_constraints_prefers_top_year():
    # DEPRECATED path kept for backward compat; no longer used by the orchestrator.
    refiner = QueryRefiner(None, None)
    fs = FacetMiner().mine(_results())
    constraints, note = refiner.auto_constraints(fs)
    assert "year" in constraints
    assert note  # assumption note surfaced for non-interactive mode


def test_rabbit_holes_diversifies_across_axes():
    refiner = QueryRefiner(None, None)
    fs = FacetMiner().mine(_results())
    out = refiner.rabbit_holes("susurluk", fs, count=3)
    assert len(out) == 3
    # All suggestions are query + a mined facet value (top year, topic, author).
    assert all(s.startswith("susurluk ") for s in out)
    # Diversified, not three years: includes the top topic and top author too.
    assert "susurluk 1997" in out
    assert "susurluk bütçe" in out
    assert "susurluk A" in out


def test_rabbit_holes_skips_facet_values_already_in_query():
    refiner = QueryRefiner(None, None)
    fs = FacetMiner().mine(_results())
    out = refiner.rabbit_holes("1997 bütçe", fs, count=3)
    # 1997 and bütçe are already in the query → they must not be re-suggested.
    assert "1997 bütçe 1997" not in out
    assert "1997 bütçe bütçe" not in out
    assert all(s != "1997 bütçe" for s in out)


def test_rabbit_holes_skips_turkish_dotted_i_author_already_in_query():
    """Regression: bare str.lower() maps "İ" (U+0130) to "i̇" (i + combining
    dot), which never substring-matches a plain-typed query — so an author
    already named in the query (e.g. "engin özkoç") used to leak back as a
    redundant suggestion ("... ENGİN ÖZKOÇ"). normalize_tr() must fix this."""
    refiner = QueryRefiner(None, None)
    fs = FacetSet(
        years=[FacetValue(value="2019", count=5)],
        authors=[FacetValue(value="ENGİN ÖZKOÇ", count=5)],
        collections=[FacetValue(value="tutanaklar", count=5)],
        total=5,
    )
    out = refiner.rabbit_holes("engin özkoç konuyla ilgili başka açıklama yaptı mı?", fs, count=3)
    # ENGİN ÖZKOÇ is already named in the query → must not be re-suggested,
    # even though the query was typed all-lowercase and the facet is uppercase.
    assert not any(s.endswith("ENGİN ÖZKOÇ") for s in out)
    # The other facet (year) is unrelated and should still come through.
    assert any(s.endswith("2019") for s in out)


def test_rabbit_holes_empty_when_no_facets():
    refiner = QueryRefiner(None, None)
    assert refiner.rabbit_holes("anything", FacetSet(total=0), count=3) == []
