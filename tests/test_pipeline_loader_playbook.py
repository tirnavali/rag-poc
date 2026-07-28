"""Unit tests for StrategyPlaybook (research_strategies.md parser) and its
wiring into PipelineConfig.get_strategy_catalog()/get_strategy().
"""
from __future__ import annotations

from src.config.pipeline_loader import StrategyPlaybook, load_pipeline_config

_SAMPLE_MD = """# Playbook

## enumerate
triggers: tüm, hepsi, listele
query_type: comprehensive
answer_directive: Eldeki TÜM kayıtları topla; kısmi kanıttan
  bile sentezle, asla boş dönme.

## factual
triggers:
query_type: fact
answer_directive:
"""


def test_playbook_parses_sections_into_by_name(tmp_path):
    md = tmp_path / "research_strategies.md"
    md.write_text(_SAMPLE_MD, encoding="utf-8")

    playbook = StrategyPlaybook(path=md)

    assert set(playbook.by_name.keys()) == {"enumerate", "factual"}
    enumerate_spec = playbook.by_name["enumerate"]
    assert enumerate_spec["query_type"] == "comprehensive"
    assert enumerate_spec["triggers"] == ["tüm", "hepsi", "listele"]
    # Continuation line is joined onto the same directive.
    assert "kısmi kanıttan bile sentezle, asla boş dönme." in enumerate_spec["answer_directive"]


def test_playbook_empty_fields_yield_falsy_values(tmp_path):
    md = tmp_path / "research_strategies.md"
    md.write_text(_SAMPLE_MD, encoding="utf-8")

    playbook = StrategyPlaybook(path=md)

    factual = playbook.by_name["factual"]
    assert factual["triggers"] == []
    assert factual["answer_directive"] == ""
    assert factual["query_type"] == "fact"


def test_playbook_catalog_text_mentions_each_strategy(tmp_path):
    md = tmp_path / "research_strategies.md"
    md.write_text(_SAMPLE_MD, encoding="utf-8")

    playbook = StrategyPlaybook(path=md)

    assert "enumerate" in playbook.catalog_text
    assert "factual" in playbook.catalog_text
    assert "comprehensive" in playbook.catalog_text


def test_playbook_missing_file_fails_open(tmp_path):
    playbook = StrategyPlaybook(path=tmp_path / "does_not_exist.md")

    assert playbook.by_name == {}
    assert playbook.catalog_text == ""


_MULTIHOP_MD = """# Playbook

## kanun_kabul_oylama
triggers: kaç oyla, kabul edildi
query_type: reasoning
mode: adaptive
max_rounds: 4
anchor: esas_no
target: oy_dokumu
exclude_seen_chunks: true
aliases: genel gerekçe -> sıra sayısı raporu ; muhalefet şerhi -> karşı oy
answer_directive: Oy sayılarını esas no ile ilişkilendirerek ver.
procedure:
  Hop 1 — ÇIPA: esas no'yu metinden çıkar.
  Hop 2 — HEDEF: oylama sonucunu ara.

## factual
query_type: fact
"""


def test_playbook_parses_multihop_optional_fields(tmp_path):
    md = tmp_path / "research_strategies.md"
    md.write_text(_MULTIHOP_MD, encoding="utf-8")

    spec = StrategyPlaybook(path=md).by_name["kanun_kabul_oylama"]

    assert spec["mode"] == "adaptive"
    assert spec["max_rounds"] == 4
    assert spec["anchor"] == "esas_no"
    assert spec["target"] == "oy_dokumu"
    assert spec["exclude_seen_chunks"] is True
    assert spec["aliases"] == [
        {"term": "genel gerekçe", "official_phrase": "sıra sayısı raporu"},
        {"term": "muhalefet şerhi", "official_phrase": "karşı oy"},
    ]
    # procedure is preserved (multi-line value collapses to one space-joined string).
    assert "Hop 1" in spec["procedure"] and "Hop 2" in spec["procedure"]


def test_playbook_optional_fields_have_safe_defaults(tmp_path):
    """A strategy declaring none of the multi-hop keys stays static/None — the
    orchestrator's existing query_type + answer_directive path is unaffected."""
    md = tmp_path / "research_strategies.md"
    md.write_text(_MULTIHOP_MD, encoding="utf-8")

    factual = StrategyPlaybook(path=md).by_name["factual"]

    assert factual["mode"] == "static"
    assert factual["max_rounds"] is None
    assert factual["anchor"] is None
    assert factual["target"] is None
    assert factual["exclude_seen_chunks"] is False
    assert factual["aliases"] == []
    assert factual["procedure"] == ""


def test_playbook_malformed_optional_fields_fail_open(tmp_path):
    md = tmp_path / "research_strategies.md"
    md.write_text(
        "## s\nmode: bogus\nmax_rounds: notanint\naliases: no arrow here\n",
        encoding="utf-8",
    )

    spec = StrategyPlaybook(path=md).by_name["s"]

    assert spec["mode"] == "static"      # unknown mode → safe default
    assert spec["max_rounds"] is None    # non-int → dropped, not raised
    assert spec["aliases"] == []         # missing '->' → skipped


def test_playbook_exclude_seen_chunks_parses_bool(tmp_path):
    md = tmp_path / "research_strategies.md"
    md.write_text(
        "## on\nexclude_seen_chunks: true\n"
        "## blank\nexclude_seen_chunks:   \n"
        "## bogus\nexclude_seen_chunks: maybe\n",
        encoding="utf-8",
    )

    by_name = StrategyPlaybook(path=md).by_name

    assert by_name["on"]["exclude_seen_chunks"] is True
    assert by_name["blank"]["exclude_seen_chunks"] is False   # blank → False
    assert by_name["bogus"]["exclude_seen_chunks"] is False   # unrecognized → False (fail-safe)


def test_playbook_parses_section_type(tmp_path):
    """section_type geçerli SECTION_TYPES değeriyse tutulur; geçersiz/boş → None (fail-open)."""
    md = tmp_path / "research_strategies.md"
    md.write_text(
        "## ok\nsection_type: kanun_gorusmeleri\n"
        "## bogus\nsection_type: bilinmeyen_bolum\n"
        "## blank\nsection_type:   \n",
        encoding="utf-8",
    )

    by_name = StrategyPlaybook(path=md).by_name

    assert by_name["ok"]["section_type"] == "kanun_gorusmeleri"
    assert by_name["bogus"]["section_type"] is None   # enum dışı → düşer, patlamaz
    assert by_name["blank"]["section_type"] is None   # boş → None


def test_pipeline_config_exposes_real_playbook():
    """Regression guard: the live research_strategies.md at the project root
    parses cleanly and is reachable via PipelineConfig's accessor methods."""
    cfg = load_pipeline_config()

    assert cfg.get_strategy_catalog()  # non-empty: the live file exists
    enumerate_strategy = cfg.get_strategy("enumerate")
    assert enumerate_strategy is not None
    assert enumerate_strategy["query_type"] == "comprehensive"
    assert cfg.get_strategy("does_not_exist_strategy") is None

    # The live multi-hop strategy parses with its adaptive schema intact.
    kanun = cfg.get_strategy("kanun_kabul_oylama")
    assert kanun is not None
    assert kanun["mode"] == "adaptive"
    assert kanun["anchor"] == "esas_no"
    assert {"term": "genel gerekçe", "official_phrase": "sıra sayısı raporu"} in kanun["aliases"]
    assert kanun["exclude_seen_chunks"] is True

    # Bölüm omurgası: üç adaptif kanun stratejisi hedef section_type'ını pinler.
    assert kanun["section_type"] == "oylama"
    assert cfg.get_strategy("kanun_gorusmeleri")["section_type"] == "kanun_gorusmeleri"
    assert cfg.get_strategy("kanun_rapor_bolumu")["section_type"] == "kanun_raporu"

    # Scoped deliberately: kanun_gorusmeleri COLLECTS across birleşims on purpose
    # (comprehensive), and kanun_rapor_bolumu's target lives in a report document —
    # neither opts into chunk-id novelty exclusion.
    assert cfg.get_strategy("kanun_gorusmeleri")["exclude_seen_chunks"] is False
    assert cfg.get_strategy("kanun_rapor_bolumu")["exclude_seen_chunks"] is False


def test_playbook_parses_evidence_priority_patterns(tmp_path):
    md = tmp_path / "research_strategies.md"
    md.write_text(
        "## s\n"
        "mode: adaptive\n"
        "evidence_priority_patterns: Oylama Sonucunu Duyuruyorum; kullanılan oy ;; \n"
        "procedure: Hop 1 — bul.\n"
        "\n## plain\nquery_type: fact\n",
        encoding="utf-8",
    )
    book = StrategyPlaybook(path=md)

    # `;`-separated, casefolded, empties dropped.
    assert book.by_name["s"]["evidence_priority_patterns"] == [
        "oylama sonucunu duyuruyorum", "kullanılan oy",
    ]
    # Safe default: strategies without the key expose an empty list.
    assert book.by_name["plain"]["evidence_priority_patterns"] == []


def test_live_playbook_kanun_kabul_oylama_has_priority_patterns():
    kanun = load_pipeline_config().get_strategy("kanun_kabul_oylama")
    assert kanun["evidence_priority_patterns"], "vote strategy must prioritize its target record"
    assert "oylama sonucunu duyuruyorum" in kanun["evidence_priority_patterns"]
