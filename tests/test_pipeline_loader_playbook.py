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


def test_pipeline_config_exposes_real_playbook():
    """Regression guard: the live research_strategies.md at the project root
    parses cleanly and is reachable via PipelineConfig's accessor methods."""
    cfg = load_pipeline_config()

    assert cfg.get_strategy_catalog()  # non-empty: the live file exists
    enumerate_strategy = cfg.get_strategy("enumerate")
    assert enumerate_strategy is not None
    assert enumerate_strategy["query_type"] == "comprehensive"
    assert cfg.get_strategy("does_not_exist_strategy") is None
