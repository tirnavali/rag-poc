"""Unit tests for the pipeline.yaml configuration loader."""
import pytest

from src.config.pipeline_loader import PipelineConfig, load_pipeline_config


def test_load_real_pipeline():
    cfg = load_pipeline_config()
    assert isinstance(cfg, PipelineConfig)
    assert "fast-01" in cfg.blocks
    assert cfg.planner.block == "fast-01"
    assert cfg.answering.num_predict > 0
    assert cfg.sanitizer.validation_criteria  # non-empty list


def test_get_block_raises_on_unknown():
    cfg = load_pipeline_config()
    with pytest.raises(KeyError):
        cfg.get_block("does-not-exist")


def test_missing_file_returns_none(tmp_path):
    assert load_pipeline_config(tmp_path / "nope.yaml") is None


def test_invalid_yaml_raises(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("- just\n- a\n- list\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_pipeline_config(bad)


def test_collection_catalog_lists_only_default_collections():
    """The planner catalog must expose only the live default collection per
    doc_type, not the experimental/comparison collections also registered in
    models.yaml (e.g. tbmm_minutes_docling_jina_v4) — otherwise the planner routes to dead
    collections.
    """
    from src.config.collections import DEFAULT_COLLECTION_FOR_TYPE

    cfg = load_pipeline_config()
    catalog = cfg.get_collection_catalog()
    defaults = set(DEFAULT_COLLECTION_FOR_TYPE.values())
    assert defaults

    for key in defaults:
        assert key in catalog

    # Experimental collections registered but not the default must be absent.
    non_defaults = set(cfg.get_collection_keys()) - defaults
    for key in non_defaults:
        assert key not in catalog


def test_retrieval_defaults_present():
    cfg = load_pipeline_config()
    assert cfg.retrieval.distance_threshold > 0
    assert cfg.retrieval.context_total_max_chars >= cfg.retrieval.context_max_chars


def test_bad_words_filter_config_loaded_from_yaml():
    """The loader exposes agent.bad_words_filter as a typed BadWordsFilterConfig."""
    from src.config.pipeline_loader import load_pipeline_config
    config = load_pipeline_config()
    assert config is not None

    bwf = config.bad_words_filter
    assert bwf.enabled is True
    assert isinstance(bwf.bad_words, list)
    assert "aptal" in [w.lower() for w in bwf.bad_words]
    assert "Lütfen" in bwf.response_message


def test_classifier_config_loaded_from_yaml():
    from src.config.pipeline_loader import load_pipeline_config
    config = load_pipeline_config()
    cls = config.classifier
    assert cls.enabled is True
    assert cls.block == "fast-01"
    assert cls.model_key == "classifier"
    assert 0.0 <= cls.confidence_threshold <= 1.0
    assert "kapı bekçisi" in cls.prompt or "kapı bekçisisin" in cls.prompt


def test_fast_01_has_classifier_model():
    from src.config.pipeline_loader import load_pipeline_config
    config = load_pipeline_config()
    fast = config.get_block("fast-01")
    assert fast.get_model("classifier") == "qwen2.5:3b-instruct"


def test_suggester_config_loaded_from_yaml():
    from src.config.pipeline_loader import load_pipeline_config
    config = load_pipeline_config()
    s = config.suggester
    assert s.block == "fast-01"
    assert s.model_key == "suggester"
    assert s.suggestion_count == 3
    assert "öneri uzmanısın" in s.prompt or "öneri uzman" in s.prompt
    assert "{catalog}" in s.prompt


def test_off_domain_template_loaded_from_yaml():
    from src.config.pipeline_loader import load_pipeline_config
    config = load_pipeline_config()
    tmpl = config.off_domain_response_template
    assert "{suggestion_0}" in tmpl
    assert "{suggestion_1}" in tmpl
    assert "{suggestion_2}" in tmpl


def test_off_domain_fallback_suggestions_loaded_from_yaml():
    from src.config.pipeline_loader import load_pipeline_config
    config = load_pipeline_config()
    fbs = config.off_domain_fallback_suggestions
    assert len(fbs) >= 3
    assert all(isinstance(s, str) and s for s in fbs)


def test_suggester_prompt_safe_for_format():
    """Loading + formatting the suggester prompt with the real YAML must not crash."""
    from src.config.pipeline_loader import load_pipeline_config
    config = load_pipeline_config()
    rendered = config.suggester.prompt.format(catalog="- foo (Bar)")
    assert '"suggestions"' in rendered  # JSON example survives the format call
    assert "- foo (Bar)" in rendered
