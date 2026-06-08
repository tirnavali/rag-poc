"""Unit tests for quality-based re-retrieval logic in PlanningAgent."""
import json
from unittest.mock import MagicMock, patch

import pytest

from src.agent.planner import NOTHING_FOUND_PATTERNS, PlanningAgent
from src.agent.schemas import SearchPlan, ValidationResult
from src.config.pipeline_loader import PlannerConfig


def _make_validation(
    addresses_query: bool = True,
    passes: bool = True,
    issues: list[str] | None = None,
) -> ValidationResult:
    return ValidationResult(
        passes=passes,
        checks={"addresses_query": addresses_query, "backed_by_sources": True},
        issues=issues or ([] if addresses_query else ["Yanıt soruyu karşılamıyor"]),
    )


def _make_agent(quality_enabled: bool = True) -> PlanningAgent:
    """Build a PlanningAgent with mocked tools for unit testing."""
    from src.config.pipeline_loader import PipelineConfig

    config_dict = {
        "deployment_blocks": {
            "fast-01": {
                "host": "http://localhost:11434",
                "models": {"planner": "test-model", "sanitizer": "test-model"},
            },
            "gpu-01": {
                "host": "http://localhost:11434",
                "models": {"answer": "test-model"},
            },
        },
        "agent": {
            "planner": {
                "block": "fast-01",
                "model_key": "planner",
                "re_retrieval": {
                    "enabled": True,
                    "max_retries": 1,
                    "trigger_min_results": 3,
                    "on_quality_failure": quality_enabled,
                },
                "fallback": {"default_collections": [], "default_queries": []},
            },
            "answering": {"block": "gpu-01", "model_key": "answer"},
            "sanitizer": {
                "block": "fast-01",
                "model_key": "sanitizer",
                "validation_criteria": [],
            },
        },
        "retrieval": {},
    }
    config = PipelineConfig(config_dict)
    client_pool = MagicMock()
    with (
        patch("src.agent.planner.SearchTool"),
        patch("src.agent.planner.ContextBuilderTool"),
        patch("src.agent.planner.AnswerTool"),
        patch("src.agent.planner.SanitizerAgent"),
    ):
        return PlanningAgent(config=config, client_pool=client_pool)


# --- NOTHING_FOUND_PATTERNS ---

class TestNothingFoundPatterns:
    def test_bulunamadi_matches(self):
        assert NOTHING_FOUND_PATTERNS.search("Bu bilgi kaynaklarda bulunamadı.")

    def test_yer_almamaktadir_matches(self):
        assert NOTHING_FOUND_PATTERNS.search("Bu konu kaynaklarda yer almamaktadır.")

    def test_bilgi_yok_matches(self):
        assert NOTHING_FOUND_PATTERNS.search("Hakkında bilgi yok.")

    def test_tespit_edilemedi_matches(self):
        assert NOTHING_FOUND_PATTERNS.search("Kim söyledi tespit edilemedi.")

    def test_positive_answer_no_match(self):
        assert not NOTHING_FOUND_PATTERNS.search(
            "Deniz Baykal mecliste 'merdikıptı' dedi."
        )

    def test_case_insensitive(self):
        assert NOTHING_FOUND_PATTERNS.search("BULUNAMADI")


# --- _needs_quality_reretrieval ---

class TestNeedsQualityReretrieval:
    def test_addresses_query_false_triggers(self):
        agent = _make_agent()
        validation = _make_validation(addresses_query=False, passes=False)
        assert agent._needs_quality_reretrieval("Herhangi bir yanıt.", validation)

    def test_keyword_match_triggers_even_when_passes(self):
        agent = _make_agent()
        validation = _make_validation(addresses_query=True, passes=True)
        assert agent._needs_quality_reretrieval(
            "Bu bilgi kaynaklarda bulunamadı.", validation
        )

    def test_clean_answer_no_trigger(self):
        agent = _make_agent()
        validation = _make_validation(addresses_query=True, passes=True)
        assert not agent._needs_quality_reretrieval(
            "Deniz Baykal 1997'de şöyle dedi: ...", validation
        )

    def test_disabled_config_no_trigger(self):
        agent = _make_agent(quality_enabled=False)
        validation = _make_validation(addresses_query=False, passes=False)
        assert not agent._needs_quality_reretrieval("bulunamadı", validation)

    def test_validation_fails_with_empty_checks_triggers(self):
        agent = _make_agent()
        validation = ValidationResult(passes=False, checks={}, issues=["unknown failure"])
        assert agent._needs_quality_reretrieval("Herhangi bir yanıt.", validation)


# --- _generate_gap_fill_plan ---

class TestGenerateGapFillPlan:
    def test_returns_plan_on_valid_llm_response(self):
        agent = _make_agent()

        plan_json = json.dumps({
            "intent": "factual",
            "resources": [
                {
                    "collection": "tutanaklar_jina_v3_4k",
                    "mode": "parallel",
                    "priority": 1,
                    "query_drafts": [
                        {"text": "merdikıptı meclis", "filters": {"period": 23}, "top_k": 8}
                    ],
                }
            ],
            "reasoning": "gap fill",
        })

        mock_client = MagicMock()
        mock_client.chat.return_value.message.content = plan_json
        agent._pool.get_client.return_value = mock_client
        agent._pool.get_model_for_block.return_value = "test-model"

        with patch.object(agent._config, "get_collection_catalog", return_value="catalog"):
            validation = _make_validation(addresses_query=False, passes=False)
            result = agent._generate_gap_fill_plan(
                query="kim kime merdikıptı dedi",
                answer="bulunamadı",
                validation=validation,
                tracer=MagicMock(),
            )

        assert result is not None
        assert isinstance(result, SearchPlan)
        assert result.resources[0].collection == "tutanaklar_jina_v3_4k"
        assert result.resources[0].query_drafts[0].top_k == 8

    def test_returns_none_on_llm_error(self):
        agent = _make_agent()
        mock_client = MagicMock()
        mock_client.chat.side_effect = RuntimeError("LLM unavailable")
        agent._pool.get_client.return_value = mock_client
        agent._pool.get_model_for_block.return_value = "test-model"

        with patch.object(agent._config, "get_collection_catalog", return_value="catalog"):
            validation = _make_validation(addresses_query=False)
            result = agent._generate_gap_fill_plan(
                query="test",
                answer="bulunamadı",
                validation=validation,
                tracer=MagicMock(),
            )

        assert result is None
