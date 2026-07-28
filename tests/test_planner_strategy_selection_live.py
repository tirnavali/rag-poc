"""Live-LLM integration tests for PLANNER strategy selection.

These hit the REAL planner LLM (Ollama block `fast-01` / model `gemma4:31b`, per
pipeline.yaml) — NOT mocked — and assert which `strategy` the planner picks for a
given query. They exist to lock in the cemal-kaşıkçı regression fix: a general
event/person query must NOT be routed to a law strategy (kanun_*), which previously
forced comprehensive + adaptive over-gather and broke the answer.

Marked `integration`; the whole module SKIPS gracefully when the planner block's own
host/model is unreachable (mirrors tests/test_filter_extractor_golden.py). Run with:
    .venv/bin/python -m pytest tests/test_planner_strategy_selection_live.py -v -m integration

Note: the planner runs at temperature 0.0 so selection is fairly deterministic, but LLM
output is never guaranteed. Event/person cases use a NEGATIVE assertion (not a law
strategy — the actual bug); positive cases target strongly-triggered strategies.
"""
from __future__ import annotations

import ollama
import pytest

from src.agent.planner import Planner
from src.common.llm_client_pool import LLMClientPool
from src.config.pipeline_loader import load_pipeline_config

LAW_STRATEGIES = {"kanun_kabul_oylama", "kanun_gorusmeleri", "kanun_rapor_bolumu"}


def _planner_host_up() -> bool:
    """Guard against the PLANNER block's OWN host/model (cfg.planner.block → fast-01 →
    http://localhost:11434), not settings.OLLAMA_HOST — the planner connects to block.host."""
    try:
        cfg = load_pipeline_config()
        block = cfg.get_block(cfg.planner.block)
        model = block.get_model(cfg.planner.model_key)
        listed = ollama.Client(host=block.host).list()
        names = {m.get("model", m.get("name", "")) for m in listed.get("models", [])}
        return any(n == model or n.split(":")[0] == model.split(":")[0] for n in names)
    except Exception:
        return False


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not _planner_host_up(),
        reason="planner Ollama block/model unreachable",
    ),
]


def _real_planner() -> Planner:
    """Real Planner against the configured LLM — deliberately does NOT patch
    _pool.get_client, so the actual BlockClient/LLM is used."""
    cfg = load_pipeline_config()
    return Planner(cfg, LLMClientPool.from_config(cfg))


def test_kasikci_query_does_not_pick_law_strategy():
    """THE regression: an event/person query wrongly routed to kanun_gorusmeleri."""
    planner = _real_planner()
    plan = planner.plan("cemal kaşıkçı nın ölümü ile ilgili kim ne konuşmuştur")
    assert plan.strategy not in LAW_STRATEGIES, (
        f"kanun-dışı olay sorgusu bir kanun stratejisi seçti: {plan.strategy!r}"
    )


@pytest.mark.parametrize("query", [
    "cemal kaşıkçı nın ölümü ile ilgili kim ne konuşmuştur",
    "15 temmuz darbe girişimi hakkında mecliste kim ne konuştu",
    "ayasofya nın açılışı hakkında hangi milletvekili ne dedi",
    "deprem sonrası yardımlar konusunda yapılan eleştiriler neler",
])
def test_non_law_queries_avoid_law_strategy(query):
    """Genel olay/kişi/gündem sorguları hiçbir kanun stratejisi seçmemeli."""
    plan = _real_planner().plan(query)
    assert plan.strategy not in LAW_STRATEGIES, (
        f"{query!r} → beklenmedik kanun stratejisi: {plan.strategy!r}"
    )


@pytest.mark.parametrize("query,expected", [
    ("Karayolları Trafik Kanunu kaç oyla kabul edildi", "kanun_kabul_oylama"),
    ("5 sıra sayılı kanun teklifinin tüm görüşmelerinde milletvekilleri ne dedi",
     "kanun_gorusmeleri"),
])
def test_law_queries_pick_expected_law_strategy(query, expected):
    """Gerçek kanun sorguları doğru kanun stratejisini seçmeli — aşırı-kapılamadığımızı kanıtlar."""
    plan = _real_planner().plan(query)
    assert plan.strategy == expected, f"{query!r} → {plan.strategy!r} (beklenen {expected!r})"


@pytest.mark.parametrize("query,expected", [
    ("tüm meclis araştırma önergelerini listele", "enumerate"),
    ("2018 yılı bütçe görüşmelerini kısaca özetle", "summarize"),
])
def test_generic_queries_pick_expected_strategy(query, expected):
    """Net tetikleyicili genel sorgular ilgili genel stratejiyi seçmeli."""
    plan = _real_planner().plan(query)
    assert plan.strategy == expected, f"{query!r} → {plan.strategy!r} (beklenen {expected!r})"
