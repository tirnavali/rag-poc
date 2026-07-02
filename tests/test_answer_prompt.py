"""Unit tests for AnswerTool system-prompt selection (offline, no LLM)."""
from __future__ import annotations

from src.agent.tools import AnswerTool
from src.generator.prompts import (
    CONVERSATIONAL_SYS_PROMPT,
    MUFETTIS_SYS_PROMPT,
    SYNTHESIS_SYS_PROMPT,
    SYS_PROMPT,
)


def test_select_prompt_mufettis_wins():
    # müfettiş mode takes precedence regardless of query_type
    assert AnswerTool._select_system_prompt(True, "fact") is MUFETTIS_SYS_PROMPT
    assert AnswerTool._select_system_prompt(True, "comprehensive") is MUFETTIS_SYS_PROMPT


def test_select_prompt_synthesis_for_gather_types():
    for qt in ("comprehensive", "summary", "comparison", "reasoning"):
        assert AnswerTool._select_system_prompt(False, qt) is SYNTHESIS_SYS_PROMPT


def test_select_prompt_strict_for_fact_and_policy():
    # fact/policy keep the strict prompt (honest refusal when no evidence)
    assert AnswerTool._select_system_prompt(False, "fact") is SYS_PROMPT
    assert AnswerTool._select_system_prompt(False, "policy") is SYS_PROMPT


def test_select_prompt_defaults_to_strict_when_unknown():
    assert AnswerTool._select_system_prompt(False, None) is SYS_PROMPT
    assert AnswerTool._select_system_prompt(False, "weird") is SYS_PROMPT


def test_synthesis_prompt_never_empty_rule():
    # The synthesis prompt must instruct against empty output (the empty-answer fix).
    assert "ASLA boş yanıt verme" in SYNTHESIS_SYS_PROMPT
    # ...while still forbidding fabrication.
    assert "uydurma" in SYNTHESIS_SYS_PROMPT


def test_conversational_prompt_claims_archive_access():
    # The chit-chat path must not deny archive access when asked "who are you /
    # what sources do you have" — it does have a real TBMM tutanak archive,
    # confirmed by the same session's retrieval turns working correctly.
    assert "TBMM" in CONVERSATIONAL_SYS_PROMPT
    assert "tutanak" in CONVERSATIONAL_SYS_PROMPT
    # Explicitly instructed not to claim it lacks archive access.
    assert "YANLIŞ" in CONVERSATIONAL_SYS_PROMPT
