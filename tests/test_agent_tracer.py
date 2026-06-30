"""Unit tests for the PipelineTracer observability layer."""
import pytest

from src.agent.tracer import PipelineTracer


def test_phase_records_event():
    t = PipelineTracer()
    with t.phase("planning", block="fast-01", model="m"):
        pass
    assert len(t.events) == 1
    e = t.events[0]
    assert e.phase == "planning"
    assert e.block == "fast-01"
    assert e.model == "m"
    assert e.latency_ms >= 0
    assert e.trace_id == t.trace_id


def test_update_details_binds_late():
    t = PipelineTracer()
    with t.phase("planning") as ctx:
        ctx.update_details(intent="factual", result_count=3)
    assert t.events[0].details["intent"] == "factual"
    assert t.events[0].details["result_count"] == 3


def test_total_latency_sums_events():
    t = PipelineTracer()
    with t.phase("a"):
        pass
    with t.phase("b"):
        pass
    assert len(t.events) == 2
    assert t.total_latency_ms == pytest.approx(sum(e.latency_ms for e in t.events))


def test_custom_trace_id():
    t = PipelineTracer(trace_id="deadbeef")
    assert t.trace_id == "deadbeef"


def test_total_latency_empty():
    assert PipelineTracer().total_latency_ms == 0.0


def test_on_phase_callback_invoked_at_phase_start():
    seen = []
    t = PipelineTracer(
        on_phase=lambda name, block, model, details: seen.append((name, details.get("collection")))
    )
    with t.phase("retrieval", details={"collection": "tbmm_minutes"}):
        # callback fires at __enter__, before the body runs
        assert seen == [("retrieval", "tbmm_minutes")]
    assert seen == [("retrieval", "tbmm_minutes")]


def test_on_phase_callback_error_does_not_break_pipeline():
    def boom(*args, **kwargs):
        raise RuntimeError("UI callback exploded")

    t = PipelineTracer(on_phase=boom)
    with t.phase("planning", block="fast-01", model="m"):
        pass
    # phase still recorded despite the callback raising
    assert len(t.events) == 1


def test_print_trace_smoke(capsys):
    t = PipelineTracer()
    with t.phase("planning", block="fast-01", model="m") as c:
        c.update_details(intent="factual", collections=["tutanaklar_nomic_chunk256_768d"])
    with t.phase("retrieval") as c:
        c.update_details(per_collection={"tutanaklar_nomic_chunk256_768d": {"fetched": 10, "returned": 3}})
    with t.phase("judge") as c:
        c.update_details(action="answer", judge_type="heuristic", confidence=0.85)
    with t.phase("answering", block="gpu-01", model="g") as c:
        c.update_details(context_chars=120)
    with t.phase("validation") as c:
        c.update_details(passes=True)
    t.print_trace()
    out = capsys.readouterr().out
    assert "Planning" in out
    assert "Retrieval" in out
    assert "total: 3 results" in out  # per_collection 'returned' is summed, not 0
    assert "Validation" in out
    assert t.trace_id in out


def test_on_phase_end_fires_with_completed_event():
    """on_phase_end receives the completed event with its filled-in details."""
    seen = []
    t = PipelineTracer(on_phase_end=lambda ev: seen.append(ev))
    with t.phase("judge") as c:
        c.update_details(action="answer", reasoning="yeterli kanıt")
    assert len(seen) == 1
    assert seen[0].phase == "judge"
    assert seen[0].details["reasoning"] == "yeterli kanıt"


def test_on_phase_end_errors_never_break_pipeline():
    def boom(ev):
        raise RuntimeError("listener exploded")
    t = PipelineTracer(on_phase_end=boom)
    with t.phase("planning"):
        pass
    assert len(t.events) == 1  # phase still recorded
