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
        c.update_details(intent="factual", resources="tbmm_minutes")
    with t.phase("retrieval") as c:
        c.update_details(collection="tbmm_minutes", result_count=2)
    with t.phase("answering", block="gpu-01", model="g") as c:
        c.update_details(context_chars=120)
    with t.phase("validation") as c:
        c.update_details(passes=True, checks={"is_turkish": True})
    t.print_trace()
    out = capsys.readouterr().out
    assert "PHASE 1: Planning" in out
    assert "PHASE 4: Validation" in out
    assert t.trace_id in out
