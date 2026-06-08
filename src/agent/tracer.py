"""Pipeline traceability — structured logging for agent pipeline phases."""
from __future__ import annotations

import time
import uuid
from datetime import datetime
from typing import Any, Optional

from rich.console import Console
from rich.panel import Panel

from src.agent.schemas import AgentTraceEvent


class PipelineTracer:
    """Collects and emits trace events for each pipeline phase.

    Usage:
        tracer = PipelineTracer()
        with tracer.phase("planning", block="fast-01", model="qwen2.5:7b-instruct"):
            ... do work ...
    """

    def __init__(
        self,
        trace_id: str | None = None,
        on_phase: Optional[Any] = None,
    ) -> None:
        self.trace_id = trace_id or uuid.uuid4().hex[:12]
        self.events: list[AgentTraceEvent] = []
        self._on_phase = on_phase
        self._start_time: float | None = None
        self._current_phase: str | None = None
        self._current_block: str | None = None
        self._current_model: str | None = None

    def _emit_phase_start(
        self,
        name: str,
        block: str | None,
        model: str | None,
        details: dict[str, Any],
    ) -> None:
        """Notify an optional listener that a phase has started (for live UI progress).

        A listener error must never break the pipeline.
        """
        if self._on_phase is None:
            return
        try:
            self._on_phase(name, block, model, details)
        except Exception:
            pass

    @property
    def total_latency_ms(self) -> float:
        if not self.events:
            return 0.0
        return sum(e.latency_ms for e in self.events)

    def phase(
        self,
        name: str,
        *,
        block: str | None = None,
        model: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> "_PhaseContext":
        return _PhaseContext(self, name, block, model, details or {})

    def _record(
        self,
        phase: str,
        latency_ms: float,
        *,
        block: str | None = None,
        model: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> AgentTraceEvent:
        event = AgentTraceEvent(
            trace_id=self.trace_id,
            phase=phase,
            block=block,
            model=model,
            latency_ms=round(latency_ms, 1),
            details=details or {},
        )
        self.events.append(event)
        return event

    def print_trace(self, console: Console | None = None) -> None:
        """Print the full pipeline trace to console."""
        console = console or Console()
        lines: list[str] = []

        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        lines.append(f"[dim]Trace:[/dim] {self.trace_id} | [dim]{ts}[/dim]")
        lines.append("")

        planning_events = [e for e in self.events if e.phase == "planning"]
        retrieval_events = [e for e in self.events if e.phase in ("retrieval", "re_retrieval", "quality_reretrieval")]
        answering_events = [e for e in self.events if e.phase == "answering"]
        validation_events = [e for e in self.events if e.phase == "validation"]

        if planning_events:
            lines.append("[bold cyan]PHASE 1: Planning[/bold cyan]")
            for ev in planning_events:
                block_info = f"[dim]{ev.block}[/dim]" if ev.block else ""
                model_info = f"[dim]{ev.model}[/dim]" if ev.model else ""
                latency = f"[yellow]{ev.latency_ms / 1000:.1f}s[/yellow]"
                parts = [p for p in [block_info, model_info, latency] if p]
                lines.append(f"  {' | '.join(parts)}")
                if ev.details.get("intent"):
                    lines.append(f"  intent: [green]{ev.details['intent']}[/green] | resources: {ev.details.get('resources', '')}")
                drafts = ev.details.get("query_drafts", {})
                if drafts:
                    for coll, queries in drafts.items():
                        lines.append(f"    [dim]{coll}:[/dim] {queries}")

        if retrieval_events:
            lines.append("")
            lines.append("[bold blue]PHASE 2: Retrieval[/bold blue]")
            for ev in retrieval_events:
                coll = ev.details.get("collection", "?")
                count = ev.details.get("result_count", 0)
                latency = f"[yellow]{ev.latency_ms / 1000:.1f}s[/yellow]"
                if ev.phase == "re_retrieval":
                    reason = ev.details.get("reason", "insufficient sources")
                    lines.append(f"  [bold yellow]Re-retrieval[/bold yellow] ({reason})")
                elif ev.phase == "quality_reretrieval":
                    lines.append(f"  [bold magenta]Quality Re-retrieval[/bold magenta] (answer quality)")
                lines.append(f"  {coll}: {count} results | {latency}")
                query_text = ev.details.get("query", "")
                if query_text:
                    lines.append(f"    [dim]q: {query_text}[/dim]")

            total_results = sum(
                e.details.get("result_count", 0)
                for e in retrieval_events
                if e.phase == "retrieval"
            )
            lines.append(f"  total: {total_results} results")

        if answering_events:
            lines.append("")
            lines.append("[bold magenta]PHASE 3: Answering[/bold magenta]")
            for ev in answering_events:
                block_info = f"[dim]{ev.block}[/dim]" if ev.block else ""
                model_info = f"[dim]{ev.model}[/dim]" if ev.model else ""
                ctx_chars = ev.details.get("context_chars", 0)
                latency = f"[yellow]{ev.latency_ms / 1000:.1f}s[/yellow]"
                parts = [p for p in [block_info, model_info, f"context: {ctx_chars} chars", latency] if p]
                lines.append(f"  {' | '.join(parts)}")

        if validation_events:
            lines.append("")
            lines.append("[bold green]PHASE 4: Validation[/bold green]")
            for ev in validation_events:
                passed = ev.details.get("passes", False)
                checks = ev.details.get("checks", {})
                check_str = ", ".join(f"{k}{'✓' if v else '✗'}" for k, v in checks.items())
                status = "[green]PASS[/green]" if passed else "[red]FAIL[/red]"
                lines.append(f"  sanitizer: {status} | [{check_str}]")

        total_sec = self.total_latency_ms / 1000
        lines.append("")
        lines.append(f"[bold]TOTAL: {total_sec:.1f}s[/bold]")

        console.print(Panel(
            "\n".join(lines),
            title="[bold]Pipeline Trace[/bold]",
            border_style="dim yellow",
            padding=(0, 1),
        ))


class _PhaseContext:
    """Context manager for timing a pipeline phase."""

    def __init__(
        self,
        tracer: PipelineTracer,
        name: str,
        block: str | None,
        model: str | None,
        details: dict[str, Any],
    ) -> None:
        self._tracer = tracer
        self._name = name
        self._block = block
        self._model = model
        self._details = details
        self._start: float = 0

    def __enter__(self) -> "_PhaseContext":
        self._start = time.perf_counter()
        self._tracer._emit_phase_start(self._name, self._block, self._model, self._details)
        return self

    def __exit__(self, *args: Any) -> None:
        latency_ms = (time.perf_counter() - self._start) * 1000
        self._tracer._record(
            self._name,
            latency_ms,
            block=self._block,
            model=self._model,
            details=self._details,
        )

    def update_details(self, **kwargs: Any) -> None:
        """Add or update detail fields before the phase ends."""
        self._details.update(kwargs)
