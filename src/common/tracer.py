"""Generalized pipeline tracer — works for both standard and agent modes.

Usage:
    tracer = PipelineTracer()
    with tracer.phase("filter_extraction", model="qwen2.5:3b"):
        extracted = filter_extractor.extract(query)
        tracer.update_details(
            hints_found=True,
            refined_query=extracted.refined_query,
            filters=_summarize_filters(extracted.filters),
        )
    tracer.print_trace(console)
"""
from __future__ import annotations

import time
import uuid
from datetime import datetime
from typing import Any, Optional

from rich.console import Console
from rich.panel import Panel


class PipelineTraceEvent:
    """Single trace event from the pipeline."""

    def __init__(
        self,
        trace_id: str,
        phase: str,
        latency_ms: float,
        *,
        block: str | None = None,
        model: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.trace_id = trace_id
        self.phase = phase
        self.block = block
        self.model = model
        self.latency_ms = round(latency_ms, 1)
        self.details = details or {}


class PipelineTracer:
    """Collects and prints trace events for RAG pipeline phases.

    Supports both standard mode (filter_extraction → retrieval → context → generation)
    and agent mode (planning → retrieval → re_retrieval → answering → validation).

    Usage:
        tracer = PipelineTracer()
        with tracer.phase("retrieval", model="nomic-embed-text-v2-moe"):
            results = retriever.retrieve(query)
            tracer.update_details(result_count=len(results["documents"][0]))
        tracer.print_trace(console)
    """

    def __init__(
        self,
        trace_id: str | None = None,
        on_phase: Optional[Any] = None,
    ) -> None:
        self.trace_id = trace_id or uuid.uuid4().hex[:12]
        self.events: list[PipelineTraceEvent] = []
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
    ) -> PipelineTraceEvent:
        event = PipelineTraceEvent(
            trace_id=self.trace_id,
            phase=phase,
            block=block,
            model=model,
            latency_ms=latency_ms,
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

        filter_events = [e for e in self.events if e.phase == "filter_extraction"]
        planning_events = [e for e in self.events if e.phase == "planning"]
        retrieval_events = [e for e in self.events if e.phase in ("retrieval", "re_retrieval")]
        context_events = [e for e in self.events if e.phase == "context_building"]
        answering_events = [e for e in self.events if e.phase in ("answering", "generation")]
        validation_events = [e for e in self.events if e.phase == "validation"]

        phase_num = 1

        if filter_events:
            lines.append(f"[bold cyan]PHASE {phase_num}: Filter Extraction[/bold cyan]")
            for ev in filter_events:
                if ev.details.get("original_query"):
                    lines.append(f"  query: [green]{ev.details['original_query']}[/green]")
                model_info = f"[dim]{ev.model}[/dim]" if ev.model else ""
                latency = f"[yellow]{ev.latency_ms / 1000:.1f}s[/yellow]"
                hints = ev.details.get("hints_found", False)
                lines.append(f"  hints_found: {'True' if hints else 'False'} | {model_info} | {latency}")
                if ev.details.get("refined_query"):
                    lines.append(f"  refined_query: [green]{ev.details['refined_query']}[/green]")
                removed = ev.details.get("removed_words")
                filters = ev.details.get("filters")
                if removed:
                    removed_str = ", ".join('"' + w + '"' for w in removed)
                    lines.append(f"  removed: {removed_str} → filters: {filters}")
                elif filters:
                    lines.append(f"  filters: {filters}")
                fallback = ev.details.get("fallback_chain")
                if fallback:
                    lines.append(f"  fallback_chain: {' → '.join(str(f) for f in fallback)}")
            phase_num += 1

        if planning_events:
            lines.append(f"[bold cyan]PHASE {phase_num}: Planning[/bold cyan]")
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
            phase_num += 1

        if retrieval_events:
            lines.append(f"[bold blue]PHASE {phase_num}: Retrieval[/bold blue]")
            for ev in retrieval_events:
                coll = ev.details.get("collection", "?")
                count = ev.details.get("result_count", 0)
                latency = f"[yellow]{ev.latency_ms / 1000:.1f}s[/yellow]"
                fallback = ev.details.get("fallback_level", "full")
                where = ev.details.get("where_filter_summary")
                if ev.phase == "re_retrieval":
                    reason = ev.details.get("reason", "insufficient sources")
                    lines.append(f"  [bold yellow]Re-retrieval[/bold yellow] ({reason})")
                parts = [f"{coll}: {count} results", f"fallback={fallback}"]
                if where:
                    parts.append(f"where={where}")
                parts.append(latency)
                lines.append(f"  {' | '.join(parts)}")

            total_results = sum(
                e.details.get("result_count", 0)
                for e in retrieval_events
                if e.phase == "retrieval"
            )
            lines.append(f"  total: {total_results} results")
            phase_num += 1

        if context_events:
            lines.append(f"[bold green]PHASE {phase_num}: Context Building[/bold green]")
            for ev in context_events:
                total_chunks = ev.details.get("total_chunks", 0)
                kept_chunks = ev.details.get("kept_chunks", 0)
                ctx_chars = ev.details.get("context_chars", 0)
                threshold = ev.details.get("distance_threshold", "?")
                lines.append(f"  chunks: {total_chunks} → {kept_chunks} (threshold={threshold}) | context: {ctx_chars} chars")
            phase_num += 1

        if answering_events:
            lines.append(f"[bold magenta]PHASE {phase_num}: Generation[/bold magenta]")
            for ev in answering_events:
                block_info = f"[dim]{ev.block}[/dim]" if ev.block else ""
                model_info = f"[dim]{ev.model}[/dim]" if ev.model else ""
                ctx_chars = ev.details.get("context_chars", 0)
                latency = f"[yellow]{ev.latency_ms / 1000:.1f}s[/yellow]"
                parts = [p for p in [block_info, model_info, f"context: {ctx_chars} chars", latency] if p]
                lines.append(f"  {' | '.join(parts)}")
            phase_num += 1

        if validation_events:
            lines.append(f"[bold green]PHASE {phase_num}: Validation[/bold green]")
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
        # Records latency regardless of success/failure; exceptions propagate normally (no return True).
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
