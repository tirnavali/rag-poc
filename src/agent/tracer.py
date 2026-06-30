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
        on_phase_end: Optional[Any] = None,
    ) -> None:
        self.trace_id = trace_id or uuid.uuid4().hex[:12]
        self.events: list[AgentTraceEvent] = []
        self._on_phase = on_phase
        self._on_phase_end = on_phase_end
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
        # Notify a listener that a phase COMPLETED, with its filled-in details
        # (used for live per-stage UI streaming). Never break the pipeline.
        if self._on_phase_end is not None:
            try:
                self._on_phase_end(event)
            except Exception:
                pass
        return event

    def print_trace(self, console: Console | None = None) -> None:
        """Print the full pipeline trace to console."""
        console = console or Console()
        lines: list[str] = []

        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        lines.append(f"[dim]Trace:[/dim] {self.trace_id} | [dim]{ts}[/dim]")
        lines.append("")

        def _ev(phase: str):
            return next((e for e in self.events if e.phase == phase), None)

        def _lat(ev) -> str:
            return f"[yellow]{ev.latency_ms / 1000:.1f}s[/yellow]"

        # ── Intent / scope ────────────────────────────────────────────────
        cls = _ev("classification")
        if cls:
            scope = cls.details.get("scope", "?")
            sel = cls.details.get("selected_collections", [])
            lines.append(f"[bold]Intent:[/bold] scope=[green]{scope}[/green] | tool/db: {sel or '—'} | {_lat(cls)}")

        # ── Clarification ─────────────────────────────────────────────────
        probe = _ev("probe")
        clar = _ev("clarification")
        if probe:
            d = probe.details
            lines.append(f"[bold]Probe:[/bold] {d.get('hits', 0)} kayıt | {d.get('years', 0)} yıl, {d.get('topics', 0)} konu")
        if clar:
            d = clar.details
            mode = "soruldu" if d.get("asked") else ("oto" if d.get("auto_applied") else "atlandı")
            lines.append(f"[bold]Clarification:[/bold] {mode} | kısıt: {d.get('constraints') or '—'}")

        # ── Planning ──────────────────────────────────────────────────────
        plan = _ev("planning")
        if plan:
            lines.append("")
            lines.append("[bold cyan]Planning[/bold cyan]")
            lines.append(f"  intent: [green]{plan.details.get('intent', '?')}[/green] | {_lat(plan)}")
            cols = plan.details.get("collections", [])
            lines.append(f"  koleksiyonlar: {', '.join(cols) if cols else '—'}")
            for coll, queries in (plan.details.get("drafts", {}) or {}).items():
                lines.append(f"    [dim]{coll}:[/dim] {queries}")

        # ── Policy / Allocation (stage-2 gates) ───────────────────────────
        pol = _ev("policy")
        alloc = _ev("allocation")
        if pol:
            on = pol.details.get("enabled")
            lines.append(f"[bold]Policy:[/bold] {'açık' if on else 'kapalı'} | allowed: {pol.details.get('allowed', [])}")
        if alloc and not alloc.details.get("enabled", True):
            lines.append("[bold]Allocation:[/bold] kapalı (düz fetch_k tek havuz)")

        # ── Retrieval (orchestrator: per_collection) ──────────────────────
        retr = _ev("retrieval")
        if retr:
            lines.append("")
            lines.append("[bold blue]Retrieval[/bold blue]")
            per = retr.details.get("per_collection", {})
            total = 0
            if per:
                for name, info in per.items():
                    returned = info.get("returned", 0)
                    fetched = info.get("fetched", 0)
                    total += returned
                    lines.append(f"  {name}: {returned}/{fetched} (returned/fetched)")
            else:
                # legacy shape fallback
                total = retr.details.get("result_count", 0)
            lines.append(f"  total: {total} results | {_lat(retr)}")

        exp = _ev("expansion")
        if exp:
            lines.append(f"  [yellow]↻ re-query:[/yellow] expanded={exp.details.get('expanded')} → {exp.details.get('post_count', '?')} chunk")

        # ── Assembly / Judge ──────────────────────────────────────────────
        asm = _ev("assembly")
        if asm:
            lines.append(f"[bold]Assembly:[/bold] {asm.details.get('primary_count', 0)} chunk | kapsam {asm.details.get('collection_coverage', 0)}")
        jdg = _ev("judge")
        if jdg:
            d = jdg.details
            lines.append(f"[bold]Judge:[/bold] {d.get('action', '?')} ([dim]{d.get('judge_type', '?')}[/dim], conf={d.get('confidence', '?')})")

        # ── Answering ─────────────────────────────────────────────────────
        ans = _ev("answering")
        if ans:
            lines.append("")
            lines.append("[bold magenta]Answering[/bold magenta]")
            lines.append(f"  context: {ans.details.get('context_chars', 0)} chars | {_lat(ans)}")

        # ── Validation ────────────────────────────────────────────────────
        val = _ev("validation")
        if val:
            lines.append("")
            passed = val.details.get("passes", False)
            status = "[green]PASS[/green]" if passed else "[red]FAIL[/red]"
            lines.append(f"[bold green]Validation[/bold green] {status}")

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
