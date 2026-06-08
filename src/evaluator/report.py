"""Format and persist evaluation reports.

The Rich table uses color coding so your team can spot problems at a glance:
  Green  (≥ 0.5): working well
  Yellow (0.3–0.5): acceptable, worth investigating
  Red    (< 0.3):  needs attention
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich import box
from rich.panel import Panel

console = Console()

# Thresholds matching harness.py — change in one place if you update them.
PASS = 0.5
WARN = 0.3


def _color(value: float | None) -> str:
    """Return a Rich color tag based on the metric value."""
    if value is None:
        return "dim"
    if value >= PASS:
        return "green"
    if value >= WARN:
        return "yellow"
    return "red"


def _fmt(value: float | None, pct: bool = False) -> str:
    """Format a metric value for display."""
    if value is None:
        return "[dim]—[/dim]"
    if pct:
        return f"[{_color(value)}]{value:.0%}[/{_color(value)}]"
    return f"[{_color(value)}]{value:.2f}[/{_color(value)}]"


def print_report(report: dict) -> None:
    """Print a color-coded Rich table with per-query metrics and an aggregate row."""
    results = report.get("results", [])
    if not results:
        console.print("[yellow]No results to display.[/yellow]")
        return

    table = Table(
        title="Evaluation Results",
        box=box.DOUBLE_EDGE,
        header_style="bold cyan",
    )

    table.add_column("Query ID", style="dim")
    table.add_column("P@10")
    table.add_column("MRR")
    table.add_column("Hit@10")
    # Context Hit = Layer 2: did retrieved docs survive the distance filter?
    table.add_column("Ctx Hit", header_style="bold magenta")
    table.add_column("Routing")
    table.add_column("Date OK")
    table.add_column("Faithful")
    table.add_column("Relevance")
    # Regression = Layer 3: did answer regress vs. stored golden answer?
    table.add_column("Regress", header_style="bold yellow")
    table.add_column("Gen ms", style="dim")

    totals: dict[str, list[float]] = {
        "p10": [], "mrr": [], "hit10": [], "ctx": [], "routing": [], "date": [],
        "faith": [], "relev": [],
    }

    for entry in results:
        m = entry.get("metrics", {})
        j = entry.get("judge", {})

        p10 = m.get("precision_10")
        mrr_val = m.get("mrr")
        hit10 = m.get("hit_rate_10")
        ctx = m.get("context_coverage")
        routing = m.get("routing_acc")
        date_acc = m.get("date_acc")
        faith = j.get("faithfulness") if j else None
        relev = j.get("relevance") if j else None
        regression = j.get("regression") if j else None
        gen_ms = entry.get("gen_ms")

        for key, val in [("p10", p10), ("mrr", mrr_val), ("hit10", hit10),
                         ("ctx", ctx), ("routing", routing), ("date", date_acc),
                         ("faith", faith), ("relev", relev)]:
            if val is not None:
                totals[key].append(float(val))

        # Regression column: ✗ (red) = regression detected, ✓ (green) = ok, — = not tested
        if regression is True:
            reg_str = "[red]✗ YES[/red]"
        elif regression is False:
            reg_str = "[green]✓ OK[/green]"
        else:
            reg_str = "[dim]—[/dim]"

        # Context Hit: show — when there's no ground truth (relevant_ids empty)
        ctx_str = _fmt(ctx) if (m.get("precision_10") is not None and
                                 set(entry.get("query", "")) and
                                 ctx is not None and
                                 (p10 or 0) + (mrr_val or 0) > 0) else _fmt(ctx)

        gen_str = f"{gen_ms:.0f}" if gen_ms is not None else "[dim]—[/dim]"
        table.add_row(
            entry.get("id", "?"),
            _fmt(p10),
            _fmt(mrr_val),
            _fmt(hit10),
            _fmt(ctx),
            _fmt(routing, pct=True),
            _fmt(date_acc, pct=True),
            _fmt(faith),
            _fmt(relev),
            reg_str,
            gen_str,
        )

    def avg(lst: list[float]) -> str:
        if not lst:
            return "[dim]—[/dim]"
        return _fmt(sum(lst) / len(lst))

    def avg_pct(lst: list[float]) -> str:
        if not lst:
            return "[dim]—[/dim]"
        return _fmt(sum(lst) / len(lst), pct=True)

    table.add_section()
    table.add_row(
        "[bold]AVG[/bold]",
        avg(totals["p10"]),
        avg(totals["mrr"]),
        avg(totals["hit10"]),
        avg(totals["ctx"]),
        avg_pct(totals["routing"]),
        avg_pct(totals["date"]),
        avg(totals["faith"]),
        avg(totals["relev"]),
        "[dim]—[/dim]",
        "[dim]—[/dim]",
    )

    console.print(table)

    # Latency summary panel
    lat = report.get("latency", {})
    ret = lat.get("retrieval", {})
    gen = lat.get("generation", {})

    lat_lines = [f"Retrieval — mean: {ret.get('mean_ms', 0):.0f}ms  "
                 f"p50: {ret.get('p50_ms', 0):.0f}ms  "
                 f"p95: {ret.get('p95_ms', 0):.0f}ms"]
    if gen.get("count", 0) > 0:
        lat_lines.append(f"Generation — mean: {gen.get('mean_ms', 0):.0f}ms  "
                         f"p50: {gen.get('p50_ms', 0):.0f}ms  "
                         f"p95: {gen.get('p95_ms', 0):.0f}ms")

    console.print(Panel("\n".join(lat_lines), title="[bold]Latency[/bold]", border_style="dim"))


def save_report(report: dict, output_dir: Path = Path("artifacts")) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = output_dir / f"eval_report_{ts}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    console.print(f"[dim]Report saved → {out_path}[/dim]")
    return out_path
