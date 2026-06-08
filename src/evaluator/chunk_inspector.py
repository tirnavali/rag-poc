"""Chunk quality inspector — sanity-check the data chunking strategy.

WHAT IS CHUNKING?
-----------------
When we ingest documents into the archive, we split them into smaller pieces
called "chunks" before storing them in ChromaDB. This is necessary because:
  1. LLMs have a context window limit (can't pass a 50-page document)
  2. Smaller, focused chunks improve retrieval precision

HOW TO RUN:
  python -m src.evaluator.chunk_inspector

WHAT TO LOOK FOR:
  - Average chunk size should be close to the configured chunk_size (2000 for
    press, 1500 for minutes). If it's much smaller, your documents are too short.
  - "Too short" chunks (< 100 chars) are noise — they waste embedding space.
  - "Too long" chunks (> configured max + 20%) mean the splitter is not working.
  - The histogram gives a visual feel for the distribution.
"""
from __future__ import annotations

import statistics
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

from src.common.chroma import open_collection
from src.config import settings

console = Console()

# Anything shorter than this is probably a header/label artifact, not real content.
MIN_USEFUL_CHUNK = 100

# Allow up to 20% over the configured chunk_size before flagging as "too long".
OVERSIZE_SLACK = 1.2


def _load_chroma_samples(chroma_path: Path, collection_name: str, max_samples: int = 200) -> Optional[list[str]]:
    """Load up to max_samples document chunks from a ChromaDB collection.

    Uses the shared open_collection helper so ChromaDB settings match the rest
    of the codebase and the singleton client doesn't raise a conflict.
    Returns None if the collection doesn't exist (e.g. minutes not yet indexed).
    """
    try:
        _client, col = open_collection(chroma_path, collection_name)

        total = col.count()
        if total == 0:
            return []

        # Sample a subset — fetching all can be slow for large collections
        sample_n = min(max_samples, total)
        result = col.get(limit=sample_n, include=["documents"])
        return result.get("documents", []) or []

    except Exception as exc:
        console.print(f"[yellow]  Warning: could not load {collection_name}: {exc}[/yellow]")
        return None


def _analyze_chunks(docs: list[str], name: str, configured_chunk_size: int) -> dict:
    """Compute size statistics for a list of document chunks."""
    lengths = [len(d) for d in docs if d]

    if not lengths:
        return {"name": name, "count": 0, "error": "no documents found"}

    avg = statistics.mean(lengths)
    median = statistics.median(lengths)
    min_len = min(lengths)
    max_len = max(lengths)

    too_short = sum(1 for l in lengths if l < MIN_USEFUL_CHUNK)
    too_long = sum(1 for l in lengths if l > configured_chunk_size * OVERSIZE_SLACK)

    return {
        "name": name,
        "count": len(lengths),
        "avg": avg,
        "median": median,
        "min": min_len,
        "max": max_len,
        "too_short": too_short,
        "too_long": too_long,
        "configured_size": configured_chunk_size,
    }


def _print_histogram(lengths: list[int], configured_size: int, bins: int = 10) -> None:
    """Print a simple ASCII bar chart of chunk size distribution."""
    if not lengths:
        return

    min_l, max_l = min(lengths), max(lengths)
    if min_l == max_l:
        console.print(f"  All chunks exactly {min_l} chars.")
        return

    bin_width = (max_l - min_l) / bins
    counts = [0] * bins

    for l in lengths:
        idx = min(int((l - min_l) / bin_width), bins - 1)
        counts[idx] = counts[idx] + 1

    max_count = max(counts) or 1
    bar_max = 30  # max bar width in chars

    console.print(f"\n  [bold]Chunk size distribution[/bold] (configured target: {configured_size})")
    for i, count in enumerate(counts):
        bucket_start = int(min_l + i * bin_width)
        bucket_end = int(min_l + (i + 1) * bin_width)
        bar_len = int(count / max_count * bar_max)
        bar = "█" * bar_len
        # Highlight the target bucket
        marker = " ← target" if bucket_start <= configured_size <= bucket_end else ""
        console.print(f"  {bucket_start:5d}–{bucket_end:<5d} [{bar:<30}] {count}{marker}")


def _print_collection_report(stats: dict, docs: list[str]) -> None:
    """Display stats for one collection in a Rich panel."""
    name = stats["name"]

    if stats.get("error"):
        console.print(Panel(f"[red]{stats['error']}[/red]", title=name))
        return

    count = stats["count"]
    avg = stats["avg"]
    configured = stats["configured_size"]
    too_short = stats["too_short"]
    too_long = stats["too_long"]

    # Health assessment
    if too_short == 0 and too_long == 0 and abs(avg - configured) < configured * 0.4:
        health = "[green]✓ HEALTHY[/green]"
    elif too_short > count * 0.1 or too_long > count * 0.1:
        health = "[red]✗ NEEDS ATTENTION[/red]"
    else:
        health = "[yellow]~ ACCEPTABLE[/yellow]"

    lines = [
        f"{health}",
        f"  Sampled chunks:   {count}",
        f"  Average length:   {avg:.0f} chars  (target ≈ {configured})",
        f"  Median length:    {stats['median']:.0f} chars",
        f"  Min / Max:        {stats['min']} / {stats['max']} chars",
        f"  Too short (<{MIN_USEFUL_CHUNK}):  {too_short} chunks"
        + ("[red] ← check ingestion![/red]" if too_short > 5 else ""),
        f"  Too long (>{configured * OVERSIZE_SLACK:.0f}):  {too_long} chunks"
        + ("[yellow] ← splitter may need tuning[/yellow]" if too_long > 5 else ""),
    ]

    console.print(Panel("\n".join(lines), title=f"[bold cyan]{name}[/bold cyan]"))

    lengths = [len(d) for d in docs if d]
    _print_histogram(lengths, configured)
    console.print()


def run_inspection(max_samples: int = 200) -> dict:
    """Inspect chunk quality for both press clips and parliament minutes.

    Returns a dict with stats for each collection.
    """
    console.print(Panel(
        "[bold cyan]Chunk Quality Inspector[/bold cyan]\n"
        f"Sampling up to [yellow]{max_samples}[/yellow] chunks per collection\n"
        f"[dim]Min useful: {MIN_USEFUL_CHUNK} chars | "
        f"Press target: {settings.PRESS_CHUNK_SIZE} chars | "
        f"Minutes target: {settings.MINUTES_CHUNK_SIZE} chars[/dim]",
        title="[bold]Chunk Inspector[/bold]",
        border_style="cyan",
    ))

    all_stats = {}

    # --- Press clips ---
    console.print("[bold]Checking press clips (gazete arşivi)…[/bold]")
    press_docs = _load_chroma_samples(
        settings.PRESS_CHROMA, settings.PRESS_COLLECTION, max_samples
    )
    if press_docs is not None:
        press_stats = _analyze_chunks(press_docs, "Press Clips (gazete_arsivi)", settings.PRESS_CHUNK_SIZE)
        _print_collection_report(press_stats, press_docs)
        all_stats["press"] = press_stats
    else:
        console.print("[yellow]Press clips collection not available — skipping.[/yellow]\n")

    # --- Parliament minutes ---
    console.print("[bold]Checking parliament minutes (TBMM tutanakları)…[/bold]")
    minutes_docs = _load_chroma_samples(
        settings.MINUTES_CHROMA, settings.MINUTES_COLLECTION, max_samples
    )
    if minutes_docs is not None:
        minutes_stats = _analyze_chunks(minutes_docs, "Parliament Minutes (tbmm_minutes)", settings.MINUTES_CHUNK_SIZE)
        _print_collection_report(minutes_stats, minutes_docs)
        all_stats["minutes"] = minutes_stats
    else:
        console.print("[yellow]Minutes collection not available — run ingestion first.[/yellow]\n")

    return all_stats


if __name__ == "__main__":
    run_inspection()
