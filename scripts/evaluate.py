"""RAG Evaluator — checks that retrieval and chunking are working correctly.

This script runs two types of checks:

  1. CHUNK INSPECTION (fast, no LLM needed)
     Samples chunks from ChromaDB and reports size statistics.
     Tells you whether your chunking strategy is producing well-sized pieces.
     Run with: --inspect-chunks

  2. RETRIEVAL + GENERATION EVALUATION (uses test queries)
     Runs a set of Turkish-language test queries through the full pipeline.
     Reports Precision@k, MRR, routing accuracy, and optionally LLM judge scores.

Usage:
  python -m scripts.evaluate                          # full eval with LLM judge
  python -m scripts.evaluate --quick                  # retrieval only, no judge (fast)
  python -m scripts.evaluate --inspect-chunks         # chunk check, then full eval
  python -m scripts.evaluate --inspect-chunks --quick # chunk check only, no judge
  python -m scripts.evaluate --queries path/to/queries.json
"""
import argparse
import json
import logging
import warnings
from pathlib import Path

from rich.console import Console

# Suppress noisy ChromaDB telemetry warnings — they're not errors
logging.getLogger("chromadb").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning, module="chromadb")
from rich.panel import Panel

console = Console()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run RAG evaluation — checks retrieval quality and chunk health.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--queries",
        default="tests/fixtures/eval_queries_tr.json",
        help="Path to JSON query fixture file (default: tests/fixtures/eval_queries_tr.json)",
    )
    parser.add_argument(
        "--no-judge",
        action="store_true",
        help="Skip LLM-as-judge scoring (faster). Use when iterating on retrieval.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Alias for --no-judge. Skip LLM judge for a fast retrieval-only run.",
    )
    parser.add_argument(
        "--inspect-chunks",
        action="store_true",
        help="Run chunk quality inspection before the query evaluation.",
    )
    parser.add_argument(
        "--golden-only",
        action="store_true",
        help="Run only queries that have a golden_answer set. Always enables the judge. "
             "Use this for fast regression checks after changing the model or config.",
    )
    args = parser.parse_args()

    skip_judge = args.no_judge or args.quick
    # --golden-only always forces the judge on (we need generation to compare answers)
    if args.golden_only:
        skip_judge = False

    console.print(Panel(
        "[bold cyan]RAG Archive Evaluator[/bold cyan]\n"
        "Verifies that chunking and retrieval are working correctly.\n\n"
        f"  Queries file:  [yellow]{args.queries}[/yellow]\n"
        f"  LLM Judge:     [yellow]{'OFF (--quick)' if skip_judge else 'ON'}[/yellow]\n"
        f"  Chunk inspect: [yellow]{'YES' if args.inspect_chunks else 'NO'}[/yellow]\n"
        f"  Golden only:   [yellow]{'YES' if args.golden_only else 'NO'}[/yellow]",
        border_style="cyan",
    ))

    # ----------------------------------------------------------------
    # Step 1: Chunk inspection (optional but recommended on first run)
    # ----------------------------------------------------------------
    if args.inspect_chunks:
        from src.evaluator.chunk_inspector import run_inspection
        run_inspection()

    # ----------------------------------------------------------------
    # Step 2: Load test queries
    # ----------------------------------------------------------------
    queries_path = Path(args.queries)
    if not queries_path.exists():
        console.print(f"[red]Query file not found: {queries_path}[/red]")
        console.print("[dim]Run from the project root, e.g.:[/dim]")
        console.print("[dim]  python -m scripts.evaluate[/dim]")
        return

    with open(queries_path, encoding="utf-8") as f:
        queries = json.load(f)

    # --golden-only: keep only queries that have a golden_answer
    if args.golden_only:
        queries = [q for q in queries if q.get("golden_answer")]
        if not queries:
            console.print("[red]No queries with golden_answer found. Add golden answers to the fixture file.[/red]")
            return

    console.print(f"[dim]Loaded {len(queries)} test queries from {queries_path}[/dim]\n")

    # ----------------------------------------------------------------
    # Step 3: Initialize the RAG system and run evaluation
    # ----------------------------------------------------------------
    console.print("[bold]Initializing RAG system…[/bold]")
    from src.generator.service import RAGService
    from src.evaluator.harness import run_eval

    service = RAGService()

    console.print("[bold]Running evaluation…[/bold]\n")
    run_eval(queries, service, run_judge=not skip_judge)


if __name__ == "__main__":
    main()
