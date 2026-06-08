"""Evaluation harness: runs the full retrieval → (optionally) generation pipeline
over a set of test queries and reports how well the system performs.

HOW TO READ THIS FILE (for newcomers to RAG evaluation):
---------------------------------------------------------
A RAG system has THREE layers that can each fail independently:

  Layer 1 — RETRIEVAL:      Did the system find the right documents?
  Layer 2 — CONTEXT BUILD:  Did those documents survive the distance filter?
  Layer 3 — GENERATION:     Did the LLM produce a faithful, correct answer?

We test all three here. Most metrics come from Layer 1 (Precision, MRR, Hit Rate).
`context_coverage` tests Layer 2. The LLM judge tests Layer 3.

If P@10 > 0 but context_coverage = 0, the system found the right docs but they
were all filtered out by the distance threshold — the LLM never saw them.
That is a Layer 2 failure, invisible if you only look at retrieval metrics.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from src.evaluator import retrieval_metrics as rm
from src.evaluator.golden import compare_to_golden
from src.evaluator.judge import LLMJudge
from src.evaluator.latency import LatencyReport, time_call
from src.evaluator.report import print_report, save_report
from src.retriever.context import context_included_ids
from src.config import settings

console = Console()

PASS_THRESHOLD = 0.5
WARN_THRESHOLD = 0.3


@dataclass
class EvalReport:
    """Holds all results from a single evaluation run."""
    results: list[dict] = field(default_factory=list)
    latency_retrieval: LatencyReport = field(default_factory=lambda: LatencyReport("retrieval"))
    latency_generation: LatencyReport = field(default_factory=lambda: LatencyReport("generation"))

    def to_dict(self) -> dict:
        return {
            "results": self.results,
            "latency": {
                "retrieval": self.latency_retrieval.summary(),
                "generation": self.latency_generation.summary(),
            },
        }


def run_eval(
    queries: list[dict],
    service,
    *,
    run_judge: bool = True,
    k_values: list[int] | None = None,
    save: bool = True,
) -> EvalReport:
    """Run the full evaluation loop over a list of test queries.

    Args:
        queries:    List of query dicts loaded from the fixtures JSON.
                    Required keys: "id", "query"
                    Optional keys: "expected_source_db", "expected_year",
                                   "relevant_kayit_nos", "golden_answer"
        service:    A RAGService instance (wires retriever + generator).
        run_judge:  If True, score answer quality with LLM-as-judge.
                    Also runs golden answer regression if golden_answer is set.
                    Skip with --quick for faster retrieval-only runs.
        k_values:   Which k cutoffs to evaluate (default: 5, 10, 20).
        save:       Write the report JSON to artifacts/ after the run.
    """
    if k_values is None:
        k_values = [1, 3, 5]

    judge = LLMJudge() if run_judge else None
    report = EvalReport()

    console.print(Panel(
        f"[bold cyan]RAG Evaluation Run[/bold cyan]\n"
        f"Queries: [yellow]{len(queries)}[/yellow]  |  "
        f"LLM Judge: [yellow]{'ON' if run_judge else 'OFF'}[/yellow]  |  "
        f"k cutoffs: [yellow]{k_values}[/yellow]",
        title="[bold]Starting[/bold]",
        border_style="cyan",
    ))

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("Evaluating queries…", total=len(queries))

        for item in queries:
            qid = item.get("id", "?")
            query = item["query"]
            expected_db = item.get("expected_source_db")
            expected_year = item.get("expected_year")
            # TODO(span-matcher): harness still scores via relevant_kayit_nos.
            # Migrate to src.evaluator.span_metrics once retrieval returns span info per chunk.
            relevant_ids = set(item.get("relevant_kayit_nos") or [])
            golden_answer = item.get("golden_answer")

            progress.update(task, description=f"[bold]{qid}[/bold]: {query[:55]}…")

            # ----------------------------------------------------------------
            # LAYER 1: RETRIEVAL
            # ----------------------------------------------------------------
            elapsed_r, results = time_call(service.retrieve, query)
            report.latency_retrieval.record(elapsed_r)

            retrieved_ids = [
                m.get("document_id")
                for m in results.get("metadatas", [[]])[0]
            ]
            parsed_dates = results.get("parsed_dates", {})
            is_minutes = results.get("is_minutes", False)

            metrics: dict = {}
            for k in k_values:
                metrics[f"precision_{k}"] = rm.precision_at_k(retrieved_ids, relevant_ids, k)
                metrics[f"recall_{k}"] = rm.recall_at_k(retrieved_ids, relevant_ids, k)
                metrics[f"hit_rate_{k}"] = rm.hit_rate_at_k(retrieved_ids, relevant_ids, k)
            metrics["mrr"] = rm.mrr(retrieved_ids, relevant_ids)
            metrics["routing_acc"] = rm.source_routing_accuracy(is_minutes, expected_db or "both")
            metrics["date_acc"] = rm.date_filter_accuracy(parsed_dates, expected_year)

            # ----------------------------------------------------------------
            # LAYER 2: CONTEXT COVERAGE
            # Did the retrieved relevant docs survive the distance filter and
            # actually make it into the text the LLM will read?
            # ----------------------------------------------------------------
            included = context_included_ids(results, settings.DISTANCE_THRESHOLD)
            metrics["context_coverage"] = rm.context_coverage(relevant_ids, included)

            # ----------------------------------------------------------------
            # LAYER 3: GENERATION + LLM JUDGE (optional, slow)
            # ----------------------------------------------------------------
            judge_scores: Optional[dict] = None
            answer = None
            gen_ms: Optional[float] = None

            if run_judge:
                context = service.build_context(results)
                elapsed_g, (_, answer) = time_call(service.ask_from_results, query, results)
                report.latency_generation.record(elapsed_g)
                gen_ms = elapsed_g * 1000

                if judge and context.strip() and answer:
                    judge_scores = judge.score(query, context, answer)

                    # Golden answer regression: compare generated answer against
                    # the stored reference answer. Flags a regression if relevance
                    # or faithfulness drops more than 1 point vs. the baseline.
                    if golden_answer:
                        golden_result = compare_to_golden(
                            query, context, answer, golden_answer, judge
                        )
                        judge_scores["regression"] = golden_result["regression"]

            report.results.append({
                "id": qid,
                "query": query,
                "metrics": metrics,
                "judge": judge_scores or {},
                "answer": answer,
                "gen_ms": gen_ms,
            })

            progress.advance(task)

    _print_summary(report)

    if save:
        save_report(report.to_dict())

    print_report(report.to_dict())
    return report


def _print_summary(report: EvalReport) -> None:
    """Print a brief PASS/WARN/FAIL summary panel after all queries are done."""
    pass_count = warn_count = fail_count = skip_count = 0

    for entry in report.results:
        m = entry.get("metrics", {})
        p10 = m.get("precision_10")
        mrr = m.get("mrr")

        if not entry.get("metrics", {}).get("hit_rate_10") and p10 == 0.0 and mrr == 0.0:
            skip_count += 1
            continue

        score = max(p10 or 0, mrr or 0)
        if score >= PASS_THRESHOLD:
            pass_count += 1
        elif score >= WARN_THRESHOLD:
            warn_count += 1
        else:
            fail_count += 1

    parts = []
    if pass_count:
        parts.append(f"[green]✓ {pass_count} PASS[/green]")
    if warn_count:
        parts.append(f"[yellow]~ {warn_count} WARN[/yellow]")
    if fail_count:
        parts.append(f"[red]✗ {fail_count} FAIL[/red]")
    if skip_count:
        parts.append(f"[dim]{skip_count} no-ground-truth[/dim]")

    lat_r = report.latency_retrieval.summary()
    lat_g = report.latency_generation.summary()
    gen_line = ""
    if lat_g.get("count", 0) > 0:
        gen_line = f"\nGeneration p50: [cyan]{lat_g.get('p50_ms', 0):.0f}ms[/cyan]"

    console.print(Panel(
        "  ".join(parts) + f"\n\n"
        f"Retrieval p50: [cyan]{lat_r.get('p50_ms', 0):.0f}ms[/cyan]  "
        f"p95: [cyan]{lat_r.get('p95_ms', 0):.0f}ms[/cyan]"
        + gen_line,
        title="[bold]Evaluation Summary[/bold]",
        border_style="green" if not fail_count else "yellow",
    ))
