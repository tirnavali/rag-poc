"""
A/B Benchmark Runner — Retrieval experimentleri için CLI.

Bir YAML konfigürasyon dosyası alır, birden fazla koleksiyonu aynı fixture
üzerinde değerlendirir ve sonuçları renkli tablo + JSON olarak verir.

Kullanım:
    python -m scripts.run_benchmark --config experiments/jina_v3_vs_nomic.yaml
    python -m scripts.run_benchmark --config experiments/jina_v3_vs_nomic.yaml --output artifacts/bench_result.json

YAML Formatı:
    name: "Deney başlığı"
    collections:
      - tbmm_minutes_docling_jina_v3
      - tbmm_minutes
    fixture: tests/fixtures/eval_queries_docling_d20.json
    top_k: [1, 3, 5, 10]
    reranker: false          # true ise cross-encoder kullanır
    fetch_k: 100             # ChromaDB'den çekilecek aday sayısı
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import yaml
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

from src.config.collections import COLLECTIONS, get_spec
from src.evaluator.benchmark import RetrievalBenchmark
from src.retriever.reranker import CrossEncoderReranker

console = Console()

PASS = 0.5
WARN = 0.3


def _color(value: float | None) -> str:
    if value is None:
        return "dim"
    if value >= PASS:
        return "green"
    if value >= WARN:
        return "yellow"
    return "red"


def _fmt(value: float | None, pct: bool = False) -> str:
    if value is None:
        return "[dim]—[/dim]"
    if pct:
        return f"[{_color(value)}]{value:.0%}[/{_color(value)}]"
    return f"[{_color(value)}]{value:.2f}[/{_color(value)}]"


def _fmt_ms(value: float | None) -> str:
    """Format a latency value in ms — lower is better, so no quality-threshold coloring."""
    if value is None:
        return "[dim]—[/dim]"
    if value < 50:
        color = "green"
    elif value < 200:
        color = "yellow"
    else:
        color = "red"
    return f"[{color}]{value:.1f} ms[/{color}]"


# (description, short label, aggregate key) — rendered with _fmt_ms, not _fmt.
_TIMING_ROWS: list[tuple[str, str, str]] = [
    ("Ortalama sorgu süresi", "Latency avg", "latency_ms_avg"),
    ("Medyan sorgu süresi", "Latency p50", "latency_ms_p50"),
    ("p95 sorgu süresi", "Latency p95", "latency_ms_p95"),
    ("Embedding süresi (ort.)", "Embed", "embed_ms_avg"),
    ("ANN arama süresi (ort.)", "ANN", "ann_ms_avg"),
    ("Reranker süresi (ort.)", "Rerank", "rerank_ms_avg"),
]


def _build_metric_rows(k_values: tuple[int, ...]) -> list[tuple[str, str, str]]:
    """Dynamically build metric rows for all requested k values."""
    rows: list[tuple[str, str, str]] = []
    for k in sorted(k_values):
        rows.append((f"İlk {k} sonucun doğruluk oranı", f"P@{k}", f"precision_{k}"))
    for k in sorted(k_values):
        rows.append((f"Doğruları ilk {k}'e getirme oranı", f"R@{k}", f"recall_{k}"))
    for k in sorted(k_values):
        rows.append((f"İlk {k}'de en az bir doğru bulma", f"Hit@{k}", f"hit_rate_{k}"))
    rows += [
        ("Doğruyu en üstte gösterme becerisi", "MRR", "mrr"),
        ("Sıralama kalitesi (Genel başarı)", "NDCG@10", "ndcg_10"),
    ]
    # Token-overlap metrikleri (excerpts matcher'da dolar)
    for k in sorted(k_values):
        rows.append((f"Altın tokenların ilk {k}'de kapsanma oranı", f"TkR@{k}", f"token_recall_{k}"))
        rows.append((f"İlk {k} çekilen tokenlarda alaka oranı", f"TkP@{k}", f"token_precision_{k}"))
        rows.append((f"Token IoU@{k}", f"IoU@{k}", f"token_iou_{k}"))
    return rows


def print_comparison(reports: list[dict], k_values: tuple[int, ...] = (5, 10)) -> None:
    """Print a side-by-side comparison table for all collections."""
    if not reports:
        console.print("[yellow]Hiç rapor yok.[/yellow]")
        return

    table = Table(
        title="Retrieval Benchmark Karşılaştırması",
        box=box.DOUBLE_EDGE,
        header_style="bold cyan",
    )

    table.add_column("Açıklama", style="dim")
    table.add_column("Metrik", style="bold")
    for r in reports:
        name = r["spec"]["name"]
        model = r["spec"]["embed_model"]
        table.add_column(f"{name}\n[dim]{model}[/dim]", justify="center")

    for desc, label, key in _build_metric_rows(k_values):
        row = [desc, label]
        for r in reports:
            val = r.get("aggregate", {}).get(key)
            row.append(_fmt(val))
        table.add_row(*row)

    # Timing rows — separate section, ms formatting (lower = better, no quality threshold)
    table.add_section()
    for desc, label, key in _TIMING_ROWS:
        row = [desc, f"[bold]{label}[/bold]"]
        for r in reports:
            row.append(_fmt_ms(r.get("aggregate", {}).get(key)))
        table.add_row(*row)

    console.print(table)


def run_experiment(config_path: Path) -> list[dict]:
    """Load YAML config and run benchmark for each listed collection."""
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    name = cfg.get("name", "Unnamed Experiment")
    collection_names = cfg["collections"]
    fixture_path = Path(cfg["fixture"])
    k_values = tuple(cfg.get("top_k", [1, 3, 5, 10]))
    use_reranker = cfg.get("reranker", False)
    fetch_k = cfg.get("fetch_k", None)

    if not fixture_path.exists():
        raise FileNotFoundError(f"Fixture bulunamadı: {fixture_path}")

    queries = json.loads(fixture_path.read_text(encoding="utf-8"))
    console.print(
        Panel(
            f"[bold cyan]{name}[/bold cyan]\n"
            f"Fixture: [yellow]{fixture_path}[/yellow] ({len(queries)} sorgu)\n"
            f"Koleksiyonlar: [yellow]{', '.join(collection_names)}[/yellow]\n"
            f"Reranker: [yellow]{'ON' if use_reranker else 'OFF'}[/yellow] | "
            f"k: [yellow]{list(k_values)}[/yellow]",
            title="[bold]Benchmark Başlıyor[/bold]",
            border_style="cyan",
        )
    )

    reranker = CrossEncoderReranker() if use_reranker else None
    reports: list[dict] = []

    for col_name in collection_names:
        spec = get_spec(col_name)
        console.print(f"[dim]→ {col_name} değerlendiriliyor...[/dim]")
        bench = RetrievalBenchmark(spec)
        report = bench.evaluate(
            queries, k_values=k_values, reranker=reranker, fetch_k=fetch_k
        )
        reports.append(report)

    return reports, k_values


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Retrieval benchmark runner (A/B collection comparison)"
    )
    parser.add_argument(
        "--config", "-c", required=True, help="YAML deney konfigürasyonu"
    )
    parser.add_argument(
        "--output", "-o", default=None, help="JSON çıktı dosyası (opsiyonel)"
    )
    args = parser.parse_args()

    reports, k_values = run_experiment(Path(args.config))
    print_comparison(reports, k_values=k_values)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        console.print(f"\n[dim]JSON rapor kaydedildi → {out_path}[/dim]")


if __name__ == "__main__":
    main()
