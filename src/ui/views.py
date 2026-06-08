"""Rich rendering helpers for the terminal chat UI."""
from __future__ import annotations

from datetime import datetime

from rich import box
from rich.align import Align
from rich.columns import Columns
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

console = Console()

BANNER = """
████████╗██████╗ ███╗   ███╗███╗   ███╗
╚══██╔══╝██╔══██╗████╗ ████║████╗ ████║
   ██║   ██████╔╝██╔████╔██║██╔████╔██║
   ██║   ██╔══██╗██║╚██╔╝██║██║╚██╔╝██║
   ██║   ██████╔╝██║ ╚═╝ ██║██║ ╚═╝ ██║
   ╚═╝   ╚═════╝ ╚═╝     ╚═╝╚═╝     ╚═╝
      Tutanak & Gazete Arşivi  ·  v2.0
"""

COMMANDS = {
    "/cikis": "Uygulamadan çık",
    "/temizle": "Ekranı temizle",
    "/debug": "Debug mesajlarını aç/kapat",
    "/kaynak n": "n. kaynağın tüm veritabanı alanlarını göster",
    "/müfettiş soru": "Derin araştırma modu (Query expansion + 32k context)",
    "/rapor konu": "Derin araştırma + Markdown rapor (artifacts/reports/ altına kaydedilir)",
    "/yardim": "Bu menüyü göster",
}


def print_banner() -> None:
    console.print(Align.center(Text(BANNER, style="bold dark_red")))
    console.print(Align.center(Text(
        "Gazete arşivine soru sorun  ·  /yardim yazın komutlar için",
        style="dim",
    )))
    console.print()


def print_help() -> None:
    items = [
        f"[bold cyan]{cmd}[/bold cyan]  [dim]→[/dim]  {desc}"
        for cmd, desc in COMMANDS.items()
    ]
    console.print(Panel(
        "\n".join(items),
        title="[bold]Komutlar[/bold]",
        border_style="cyan",
        padding=(1, 2),
    ))


def print_user(question: str) -> None:
    ts = datetime.now().strftime("%H:%M")
    console.print()
    console.print(Panel(
        Text(question, style="bold white"),
        title=f"[dim]{ts}[/dim]  [bold green]Siz[/bold green]",
        title_align="right",
        border_style="green",
        padding=(0, 2),
    ))


def print_sources(sources: list[dict] | None, distances: list[float] | None) -> None:
    if not sources:
        return
    source_texts = []
    for i, (meta, dist) in enumerate(zip(sources, distances or []), 1):
        publication = meta.get("publication", "?")
        date = meta.get("date", "?")
        author = meta.get("author", "?")
        title = meta.get("title", "?")
        collection = meta.get("collection", "")
        dist_bar = _dist_bar(dist)

        # Add collection attribution if available
        collection_str = f"[dim][{collection}][/dim]\n" if collection else ""

        source_texts.append(Panel(
            f"{collection_str}"
            f"[bold]{publication}[/bold]\n"
            f"[dim]{date}[/dim]\n"
            f"[cyan]{author}[/cyan]\n"
            f"[italic dim]{title[:55]}{'…' if len(title) > 55 else ''}[/italic dim]\n"
            f"{dist_bar}",
            title=f"[dim]Kaynak {i}[/dim]",
            border_style="dim",
            padding=(0, 1),
            box=box.ROUNDED,
        ))
    if source_texts:
        console.print(Columns(source_texts, equal=False, expand=False))


def print_error(msg: str) -> None:
    console.print(Panel(
        f"[bold red]{msg}[/bold red]",
        border_style="red",
        title="[red]Hata[/red]",
    ))


def print_debug(debug_info: dict, debug_mode: bool) -> None:
    if not debug_mode or not debug_info:
        return
    if debug_info.get("agent_mode"):
        lines = []
        lines.append(f"[bold cyan]🤖 Intent:[/bold cyan] {debug_info.get('intent', '?')}")
        reasoning = debug_info.get("plan_reasoning", "")
        if reasoning:
            lines.append(f"[dim]{reasoning[:300]}[/dim]")
        lines.append("")
        lines.append("[bold cyan]📚 Plan Resources:[/bold cyan]")
        for r in debug_info.get("plan_resources", []):
            lines.append(f"  • {r['collection']}: {r['query']}")
        lines.append("")
        re_r = debug_info.get("re_retrieved", False)
        qrr = debug_info.get("quality_re_retrieved", False)
        vp = debug_info.get("validation_passed")
        vp_str = "PASS" if vp else ("FAIL" if vp is not None else "N/A")
        lines.append(f"  re_retrieved={re_r}  quality_re_retrieved={qrr}  validation={vp_str}")
        console.print(Panel(
            "\n".join(lines),
            title="[bold dim]🔍 DEBUG (agent)[/bold dim]",
            border_style="dim yellow",
            padding=(0, 1),
        ))
        return
    lines = []
    chunks = debug_info.get("chunks", [])
    threshold = debug_info.get("threshold", 1.5)
    lines.append(f"[bold cyan]📦 Bulunan Chunk'lar (threshold={threshold}):[/bold cyan]")
    for i, c in enumerate(chunks):
        dist = c["dist"]
        passed = dist <= threshold
        status = "[green]✓ GEÇTİ[/green]" if passed else "[red]✗ ELENDİ[/red]"
        baslik = c.get("title", "")[:45]
        publication = c.get("publication", "?")
        metin = c.get("metin", "")[:70].replace("\n", " ")
        lines.append(
            f"  [{i}] {status}  dist=[yellow]{dist:.4f}[/yellow]  "
            f"[bold]{publication}[/bold] | {baslik}"
        )
        lines.append(f"      [dim]{metin}…[/dim]")
    ctx_len = debug_info.get("context_len", 0)
    ctx_empty = debug_info.get("context_empty", True)
    ctx_color = "red" if ctx_empty else "green"
    lines.append("")
    lines.append(
        f"[bold cyan]📄 Context:[/bold cyan]  "
        f"[{ctx_color}]{'BOŞ ⚠' if ctx_empty else f'{ctx_len} karakter ✓'}[/{ctx_color}]"
    )
    if not ctx_empty:
        preview = debug_info.get("context_preview", "")
        lines.append(f"[dim]{preview}[/dim]")
    console.print(Panel(
        "\n".join(lines),
        title="[bold dim]🔍 DEBUG[/bold dim]",
        border_style="dim yellow",
        padding=(0, 1),
    ))


def print_full_source(service, source_meta: dict) -> None:
    """Fetch all columns for a record from the service and display as a table."""
    try:
        # Derive collection routing from document_type (canonical key)
        doc_type = source_meta.get("document_type", "")
        if doc_type == "tutanak":
            source_db = "minutes"
        elif doc_type in ("kanun_teklifi", "onerge"):
            source_db = "onerge"
        else:
            source_db = "gazete"
        chunk_id = source_meta.get("chunk_id")
        if not chunk_id:
            print_error("Kaynak ID bulunamadı.")
            return

        row_dict = service.inspect_record(source_db, chunk_id)
        if row_dict is None:
            print_error(f"Kayıt bulunamadı: {chunk_id} ({source_db})")
            return

        table = Table(
            title=f"Kayıt Detayı ({source_db.upper()} - {chunk_id})",
            box=box.DOUBLE_EDGE,
            show_header=True,
            header_style="bold cyan",
        )
        table.add_column("Alan", style="dim", width=20)
        table.add_column("Değer", style="white")

        full_text = ""
        for col, val in row_dict.items():
            val_str = str(val)
            if col in ("DOKUMAN_METNI", "content"):
                full_text = val_str
                table.add_row(col, f"[blue]{len(val_str)} karakter (Aşağıda tam metin)[/blue]")
            else:
                table.add_row(col, val_str)

        console.print(table)
        if full_text:
            console.print(Panel(
                full_text,
                title="[bold]Döküman Tam Metni[/bold]",
                border_style="blue",
                padding=(1, 2),
                subtitle="--- Metnin Sonu ---",
            ))
    except Exception as e:
        print_error(f"Veritabanı hatası: {e}")


def _dist_bar(dist: float) -> str:
    pct = max(0.0, min(1.0, 1.0 - dist / 2.0))
    filled = int(pct * 10)
    bar = "█" * filled + "░" * (10 - filled)
    color = "green" if pct > 0.65 else ("yellow" if pct > 0.45 else "red")
    return f"[{color}]{bar}[/{color}] [dim]{pct * 100:.0f}%[/dim]"
