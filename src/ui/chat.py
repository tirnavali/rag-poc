"""Interactive terminal chat UI for the RAG archive assistant."""
from __future__ import annotations

import re
import sys
import os
import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

# Suppress library noise before importing chromadb/ollama
if os.environ.get("DEBUG_RAG", "0") != "1":
    warnings.filterwarnings("ignore", module="urllib3")
    import logging
    logging.getLogger("chromadb").setLevel(logging.ERROR)

from rich.align import Align
from rich.console import Group
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.spinner import Spinner
from rich.text import Text

from src.config import settings
from src.ui.commands import Action, parse_command
from src.ui.views import (
    console,
    print_banner,
    print_debug,
    print_error,
    print_full_source,
    print_help,
    print_sources,
    print_user,
)
from src.ui.components.collection_selector import select_collections_interactive
from src.retriever.multi_source import MultiSourceRetriever


REPORTS_DIR = Path(__file__).resolve().parents[2] / "artifacts" / "reports"


@dataclass
class ChatState:
    debug_mode: bool = False
    chat_history: list[dict] = field(default_factory=list)
    last_sources: list[dict] = field(default_factory=list)


def _slugify(text: str, max_len: int = 50) -> str:
    """Turkish-friendly slug for report filenames."""
    tr_map = str.maketrans("çğıöşüÇĞİÖŞÜ", "cgiosuCGIOSU")
    s = text.translate(tr_map).lower()
    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")
    return (s[:max_len] or "rapor").rstrip("-")


def save_report(
    query: str,
    markdown: str,
    sources: list[dict],
    distances: list[float],
    expanded_query: str | None,
    elapsed_seconds: float,
) -> Path:
    """Persist a /rapor result as a self-contained Markdown file with citations."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now()
    fname = f"{ts.strftime('%Y%m%d-%H%M')}_{_slugify(query)}.md"
    path = REPORTS_DIR / fname

    front = [
        "---",
        f"query: {query!r}",
        f"timestamp: {ts.isoformat(timespec='seconds')}",
        f"model: {settings.LLM_MODEL}",
        f"k: {settings.MUFETTIS_TOP_K}",
        f"elapsed_seconds: {elapsed_seconds:.1f}",
    ]
    if expanded_query:
        front.append(f"expanded_query: {expanded_query!r}")
    front.append("---")

    body = [markdown.strip() or "_(boş yanıt)_", "", "## Kaynaklar", ""]
    for i, (meta, dist) in enumerate(zip(sources, distances), 1):
        src = meta.get("document_type", "?")
        pub = meta.get("source_name") or "?"
        date = meta.get("date") or "?"
        author = meta.get("author") or "?"
        title = meta.get("title") or ""
        doc_id = meta.get("document_id") or "?"
        body.append(
            f"{i}. **[{src}#{doc_id}]** {pub} | {date} | {author}"
            + (f" — {title}" if title else "")
            + f"  _(d={dist:.3f})_"
        )

    path.write_text("\n".join(front) + "\n\n" + "\n".join(body) + "\n", encoding="utf-8")
    return path


def _run_agent_query(
    service,
    query: str,
    is_rapor: bool,
    mufettis_active: bool,
    console,
    debug_mode: bool = False,
    session_collections: list[str] | None = None,
) -> tuple[list[dict], list[float], str, str, dict]:
    """Execute a query via the Planning Agent pipeline."""
    sources: list[dict] = []
    dists: list[float] = []
    thinking_text = ""
    answer_text = ""
    debug_info: dict = {}

    if is_rapor:
        status_msg = "[bold yellow]📄 Agent: Arşiv raporu hazırlanıyor...[/bold yellow]"
    elif mufettis_active:
        status_msg = "[bold yellow]🕵️ Agent: Derin araştırma...[/bold yellow]"
    else:
        status_msg = "[bold yellow]🤖 Agent: Planlama ve arama...[/bold yellow]"

    with console.status(status_msg, spinner="dots") as status:
        def on_phase(name: str, block, model, details: dict) -> None:
            if name == "planning":
                status.update(f"[bold yellow]🤖 Planlama ({model})…[/bold yellow]")
            elif name == "retrieval":
                coll = details.get("collection", "?")
                q = details.get("query", "")
                status.update(f"[bold yellow]🔍 Arama: {coll} — {q}[/bold yellow]")
            elif name == "re_retrieval":
                status.update("[bold yellow]↻ Yeniden arama (filtreler gevşetildi)…[/bold yellow]")
            elif name == "answering":
                status.update(f"[bold yellow]✍️ Yanıt üretiliyor ({model})…[/bold yellow]")
            elif name == "validation":
                status.update("[bold yellow]✅ Doğrulama…[/bold yellow]")

        try:
            output = service.run_agent(
                query,
                on_phase=on_phase,
                session_collections=session_collections,
            )
            thinking_text = output.thinking
            answer_text = output.answer
            sources = output.sources
            dists = [0.0] * len(sources)

            if getattr(output, "scope", "in_scope") == "bad_word":
                console.print(Panel(
                    output.answer,
                    title="Uygunsuz dil",
                    border_style="red",
                ))
                return sources, dists, thinking_text, output.answer, debug_info

            if getattr(output, "scope", "in_scope") == "off_domain":
                console.print(Panel(
                    output.answer.split("Belki şunu")[0].strip(),
                    title="Alan dışı sorgu",
                    border_style="yellow",
                ))
                console.print()
                console.print("[bold]Belki şunu sormak istediniz:[/bold]")
                for i, s in enumerate(output.suggestions, 1):
                    console.print(f"  [cyan]{i}.[/cyan] {s}")
                console.print("\n[dim]Bir öneriyi seçmek için numarasını yaz veya yeni soru yaz.[/dim]")
                return sources, dists, thinking_text, output.answer, debug_info

            if debug_mode:
                from src.agent.tracer import PipelineTracer
                tracer = PipelineTracer()
                tracer.events = output.trace
                tracer.print_trace(console)

            if output.re_retrieved:
                console.print("  [bold cyan]↻ Re-retrieval tetiklendi (yetersiz sonuç)[/bold cyan]")

            if output.quality_re_retrieved:
                console.print("  [bold magenta]↻ Quality re-retrieval tetiklendi (yanıt yetersiz)[/bold magenta]")

            # Orchestrator-only UX notes
            notes: list[str] = []
            if output.evidence_decision and output.evidence_decision.judge_type == "llm":
                notes.append("kanıt değerlendirildi")
            if output.expanded:
                notes.append("genişletildi")
            if notes:
                console.print(f"  [dim]({' · '.join(notes)})[/dim]")

            if output.validation:
                v = output.validation
                status = "[green]GEÇTI[/green]" if v.passes else "[yellow]BAŞARISIZ[/yellow]"
                console.print(f"  [dim]Doğrulama: {status}[/dim]")

            debug_info = {
                "agent_mode": True,
                "intent": output.plan.intent if output.plan else "unknown",
                "plan_reasoning": output.plan.reasoning if output.plan else "",
                "plan_resources": [
                    {"collection": r.collection, "query": r.query_drafts[0].text if r.query_drafts else ""}
                    for r in (output.plan.resources if output.plan else [])
                ],
                "re_retrieved": output.re_retrieved,
                "quality_re_retrieved": output.quality_re_retrieved,
                "validation_passed": output.validation.passes if output.validation else None,
            }
        except Exception as e:
            answer_text = f"**Hata:** {type(e).__name__} – {e}"
            console.print(f"  [bold red]Agent hatası: {e}[/bold red]")

    return sources, dists, thinking_text, answer_text, debug_info


def main(agent_mode: bool = False, pipeline_path: str | None = None) -> None:
    console.clear()
    print_banner()

    mode_label = "AGENT" if agent_mode else "STANDARD"
    with console.status(f"[bold cyan]Arşiv yükleniyor ({mode_label})…[/bold cyan]", spinner="dots"):
        try:
            from src.generator.service import RAGService
            service = RAGService(pipeline_config_path=pipeline_path)
        except Exception as e:
            print_error(f"RAG sistemi başlatılamadı:\n{e}")
            sys.exit(1)

    if agent_mode:
        console.print(Align.center(Text("✓  Agent modu aktif — Planning Agent ile arama", style="bold yellow")))
    else:
        console.print(Align.center(Text("✓  Arşiv hazır — sorularınızı yazabilirsiniz", style="bold green")))
    console.print()

    # ─── Startup: Collection Selection ────────────────────────────────
    console.print("[bold cyan]📚 Koleksiyon Seçimi[/bold cyan]")
    console.print("Sorgulamak için koleksiyonları seçin.\n")

    try:
        selected_specs = select_collections_interactive(
            defaults=["gazete_arsivi", "tbmm_minutes"]
        )
    except ValueError as e:
        print_error(f"Hata: {e}")
        sys.exit(1)

    console.print(f"[green]✓ {len(selected_specs)} koleksiyon seçildi[/green]\n")

    # Collection names used by the orchestrator's policy stage.
    selected_collection_names: list[str] = [spec.name for spec in selected_specs]

    # Create multi-collection retriever for this session
    multi_retriever = MultiSourceRetriever(specs=selected_specs)

    state = ChatState()

    while True:
        try:
            console.print(Rule(style="dim"))
            raw = console.input("[bold cyan]❯ [/bold cyan]").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Görüşmek üzere![/dim]")
            break

        result = parse_command(raw)

        if result.action == Action.EMPTY:
            continue
        if result.action == Action.EXIT:
            console.print("[dim]Görüşmek üzere![/dim]")
            break
        if result.action == Action.CLEAR:
            console.clear()
            print_banner()
            continue
        if result.action == Action.TOGGLE_DEBUG:
            state.debug_mode = not state.debug_mode
            status = "[green]AÇIK[/green]" if state.debug_mode else "[red]KAPALI[/red]"
            console.print(f"  [dim]Debug modu: {status}[/dim]")
            continue
        if result.action == Action.SHOW_HELP:
            print_help()
            continue
        if result.action == Action.UNKNOWN_COMMAND:
            print_error(result.error_msg or "Bilinmeyen komut.")
            continue
        if result.action == Action.SHOW_SOURCE:
            if result.error_msg:
                print_error(result.error_msg)
                continue
            idx = result.source_index
            if idx is None or idx < 0 or idx >= len(state.last_sources):
                print_error("Geçerli bir kaynak numarası girmediniz.")
                continue
            print_full_source(service, state.last_sources[idx])
            continue

        # NORMAL_QUERY, MUFETTIS, or RAPOR
        is_rapor = result.action == Action.RAPOR
        mufettis_active = result.action == Action.MUFETTIS or is_rapor
        if result.error_msg:
            print_error(result.error_msg)
            continue
        query = result.query or raw
        print_user(raw)

        # --- Agent mode ---
        if agent_mode:
            sources, dists, thinking_text, answer_text, debug_info = _run_agent_query(
                service,
                query,
                is_rapor,
                mufettis_active,
                console,
                debug_mode=state.debug_mode,
                session_collections=selected_collection_names,
            )
            ts = datetime.now().strftime("%H:%M")
            if thinking_text.strip():
                console.print(Panel(
                    f"[dim]{thinking_text.strip()}[/dim]",
                    title="[bold dim]🧠 Düşünce Süreci[/bold dim]",
                    border_style="dim yellow",
                    padding=(0, 2),
                ))
            console.print(Panel(
                Markdown(answer_text or "…"),
                title=f"[bold cyan]🤖 Agent Yanıtı[/bold cyan]  [dim]{ts}[/dim]",
                title_align="left",
                border_style="cyan",
                padding=(1, 2),
            ))
            if state.debug_mode:
                print_debug(debug_info, state.debug_mode)
            print_sources(sources, dists)
            state.chat_history.append({"soru": raw, "yanit": answer_text})
            state.last_sources = sources
            continue

        # --- Standard mode: Retrieval phase ---
        if is_rapor:
            status_msg = "[bold yellow]📄 Arşiv raporu hazırlanıyor...[/bold yellow]"
        elif mufettis_active:
            status_msg = "[bold yellow]🕵️ Müfettiş araştırma yapıyor...[/bold yellow]"
        else:
            status_msg = "[dim]  Arşivde arama yapılıyor...[/dim]"
        sources: list[dict] = []
        dists: list[float] = []
        debug_info: dict = {}

        from src.common.tracer import PipelineTracer
        tracer = PipelineTracer()

        with console.status(status_msg, spinner="dots") as spinner:
            if mufettis_active:
                spinner.update("[bold cyan]🔍 Sorgu genişletiliyor (Derin Araştırma)...[/bold cyan]")

            # Multi-collection retrieval (balanced: equal results per collection)
            if mufettis_active:
                expanded = service.generator.expand_query(query)
                combined_query = f"{query} {expanded}"
                results = multi_retriever.retrieve_balanced(
                    combined_query,
                    per_collection_k=10,
                )
                results["expanded_query"] = expanded
            else:
                results = multi_retriever.retrieve_balanced(query)

            if mufettis_active and results.get("expanded_query"):
                console.print(f"  [italic dim]✨ Genişletilmiş sorgu: {results['expanded_query']}[/italic dim]")

            # Show multi-collection search summary
            collections_searched = set()
            for meta in results.get("metadatas", [[]])[0]:
                coll = meta.get("collection", "?")
                if coll:
                    collections_searched.add(coll)

            if collections_searched:
                colls_str = ", ".join(sorted(collections_searched))
                console.print(f"  [bold cyan]🔍 Koleksiyonlar aranda:[/bold cyan] {colls_str}")

            ctx_text = service.build_context(
                results,
                max_chars=settings.MUFETTIS_CONTEXT_MAX_CHARS if mufettis_active else settings.CONTEXT_MAX_CHARS,
                total_max_chars=settings.MUFETTIS_CONTEXT_TOTAL_MAX if mufettis_active else settings.CONTEXT_TOTAL_MAX,
            )
            debug_info = {
                "threshold": settings.DISTANCE_THRESHOLD,
                "chunks": [
                    {
                        "dist": d,
                        "collection": m.get("collection", "?"),
                        "source": m.get("source_name", m.get("gazete", "?")),
                        "title": m.get("title", m.get("baslik", "?")),
                        "metin": doc[:120],
                    }
                    for doc, m, d in zip(
                        results["documents"][0],
                        results["metadatas"][0],
                        results["distances"][0],
                    )
                ],
                "context_len": len(ctx_text),
                "context_empty": not ctx_text.strip(),
                "context_preview": ctx_text[:250].replace("\n", " "),
            }
            for meta, dist in zip(results["metadatas"][0], results["distances"][0]):
                if dist is not None and dist <= settings.DISTANCE_THRESHOLD:
                    sources.append(meta)
                    dists.append(dist)

        # --- Streaming generation phase ---
        ts = datetime.now().strftime("%H:%M")
        thinking_text = ""
        answer_text = ""

        def get_renderable(thinking: str, content: str, is_final: bool = False):
            elements = []
            if not thinking.strip() and not content.strip() and not is_final:
                loading_text = (
                    "  Arşiv müfettişi rapor hazırlıyor…"
                    if mufettis_active
                    else "  Arşiv asistanı cevap hazırlıyor…"
                )
                elements.append(Align.center(Spinner("dots", text=Text(loading_text, style="dim"))))
                return Group(*elements)
            if thinking.strip():
                preview = thinking.strip()
                if not is_final and len(preview) > 1200:
                    preview = "… " + preview[-1200:]
                elements.append(Panel(
                    f"[dim]{preview}[/dim]",
                    title="[bold dim]🧠 Düşünce Süreci[/bold dim]",
                    border_style="dim yellow",
                    padding=(0, 2),
                ))
            if content.strip() or is_final:
                if is_rapor:
                    assistant_title = "[bold yellow]📄 Arşiv Raporu[/bold yellow]"
                elif mufettis_active:
                    assistant_title = "[bold yellow]🕵️ Arşiv Müfettişi (Derin Araştırma)[/bold yellow]"
                else:
                    assistant_title = "[bold magenta]🗞  Arşiv Asistanı[/bold magenta]"
                elements.append(Panel(
                    Markdown(content or "…"),
                    title=f"{assistant_title}  [dim]{ts}[/dim]",
                    title_align="left",
                    border_style="yellow" if mufettis_active else "magenta",
                    padding=(1, 2),
                ))
            return Group(*elements)

        gen_start = time.perf_counter()
        with Live(get_renderable("", ""), console=console, refresh_per_second=10) as live:
            try:
                for chunk in service.ask_stream(query, mufettis_mode=mufettis_active, tracer=tracer, results=results):
                    if chunk["type"] == "thinking":
                        thinking_text += chunk["content"]
                    else:
                        answer_text += chunk["content"]
                    live.update(get_renderable(thinking_text, answer_text))
            except Exception as e:
                answer_text = f"**Hata:** {type(e).__name__} – {e}"
                live.update(get_renderable(thinking_text, answer_text))
        gen_elapsed = time.perf_counter() - gen_start

        # Print pipeline trace
        tracer.print_trace(console)
        console.print(f"  [dim]⏱ {gen_elapsed:.1f}s[/dim]")

        if is_rapor and answer_text.strip():
            try:
                path = save_report(
                    query=query,
                    markdown=answer_text,
                    sources=sources,
                    distances=dists,
                    expanded_query=results.get("expanded_query"),
                    elapsed_seconds=gen_elapsed,
                )
                rel = path.relative_to(Path.cwd()) if str(path).startswith(str(Path.cwd())) else path
                console.print(f"  [green]💾 Rapor kaydedildi:[/green] [link=file://{path}]{rel}[/link]")
            except Exception as e:
                print_error(f"Rapor kaydedilemedi: {type(e).__name__} – {e}")

        if state.debug_mode:
            print_debug(debug_info, state.debug_mode)
        print_sources(sources, dists)

        state.chat_history.append({"soru": raw, "yanit": answer_text})
        state.last_sources = sources


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="RAG Archive Chat Interface")
    parser.add_argument("--agent", action="store_true", help="Enable Planning Agent mode")
    parser.add_argument("--pipeline", type=str, default=None, help="Path to pipeline.yaml")
    args = parser.parse_args()
    main(agent_mode=args.agent, pipeline_path=args.pipeline)
