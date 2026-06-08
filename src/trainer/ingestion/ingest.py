"""CLI entry point for document ingestion.

Commands:
    --request FILE          Ingest documents from a JSON manifest file
    --validate FILE         Validate a request file (dry run, no ingestion)
    --diff FILE             Show what would be ingested vs skipped
    --list-collections      Show available collections and models
    --list-types            Show supported document types
    --status                Show manifest status summary
    --delete DOCUMENT_ID    Delete a document and its chunks

Examples:
    python -m src.trainer.ingestion.ingest --request ingest_d20.json
    python -m src.trainer.ingestion.ingest --request ingest_d20.json --only-changed
    python -m src.trainer.ingestion.ingest --validate ingest_d20.json
    python -m src.trainer.ingestion.ingest --status --collection minutes_jina_v4
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table

from src.config import settings
from src.config.collections import COLLECTIONS, MODEL_SPECS, get_spec, CollectionSpec
from src.trainer.ingestion.adapters import list_adapter_types, get_adapter
from src.trainer.ingestion.adapters.base import DocumentInput
from src.trainer.ingestion.downloader import is_url, validate_url
from src.trainer.ingestion.manifest import DocumentManifest
from src.trainer.ingestion.pipeline import IngestionPipeline

console = Console()


def _load_request(path: Path) -> dict:
    """Load and validate an ingest_request.json file."""
    if not path.exists():
        raise FileNotFoundError(f"Request dosyası bulunamadı: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("version") != "1.0":
        raise ValueError(f"Bilinmeyen version: {data.get('version')!r} (beklenen: '1.0')")
    if "collection" not in data:
        raise ValueError("'collection' alanı zorunlu")
    if "documents" not in data:
        raise ValueError("'documents' alanı zorunlu")
    return data


def _validate_request(data: dict) -> list[str]:
    """Validate request structure. Return list of error messages."""
    errors = []

    # Python-level validations (registry, file existence, duplicate IDs)
    collection_name = data.get("collection")
    if collection_name not in COLLECTIONS:
        errors.append(f"Koleksiyon '{collection_name}' tanımlı değil. Mevcut: {list(COLLECTIONS.keys())}")
    else:
        spec = COLLECTIONS[collection_name]
        console.print(f"[dim]Koleksiyon: {collection_name} → {spec.embed_model} ({spec.max_context_tokens} context, {spec.embed_dim} dim)[/dim]")

    docs = data.get("documents", [])
    if not docs:
        errors.append("'documents' listesi boş")

    seen_ids = set()
    for i, d in enumerate(docs):
        prefix = f"documents[{i}]"
        if "document_id" not in d:
            errors.append(f"{prefix}: 'document_id' zorunlu")
        elif d["document_id"] in seen_ids:
            errors.append(f"{prefix}: 'document_id' tekrar ediyor: {d['document_id']}")
        else:
            seen_ids.add(d["document_id"])

        if "document_type" not in d:
            errors.append(f"{prefix}: 'document_type' zorunlu")
        elif d["document_type"] not in list_adapter_types():
            errors.append(f"{prefix}: Bilinmeyen document_type: {d['document_type']!r}")

        if "document_source" in d and d["document_source"]:
            src = d["document_source"]
            if is_url(src):
                # URL — existence will be checked during ingest, not validation
                pass
            else:
                p = Path(src)
                if not p.exists():
                    errors.append(f"{prefix}: document_source bulunamadı: {p}")

    return errors


def cmd_request(args) -> None:
    """Ingest from a request JSON file."""
    data = _load_request(Path(args.request))
    errors = _validate_request(data)
    if errors:
        console.print(Panel("\n".join(f"[red]✗[/red] {e}" for e in errors), title="[bold red]Doğrulama Hatası[/bold red]", border_style="red"))
        sys.exit(1)

    collection_name = data["collection"]
    spec = get_spec(collection_name)
    manifest = DocumentManifest()

    # DocumentInput listesi oluştur
    # collection_name JSON root'tan gelir, per-document değildir
    collection_name = data["collection"]
    for d in data["documents"]:
        d.setdefault("collection_name", collection_name)
    documents = [DocumentInput.from_dict(d) for d in data["documents"]]

    # Diff (eğer --only-changed varsa)
    if args.only_changed:
        new, changed, unchanged = manifest.diff(documents)
        to_process = new + changed
        console.print(
            f"[dim]Yeni: {len(new)} | Değişmiş: {len(changed)} | "
            f"Atlanacak: {len(unchanged)}[/dim]\n"
        )
        if not to_process:
            console.print("[green]Tüm belgeler zaten güncel. İşlem yapılmadı.[/green]")
            return
        documents = to_process

    # Pipeline oluştur ve çalıştır
    pipeline = IngestionPipeline(spec=spec, manifest=manifest)
    results = pipeline.run_batch(documents, force=args.force)

    # Özet
    done = sum(1 for r in results if r.status == "done")
    skipped = sum(1 for r in results if r.status == "skipped")
    failed = sum(1 for r in results if r.status == "failed")
    total_chunks = sum(r.chunk_count for r in results if r.status == "done")

    parts = []
    if done: parts.append(f"[green]{done} işlendi[/green]")
    if skipped: parts.append(f"[dim]{skipped} atlandı[/dim]")
    if failed: parts.append(f"[red]{failed} hata[/red]")

    console.print(Panel(
        "  ".join(parts) + f"\nToplam {total_chunks} parça eklendi.",
        title="[bold]İşlem Özeti[/bold]",
        border_style="green" if not failed else "yellow",
    ))

    if failed:
        console.print("\n[red]Hatalı belgeler:[/red]")
        for r in results:
            if r.status == "failed":
                console.print(f"  [red]•[/red] {r.document_id}: {r.reason}")
        sys.exit(1)


def cmd_validate(args) -> None:
    """Validate a request file without ingesting."""
    data = _load_request(Path(args.validate))
    errors = _validate_request(data)

    # Optional: HEAD check for URLs
    url_errors = []
    for i, d in enumerate(data.get("documents", [])):
        src = d.get("document_source")
        if src and is_url(src):
            ok, msg = validate_url(src)
            if not ok:
                url_errors.append(f"documents[{i}]: URL erişilemez: {src} — {msg}")

    if url_errors:
        errors.extend(url_errors)

    if errors:
        console.print(Panel("\n".join(f"[red]✗[/red] {e}" for e in errors), title="[bold red]Doğrulama Başarısız[/bold red]", border_style="red"))
        sys.exit(1)

    docs = data["documents"]
    console.print(Panel(
        f"[green]✓[/green] {len(docs)} belge doğrulandı\n"
        f"[dim]Koleksiyon: {data['collection']}[/dim]",
        title="[bold green]Doğrulama Başarılı[/bold green]",
        border_style="green",
    ))


def cmd_diff(args) -> None:
    """Show what would be ingested vs skipped."""
    data = _load_request(Path(args.diff))
    errors = _validate_request(data)
    if errors:
        console.print(Panel("\n".join(f"[red]✗[/red] {e}" for e in errors), title="[bold red]Doğrulama Hatası[/bold red]", border_style="red"))
        sys.exit(1)

    collection_name = data["collection"]
    for d in data["documents"]:
        d.setdefault("collection_name", collection_name)
    documents = [DocumentInput.from_dict(d) for d in data["documents"]]
    manifest = DocumentManifest()
    new, changed, unchanged = manifest.diff(documents)

    table = Table(title="Manifest Diff")
    table.add_column("Durum", style="bold")
    table.add_column("Sayı", justify="right")
    table.add_column("Örnek", style="dim")

    if new:
        table.add_row("[green]Yeni[/green]", str(len(new)), new[0].document_id if new else "")
    if changed:
        table.add_row("[yellow]Değişmiş[/yellow]", str(len(changed)), changed[0].document_id if changed else "")
    if unchanged:
        table.add_row("[dim]Atlanacak[/dim]", str(len(unchanged)), "")

    console.print(table)


def cmd_list_collections(args) -> None:
    """Show available collections."""
    table = Table(title="Koleksiyonlar")
    table.add_column("İsim", style="bold")
    table.add_column("Model")
    table.add_column("Context", justify="right")
    table.add_column("Dim", justify="right")
    table.add_column("Late Chunking")
    table.add_column("Doküman Tipi")

    for name, spec in COLLECTIONS.items():
        table.add_row(
            name,
            spec.embed_model,
            str(spec.max_context_tokens),
            str(spec.embed_dim),
            "✓" if spec.supports_late_chunking else "✗",
            spec.doc_type.value,
        )
    console.print(table)


def cmd_list_types(args) -> None:
    """Show supported document types."""
    table = Table(title="Document Tipleri")
    table.add_column("Tip", style="bold")
    table.add_column("Açıklama")
    table.add_column("Kaynak")

    table.add_row("tutanak", "TBMM tutanak PDF (Docling + late chunking)", "PDF dosyası veya URL")
    table.add_row("press_clip", "Gazete kupürü (inline metin)", "JSON metadata")
    table.add_row("pdf_report", "Genel PDF rapor (Docling)", "PDF dosyası veya URL")
    table.add_row("kanun_teklifi", "TBMM kanun teklifi/önerge (Docling)", "PDF URL")
    console.print(table)


def cmd_status(args) -> None:
    """Show manifest status."""
    manifest = DocumentManifest()

    if args.collection:
        records = manifest.list_by_collection(args.collection)
        console.print(f"[dim]Koleksiyon: {args.collection} ({len(records)} kayıt)[/dim]\n")
    elif args.document_type:
        records = manifest.list_by_type(args.document_type)
        console.print(f"[dim]Tip: {args.document_type} ({len(records)} kayıt)[/dim]\n")
    else:
        counts = manifest.count_by_collection()
        if not counts:
            console.print("[dim]Manifest boş. Henüz hiç belge işlenmemiş.[/dim]")
            return

        table = Table(title="Manifest Durumu")
        table.add_column("Koleksiyon", style="bold")
        table.add_column("Tip")
        table.add_column("Tamamlandı", justify="right")
        table.add_column("Bekliyor", justify="right")
        table.add_column("Hata", justify="right")

        for col_name, types in counts.items():
            for dtype, statuses in types.items():
                done = statuses.get("done", 0)
                pending = statuses.get("pending", 0)
                failed = statuses.get("failed", 0)
                table.add_row(col_name, dtype, str(done), str(pending), str(failed))
        console.print(table)
        return

    # Detay listesi
    table = Table(title="Belge Listesi")
    table.add_column("document_id", style="dim")
    table.add_column("Tip")
    table.add_column("Durum")
    table.add_column("Chunk", justify="right")
    table.add_column("Tarih")

    for r in records[:50]:  # İlk 50
        status_color = {
            "done": "green",
            "pending": "yellow",
            "failed": "red",
            "skipped": "dim",
        }.get(r.status, "white")
        table.add_row(
            r.document_id,
            r.document_type,
            f"[{status_color}]{r.status}[/{status_color}]",
            str(r.chunk_count),
            r.document_date or "?",
        )
    console.print(table)


def cmd_delete(args) -> None:
    """Delete a document and its chunks."""
    if not args.collection:
        console.print("[red]Hata: Silme işlemi için --collection parametresi zorunludur.[/red]")
        sys.exit(1)

    manifest = DocumentManifest()
    record = manifest.get(args.delete, args.collection)
    if not record:
        console.print(f"[red]Belge bulunamadı: {args.delete} (Koleksiyon: {args.collection})[/red]")
        sys.exit(1)

    spec = get_spec(args.collection)
    pipeline = IngestionPipeline(spec=spec, manifest=manifest)
    deleted = pipeline._delete_chunks(args.delete)
    manifest.delete(args.delete, args.collection)

    console.print(f"[green]{args.delete} silindi ({args.collection}). {deleted} parça kaldırıldı.[/green]")


def _wiz_text(step: int, total: int, title: str, hint: str, prompt_label: str) -> str:
    while True:
        console.print()
        console.print(Panel(
            f"[dim]{hint}[/dim]",
            title=f"[bold]Adım {step} / {total}  —  {title}[/bold]",
            border_style="blue",
        ))
        value = Prompt.ask(f"  [bold cyan]>[/bold cyan] {prompt_label}").strip()
        if value:
            return value
        console.print("  [red]✗ Boş bırakılamaz.[/red]")


def _wiz_choice(step: int, total: int, title: str, hint: str, choices: list[tuple[str, str]]) -> str:
    while True:
        console.print()
        lines = [f"[dim]{hint}[/dim]\n"]
        for i, (key, label) in enumerate(choices, 1):
            lines.append(f"  [{i}] {key:<45} [dim]{label}[/dim]")
        console.print(Panel(
            "\n".join(lines),
            title=f"[bold]Adım {step} / {total}  —  {title}[/bold]",
            border_style="blue",
        ))
        raw = Prompt.ask(f"  [bold cyan]>[/bold cyan] Seçim (1-{len(choices)})").strip()
        if raw.isdigit() and 1 <= int(raw) <= len(choices):
            return choices[int(raw) - 1][0]
        console.print(f"  [red]✗ Geçersiz. 1-{len(choices)} arası bir sayı girin.[/red]")


def cmd_add_collection(args) -> None:
    """Interaktif koleksiyon ekleme sihirbazı — models.yaml'a yeni kayıt ekler."""
    from pathlib import Path as _Path
    from ruamel.yaml import YAML
    from ruamel.yaml.comments import CommentedMap
    from src.config.collections import _MODELS_YAML, CollectionSpec
    from src.config.document_types import DocumentType

    TOTAL = 6

    console.print(Panel(
        "[bold]models.yaml'a yeni bir koleksiyon kaydı ekler.[/bold]\n"
        "[dim]Her adımda girişiniz doğrulanır. Kaydetmeden önce özet gösterilir.[/dim]",
        title="[bold cyan]─── Yeni Koleksiyon Ekle ───[/bold cyan]",
        border_style="cyan",
    ))

    # Show existing collections for reference
    if COLLECTIONS:
        t = Table(title="Mevcut Koleksiyonlar", show_header=True, border_style="dim")
        t.add_column("Anahtar", style="dim")
        t.add_column("Koleksiyon Adı")
        t.add_column("Model")
        t.add_column("Tip")
        for k, spec in COLLECTIONS.items():
            t.add_row(k, spec.name, spec.embed_model.split("/")[-1], spec.doc_type.value)
        console.print(t)

    # Step 1 — registry key
    while True:
        registry_key = _wiz_text(1, TOTAL, "Koleksiyon Anahtarı",
            "models.yaml'daki kayıt anahtarı. snake_case, benzersiz olmalı.\n  Örnek: press_jina_v4, tutanaklar_jina_v3",
            "Koleksiyon anahtarı")
        if not re.match(r'^[a-z][a-z0-9_]*$', registry_key):
            console.print("  [red]✗ Yalnızca küçük harf, rakam ve alt çizgi. Harf ile başlamalı.[/red]")
            continue
        if registry_key in COLLECTIONS:
            console.print(f"  [red]✗ '{registry_key}' zaten kayıtlı. Başka bir isim girin.[/red]")
            continue
        break

    # Step 2 — collection_name (ChromaDB name)
    existing_names = {spec.name for spec in COLLECTIONS.values()}
    while True:
        collection_name = _wiz_text(2, TOTAL, "ChromaDB Koleksiyon Adı",
            "ChromaDB'deki koleksiyon adı. Benzersiz olmalı.\n  Örnek: gazete_arsivi_jina_v4",
            "ChromaDB koleksiyon adı")
        if collection_name in existing_names:
            console.print(f"  [red]✗ '{collection_name}' adlı bir ChromaDB koleksiyonu zaten var.[/red]")
            continue
        break

    # Step 3 — chroma_path
    chroma_path = _wiz_text(3, TOTAL, "ChromaDB Dizini",
        "Vektör veritabanının kaydedileceği dizin (göreli ya da mutlak).\n  Örnek: data_lake/press_clips_vectors",
        "ChromaDB dizini")

    # Step 4 — embed_model (numbered list with specs)
    model_choices = []
    for name, spec in MODEL_SPECS.items():
        ctx = spec["max_context_tokens"]
        ctx_label = f"{ctx//1024}K" if ctx >= 1024 else str(ctx)
        dim = spec["embed_dim"]
        lc = "· late chunking ✓" if spec["supports_late_chunking"] else ""
        model_choices.append((name, f"{ctx_label} context · {dim} dim {lc}".strip()))
    embed_model = _wiz_choice(4, TOTAL, "Embed Modeli",
        "Bu koleksiyonu indeksleyecek ve sorgulayacak embedding modeli.", model_choices)

    # Step 5 — doc_type
    type_labels = {
        "gazete": "Gazete kupürü",
        "tutanak": "TBMM tutanağı",
        "onerge": "Kanun teklifi / önerge",
        "custom": "Özel kaynak",
    }
    type_choices = [(dt.value, type_labels.get(dt.value, dt.value)) for dt in DocumentType]
    doc_type_value = _wiz_choice(5, TOTAL, "Belge Tipi",
        "Koleksiyondaki belgelerin türü.", type_choices)

    # Step 6 — chunk params
    chunk_defaults = dict(min_chunk_chars=400, max_chunk_chars=1500, max_chunk_tokens=512, min_chunk_tokens=384)
    console.print()
    console.print(Panel(
        "[dim]Varsayılan değerler:[/dim]\n\n"
        f"  min_chunk_chars={chunk_defaults['min_chunk_chars']}   "
        f"max_chunk_chars={chunk_defaults['max_chunk_chars']}\n"
        f"  max_chunk_tokens={chunk_defaults['max_chunk_tokens']}  "
        f"min_chunk_tokens={chunk_defaults['min_chunk_tokens']}",
        title=f"[bold]Adım 6 / {TOTAL}  —  Chunk Parametreleri[/bold]",
        border_style="blue",
    ))
    chunk_params = dict(chunk_defaults)
    if Confirm.ask("  [bold cyan]>[/bold cyan] Özelleştir mi?", default=False):
        for field in ("min_chunk_chars", "max_chunk_chars", "max_chunk_tokens", "min_chunk_tokens"):
            while True:
                raw = Prompt.ask(f"  {field}", default=str(chunk_defaults[field])).strip()
                if raw.isdigit() and int(raw) > 0:
                    chunk_params[field] = int(raw)
                    break
                console.print("  [red]✗ Pozitif tam sayı girin.[/red]")

    # Ask: set as default for doc_type?
    set_as_default = Confirm.ask(
        f"\n  Bu koleksiyonu '[bold cyan]{doc_type_value}[/bold cyan]' tipi için varsayılan yap?",
        default=False,
    )

    # Summary
    console.print()
    summary = Table(title="Özet", show_header=True, border_style="green")
    summary.add_column("Alan", style="bold")
    summary.add_column("Değer", style="cyan")
    rows = [
        ("registry_key", registry_key),
        ("collection_name", collection_name),
        ("chroma_path", chroma_path),
        ("embed_model", embed_model),
        ("doc_type", doc_type_value),
        ("min_chunk_chars", str(chunk_params["min_chunk_chars"])),
        ("max_chunk_chars", str(chunk_params["max_chunk_chars"])),
        ("max_chunk_tokens", str(chunk_params["max_chunk_tokens"])),
        ("min_chunk_tokens", str(chunk_params["min_chunk_tokens"])),
    ]
    if set_as_default:
        rows.append(("varsayılan yap", f"✓ ({doc_type_value})"))
    for k, v in rows:
        summary.add_row(k, v)
    console.print(summary)

    if not Confirm.ask("\n  [bold]Kaydet mi?[/bold]", default=True):
        console.print("[dim]İptal edildi.[/dim]")
        return

    # Write YAML with ruamel (preserves comments)
    yaml_rt = YAML()
    yaml_rt.preserve_quotes = True
    with open(_MODELS_YAML, "r", encoding="utf-8") as f:
        data = yaml_rt.load(f)

    new_entry = CommentedMap()
    new_entry["collection_name"] = collection_name
    new_entry["chroma_path"] = chroma_path
    new_entry["embed_model"] = embed_model
    new_entry["doc_type"] = doc_type_value
    new_entry["min_chunk_chars"] = chunk_params["min_chunk_chars"]
    new_entry["max_chunk_chars"] = chunk_params["max_chunk_chars"]
    new_entry["max_chunk_tokens"] = chunk_params["max_chunk_tokens"]
    new_entry["min_chunk_tokens"] = chunk_params["min_chunk_tokens"]

    data["collections"][registry_key] = new_entry
    if set_as_default:
        _original_default = data.get("defaults", {}).get(doc_type_value)
        data["defaults"][doc_type_value] = registry_key

    tmp = _MODELS_YAML.with_suffix(".yaml.tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            yaml_rt.dump(data, f)
        tmp.replace(_MODELS_YAML)
    except Exception as e:
        tmp.unlink(missing_ok=True)
        console.print(f"[red]✗ Yazma hatası: {e}[/red]")
        sys.exit(1)

    # Post-save validation — construct CollectionSpec to confirm MODEL_SPECS integrity
    try:
        CollectionSpec(
            name=collection_name,
            db_path=_Path(chroma_path),
            embed_model=embed_model,
            doc_type=DocumentType(doc_type_value),
            min_chunk_chars=chunk_params["min_chunk_chars"],
            max_chunk_chars=chunk_params["max_chunk_chars"],
            max_chunk_tokens=chunk_params["max_chunk_tokens"],
            min_chunk_tokens=chunk_params["min_chunk_tokens"],
        )
    except Exception as e:
        console.print(f"[red]✗ Koleksiyon doğrulama hatası: {e}[/red]")
        console.print("[dim]Değişiklikler geri alınıyor...[/dim]")
        yaml_rb = YAML()
        yaml_rb.preserve_quotes = True
        with open(_MODELS_YAML, "r", encoding="utf-8") as f:
            rollback_data = yaml_rb.load(f)
        if registry_key in rollback_data.get("collections", {}):
            del rollback_data["collections"][registry_key]
        if set_as_default and rollback_data.get("defaults", {}).get(doc_type_value) == registry_key:
            if _original_default is not None:
                rollback_data["defaults"][doc_type_value] = _original_default
            else:
                del rollback_data["defaults"][doc_type_value]
        try:
            with open(_MODELS_YAML, "w", encoding="utf-8") as f:
                yaml_rb.dump(rollback_data, f)
        except Exception as rb_err:
            console.print(f"[red]✗ Geri alma başarısız: {rb_err}[/red]")
            console.print(f"[red]  models.yaml bozulmuş olabilir. Elle kontrol edin: {_MODELS_YAML}[/red]")
        sys.exit(1)

    # Success + flow guidance
    console.print()
    console.print(Panel(
        f"[green]✓[/green] Koleksiyon eklendi: [bold]{registry_key}[/bold]",
        border_style="green",
    ))

    example_map = {"tutanak": "ornek_tutanak.json", "onerge": "ornek_onerge_manifest.json"}
    example_file = example_map.get(doc_type_value, "ornek_ingestion.json")

    console.print(
        f"\n[bold]Sıradaki adımlar:[/bold]\n\n"
        f"  [bold cyan]1.[/bold cyan] Belge manifestinizi hazırlayın — mevcut örnek:\n"
        f"       [dim]{example_file}[/dim]\n\n"
        f"     Manifestinizde [bold]\"collection\": \"{registry_key}\"[/bold] olmalı.\n\n"
        f"  [bold cyan]2.[/bold cyan] Doğrulayın:\n"
        f"     [dim]python -m src.trainer.ingestion.ingest --validate manifest.json[/dim]\n\n"
        f"  [bold cyan]3.[/bold cyan] İndeks oluşturun:\n"
        f"     [dim]python -m src.trainer.ingestion.ingest --request manifest.json[/dim]\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RAG Document Ingestion CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Subcommands via mutually exclusive
    parser.add_argument("--request", "-r", help="JSON ingest request dosyası")
    parser.add_argument("--validate", "-v", help="Doğrula (işlem yapma)")
    parser.add_argument("--diff", help="Manifest diff göster")
    parser.add_argument("--list-collections", action="store_true", help="Koleksiyonları listele")
    parser.add_argument("--list-types", action="store_true", help="Document tiplerini listele")
    parser.add_argument("--status", action="store_true", help="Manifest durumunu göster")
    parser.add_argument("--delete", help="Belge ve chunk'larını sil")
    parser.add_argument("--add-collection", action="store_true", help="Yeni koleksiyon ekle (interaktif sihirbaz)")

    # Modifiers
    parser.add_argument("--force", action="store_true", help="Zaten işlenmiş belgeleri tekrar işle.")
    parser.add_argument("--only-changed", action="store_true", help="Sadece değişmiş/yeni belgeleri işle")
    parser.add_argument("--collection", "-c", help="Status filtreleme")
    parser.add_argument("--document-type", "-t", help="Status filtreleme")

    args = parser.parse_args()

    if args.request:
        cmd_request(args)
    elif args.validate:
        cmd_validate(args)
    elif args.diff:
        cmd_diff(args)
    elif args.list_collections:
        cmd_list_collections(args)
    elif args.list_types:
        cmd_list_types(args)
    elif args.status:
        cmd_status(args)
    elif args.delete:
        cmd_delete(args)
    elif args.add_collection:
        cmd_add_collection(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
