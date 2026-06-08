#!/usr/bin/env python
"""Clear an entire collection from both ChromaDB and SQLite manifest.

Usage:
    python devtools/clear_collection.py <collection_name>

Example:
    python devtools/clear_collection.py tbmm_tutanaklar_nomic_v2
"""
from __future__ import annotations

import sys
from pathlib import Path

from rich.console import Console

from src.config.collections import get_spec
from src.trainer.ingestion.manifest import DocumentManifest
import chromadb

console = Console()


def clear_collection(collection_name: str) -> None:
    # 1. Get collection spec
    try:
        spec = get_spec(collection_name)
    except KeyError as e:
        console.print(f"[red]Hata: {e}[/red]")
        sys.exit(1)

    console.print(f"[yellow]'{collection_name}' koleksiyonu tamamen temizleniyor...[/yellow]")
    console.print(f"  Chroma Yolu: {spec.db_path}")
    console.print(f"  Chroma Koleksiyon Adı: {spec.name}")

    # 2. Clear ChromaDB
    try:
        client = chromadb.PersistentClient(path=str(spec.db_path))
        # Drop and recreate collection
        try:
            client.delete_collection(name=spec.name)
            console.print("[green]✓ ChromaDB koleksiyonu silindi.[/green]")
        except ValueError:
            # Collection might not exist yet
            console.print("[dim]ChromaDB koleksiyonu zaten mevcut değil (atlanıyor).[/dim]")

        # Recreate empty collection with correct metadata
        client.get_or_create_collection(
            name=spec.name,
            metadata={"hnsw:space": "cosine"},
        )
        console.print("[green]✓ Boş ChromaDB koleksiyonu yeniden oluşturuldu.[/green]")
    except Exception as e:
        console.print(f"[red]ChromaDB temizleme hatası: {e}[/red]")
        sys.exit(1)

    # 3. Clear SQLite manifest
    try:
        manifest = DocumentManifest()
        # Count records first
        cursor = manifest._conn.execute(
            "SELECT COUNT(*) FROM document_manifest WHERE collection_name = ?",
            (collection_name,),
        )
        count = cursor.fetchone()[0]

        if count > 0:
            manifest._conn.execute(
                "DELETE FROM document_manifest WHERE collection_name = ?",
                (collection_name,),
            )
            manifest._conn.commit()
            console.print(f"[green]✓ SQLite manifest veri tabanından {count} adet belge kaydı silindi.[/green]")
        else:
            console.print("[dim]SQLite manifest veri tabanında bu koleksiyona ait kayıt bulunamadı.[/dim]")
    except Exception as e:
        console.print(f"[red]Manifest temizleme hatası: {e}[/red]")
        sys.exit(1)

    console.print(f"[bold green]Başarılı: '{collection_name}' koleksiyonu tamamen sıfırlandı.[/bold green]")


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        console.print("Kullanım: [bold]python scratch/clear_collection.py <koleksiyon_adi>[/bold]")
        sys.exit(1)

    collection_name = sys.argv[1]
    clear_collection(collection_name)


if __name__ == "__main__":
    main()
