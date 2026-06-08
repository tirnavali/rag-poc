"""Document type registry — enum + display specs + prefix formatting.

Enables query agents to route by document type and format display metadata
with appropriate field order and labels (gazete vs tutanak vs onerge).
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class DocumentType(str, Enum):
    """Document source types for routing and display."""

    GAZETE = "gazete"
    TUTANAK = "tutanak"
    ONERGE = "onerge"
    CUSTOM = "custom"


@dataclass(frozen=True)
class DocumentTypeSpec:
    """Display and filter spec for a document type."""

    type: DocumentType
    display_name_tr: str
    prefix_fields: tuple[str, ...]
    prefix_labels: tuple[str, ...]
    filter_fields: tuple[str, ...]


DOCUMENT_TYPES: dict[DocumentType, DocumentTypeSpec] = {
    DocumentType.GAZETE: DocumentTypeSpec(
        type=DocumentType.GAZETE,
        display_name_tr="Gazete Küpürü",
        prefix_fields=("source_name", "date", "author", "source_title"),
        prefix_labels=("Kaynak", "Tarih", "Yazar", "Başlık"),
        filter_fields=("year", "author", "source_name"),
    ),
    DocumentType.TUTANAK: DocumentTypeSpec(
        type=DocumentType.TUTANAK,
        display_name_tr="TBMM Tutanağı",
        prefix_fields=("source_name", "date", "author", "source_title"),
        prefix_labels=("Kaynak", "Tarih", "Konuşmacı", "Konu"),
        filter_fields=("year", "author"),
    ),
    DocumentType.ONERGE: DocumentTypeSpec(
        type=DocumentType.ONERGE,
        display_name_tr="Kanun Teklifi / Önerge",
        prefix_fields=("source_name", "date", "author", "source_title"),
        prefix_labels=("Kaynak", "Tarih", "Yazar", "Başlık"),
        filter_fields=("year", "author"),
    ),
    DocumentType.CUSTOM: DocumentTypeSpec(
        type=DocumentType.CUSTOM,
        display_name_tr="Özel Kaynak",
        prefix_fields=("source_name", "date", "author", "source_title"),
        prefix_labels=("Kaynak", "Tarih", "Yazar", "Başlık"),
        filter_fields=(),
    ),
}


def get_doc_type_spec(dt: DocumentType) -> DocumentTypeSpec:
    """Fetch spec for a document type."""
    return DOCUMENT_TYPES[dt]


_LEGACY_KEY_MAP: dict[str, str] = {
    "gazete": "source_name",
    "tarih": "date",
    "yazar": "author",
    "baslik": "source_title",
    "konular": "topics",
}


def normalize_metadata(meta: dict) -> dict:
    """Map legacy Turkish metadata keys to the canonical schema.

    Older gazete collection (`gazete_arsivi`, embedder `press_nomic`) stored
    metadata under TR keys (gazete/tarih/yazar/baslik/konular); the canonical
    schema uses (source_name/date/author/source_title/topics) plus a derived
    `year`. This shim brings legacy records up to the canonical shape at
    read-time so downstream (format_prefix, sanitizer, context builder) works
    uniformly. Idempotent: canonical keys present in `meta` are left as-is.
    """
    if not meta:
        return meta
    out = dict(meta)
    for legacy, canonical in _LEGACY_KEY_MAP.items():
        if legacy in out and canonical not in out:
            out[canonical] = out[legacy]
    if "year" not in out:
        date_val = out.get("date")
        if isinstance(date_val, str) and len(date_val) >= 4 and date_val[:4].isdigit():
            out["year"] = int(date_val[:4])
    return out


def format_prefix(meta: dict, dt: DocumentType) -> str:
    """Build display prefix string for a chunk based on its document type.

    Args:
        meta: chunk metadata dict
        dt: document type

    Returns:
        Formatted prefix string like "Kaynak: Sabah | Tarih: 1997-01-04\n"
        or empty string if no matching fields.
    """
    spec = DOCUMENT_TYPES[dt]
    parts = []
    for field, label in zip(spec.prefix_fields, spec.prefix_labels):
        val = meta.get(field)
        if val:
            parts.append(f"{label}: {val}")
    return " | ".join(parts) + "\n" if parts else ""
