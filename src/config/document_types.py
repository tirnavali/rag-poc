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


# Which FilterCriteria fields are meaningful for each document type's indexed
# metadata. Kept separate from DocumentTypeSpec.filter_fields (which drives
# display and is intentionally minimal) to avoid side effects, and expanded to
# the full FilterCriteria schema: the year family (year/year_lte/year_gte) maps
# to the single `year` column, and parliament-only fields (period/session) are
# included for tutanak/onerge — both are verified present in the live
# tbmm_tutanaklar_nomic_v2 index, and absent from gazete. `source_name` is the
# discriminating press field; for tutanak it is a constant ("TBMM Tutanakları"),
# so filtering on it there only risks cross-type zero-results.
#
# `author` is applicable for all real types. In gazete the author is a clean
# journalist name. In tutanak/onerge the indexed speaker label is the FULL title
# as it appears in the record — e.g. "BAŞBAKAN RECEP TAYYİP ERDOĞAN" — so an
# exact-match filter for a plain person name would zero out. That is handled NOT
# by masking author away but by `build_chroma_where` (filter_translators.py),
# which resolves the name to the collection's matching labels and emits
# `author $in [...]` (see AuthorResolver). `author_role` (bare role like "bakan")
# has no reliable indexed column and stays OMITTED for tutanak/onerge.
# A value of None means "do not mask" (unknown/custom collections, fail-open).
FILTER_APPLICABILITY: dict[DocumentType, set[str] | None] = {
    DocumentType.GAZETE: {
        "year", "year_lte", "year_gte",
        "author", "author_role", "source_name", "document_type",
    },
    DocumentType.TUTANAK: {
        "year", "year_lte", "year_gte",
        "author", "period", "session", "document_type",
    },
    DocumentType.ONERGE: {
        "year", "year_lte", "year_gte",
        "author", "period", "session", "document_type",
    },
    DocumentType.CUSTOM: None,
}


_LEGACY_KEY_MAP: dict[str, str] = {
    "gazete": "source_name",
    "tarih": "date",
    "yazar": "author",
    "baslik": "source_title",
    "konular": "topics",
}


def normalize_metadata(meta: dict) -> dict:
    """Map legacy Turkish metadata keys to the canonical schema.

    Older gazete collection (`gazete_arsivi`, embedder `gazete_arsivi`) stored
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
