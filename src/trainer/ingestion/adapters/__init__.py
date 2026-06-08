"""Document adapter registry.

Maps document_type strings to adapter classes.
Adding a new document type means:
1. Create a new adapter module inheriting DocumentAdapter
2. Register it here
3. Done — no changes to pipeline or CLI needed
"""
from __future__ import annotations

from src.trainer.ingestion.adapters.base import DocumentAdapter
from src.trainer.ingestion.adapters.tutanak_pdf import TutanakPdfAdapter
from src.trainer.ingestion.adapters.press_clip import PressClipAdapter
from src.trainer.ingestion.adapters.pdf_report import PdfReportAdapter
from src.trainer.ingestion.adapters.kanun_teklifi import KanunTeklifiAdapter


ADAPTER_REGISTRY: dict[str, type[DocumentAdapter]] = {
    "tutanak":       TutanakPdfAdapter,
    "press_clip":    PressClipAdapter,
    "pdf_report":    PdfReportAdapter,
    "kanun_teklifi": KanunTeklifiAdapter,
}


def get_adapter(document_type: str) -> DocumentAdapter:
    """Get an adapter instance for a document type.

    Args:
        document_type: Must exist in ADAPTER_REGISTRY.

    Raises:
        ValueError: If document_type is not registered.
    """
    if document_type not in ADAPTER_REGISTRY:
        raise ValueError(
            f"Unknown document_type: '{document_type}'. "
            f"Available: {list(ADAPTER_REGISTRY.keys())}"
        )
    return ADAPTER_REGISTRY[document_type]()


def list_adapter_types() -> list[str]:
    """Return all registered document type names."""
    return list(ADAPTER_REGISTRY.keys())
