"""Per-type AuthorSegmentExtractor registry.

Lookup by DocumentInput.document_type. Missing types fall back to NoopAuthorExtractor.
"""
from __future__ import annotations

from src.common.parsing.author_extractor import AuthorSegmentExtractor
from src.common.parsing.extractors.gazete import GazeteAuthorExtractor
from src.common.parsing.extractors.noop import NoopAuthorExtractor
from src.common.parsing.extractors.onerge import OnergeAuthorExtractor
from src.common.parsing.extractors.tutanak import TutanakAuthorExtractor


EXTRACTORS: dict[str, AuthorSegmentExtractor] = {
    "tutanak": TutanakAuthorExtractor(),
    "gazete": GazeteAuthorExtractor(),
    "press_clip": GazeteAuthorExtractor(),
    "onerge": OnergeAuthorExtractor(),
    "kanun_teklifi": OnergeAuthorExtractor(),
    "pdf_report": NoopAuthorExtractor(),
}

_NOOP = NoopAuthorExtractor()


def get_extractor(document_type: str) -> AuthorSegmentExtractor:
    """Lookup with safe fallback."""
    return EXTRACTORS.get(document_type, _NOOP)
