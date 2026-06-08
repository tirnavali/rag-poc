"""TBMM tutanak author transition patterns.

TBMM minutes have highly structured speaker headers:
  BAŞKAN — ...
  AHMET YILMAZ (İstanbul) — ...
  DEVLET BAKANI HASAN GEMİCİ — ...
"""
from __future__ import annotations

import re
from typing import Optional

from src.common.parsing.author_extractor import AuthorSegmentExtractor, AuthorTransition


_DASH = r"[—\-–]"
_NAME_PART = r"[A-ZÇĞİÖŞÜ][A-ZÇĞİÖŞÜa-zçğıöşü]+"
_UPPER_NAME = rf"{_NAME_PART}(?:\s+{_NAME_PART})+"

_TITLE_KW = r"(?:BAKANI?|BAŞBAKAN(?:\s+YARDIMCISI)?|BAKANLIĞI|DEVLET\s+BAKANI)"
_MINISTRY_BLOCK = rf"(?:[A-ZÇĞİÖŞÜ]+\s+)*{_TITLE_KW}"
_COMPOUND_TITLE = rf"{_MINISTRY_BLOCK}(?:\s+VE\s+{_MINISTRY_BLOCK})*"

PAT_CHAIR = re.compile(rf"^\s*(BAŞKAN|BAŞKANVEKİLİ)\s*{_DASH}")
PAT_DEPUTY = re.compile(
    rf"^\s*({_UPPER_NAME})\s*\(([^)]+)\)\s*{_DASH}"
)
PAT_MINISTER = re.compile(
    rf"^\s*({_COMPOUND_TITLE})\s+({_UPPER_NAME})\s*{_DASH}"
)


def _compute_confidence(name: str) -> float:
    if len(name) > 35:
        return 0.5
    if any(kw in name for kw in ("BAKANI", "YARDIMCISI", "BAŞBAKAN", " VE ")):
        return 0.5
    if re.search(r'\b[A-ZÇĞİÖŞÜ]\b', name):
        return 0.5
    return 1.0


class TutanakAuthorExtractor(AuthorSegmentExtractor):
    def detect_transition(self, atom_text: str) -> Optional[AuthorTransition]:
        head = re.sub(r'\s+', ' ', atom_text[:200]).strip()

        m = PAT_MINISTER.search(head)
        if m:
            role, name = m.group(1).strip(), m.group(2).strip()
            return AuthorTransition(
                author=name,
                author_role=role.lower(),
                confidence=_compute_confidence(name),
            )

        m = PAT_DEPUTY.search(head)
        if m:
            name, constituency = m.group(1).strip(), m.group(2).strip()
            return AuthorTransition(
                author=name,
                author_role="milletvekili",
                extra={"constituency": constituency},
                confidence=_compute_confidence(name),
            )

        m = PAT_CHAIR.search(head)
        if m:
            role = m.group(1).strip()
            return AuthorTransition(
                author=role, author_role=role.lower(), confidence=1.0
            )

        return None

    def clean_transition(
        self, transition: AuthorTransition, text: str
    ) -> AuthorTransition:
        from src.common.parsing.llm_transition_cleaner import clean_author_transition

        return clean_author_transition(transition, text, document_type="tutanak")
