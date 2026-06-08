"""Gazete/dergi byline patterns.

Most newspaper clippings are single-author and inherit DocumentInput.author.
Detect explicit byline only when present.
"""
from __future__ import annotations

import re
from typing import Optional

from src.common.parsing.author_extractor import AuthorSegmentExtractor, AuthorTransition


_NAME = r"[A-ZÇĞİÖŞÜ][A-ZÇĞİÖŞÜa-zçğıöşü]+(?:\s+[A-ZÇĞİÖŞÜ][A-ZÇĞİÖŞÜa-zçğıöşü]+)+"

PAT_YAZAN = re.compile(rf"^\s*Yazan\s*[:\-—]\s*({_NAME})", re.IGNORECASE)
PAT_BYLINE = re.compile(rf"^\s*({_NAME})\s*[—\-]\s*(?:Hürriyet|Milliyet|Sabah|Cumhuriyet|Tercüman|Türkiye)")


class GazeteAuthorExtractor(AuthorSegmentExtractor):
    def detect_transition(self, atom_text: str) -> Optional[AuthorTransition]:
        head = atom_text[:200]

        m = PAT_YAZAN.search(head)
        if m:
            return AuthorTransition(author=m.group(1).strip(), author_role="köşe yazarı")

        m = PAT_BYLINE.search(head)
        if m:
            return AuthorTransition(author=m.group(1).strip(), author_role="muhabir")

        return None
