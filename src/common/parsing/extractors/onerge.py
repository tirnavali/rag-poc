"""Önerge / kanun teklifi signatory patterns.

Law proposals list signatories at the top:
  "Konya Milletvekili Ahmet Yılmaz ve 14 arkadaşının ..."
  "(İmza: AHMET YILMAZ, MEHMET KAYA, ...)"
"""
from __future__ import annotations

import re
from typing import Optional

from src.common.parsing.author_extractor import AuthorSegmentExtractor, AuthorTransition


_NAME = r"[A-ZÇĞİÖŞÜ][A-ZÇĞİÖŞÜa-zçğıöşü]+(?:\s+[A-ZÇĞİÖŞÜ][A-ZÇĞİÖŞÜa-zçğıöşü]+)+"

PAT_PROPOSER = re.compile(
    rf"([A-ZÇĞİÖŞÜ][a-zçğıöşü]+)\s+Milletvekili\s+({_NAME})(?:\s+ve\s+(\d+)\s+arkadaşı)?"
)
PAT_SIGN_LIST = re.compile(rf"\(\s*İmza\s*[:\-—]\s*([^)]+)\)")


class OnergeAuthorExtractor(AuthorSegmentExtractor):
    def detect_transition(self, atom_text: str) -> Optional[AuthorTransition]:
        head = atom_text[:500]

        m = PAT_PROPOSER.search(head)
        if m:
            constituency = m.group(1).strip()
            name = m.group(2).strip()
            others = m.group(3)
            extra = {"constituency": constituency}
            if others:
                extra["co_signers_count"] = int(others)
            return AuthorTransition(
                author=name, author_role="milletvekili", extra=extra
            )

        m = PAT_SIGN_LIST.search(head)
        if m:
            raw = m.group(1)
            signatories = [s.strip() for s in re.split(r"[,;]", raw) if s.strip()]
            if signatories:
                return AuthorTransition(
                    author=signatories[0],
                    author_role="imzacı",
                    extra={"signatories": signatories},
                )

        return None
