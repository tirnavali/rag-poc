"""Noop extractor — never detects transitions.

All atoms inherit initial_author from DocumentInput (single-author docs).
"""
from __future__ import annotations

from typing import Optional

from src.common.parsing.author_extractor import AuthorSegmentExtractor, AuthorTransition


class NoopAuthorExtractor(AuthorSegmentExtractor):
    def detect_transition(self, atom_text: str) -> Optional[AuthorTransition]:
        return None
