"""Parça (chunk) düzeyinde genel yazar metaverisi çıkarımı.

Durum makinesi (State-machine) yapısı: atomları yukarıdan aşağıya tarar, mevcut
yazar/rol bilgisini atom sınırları boyunca yayar. AuthorSegmentExtractor aracılığıyla tipe özel strateji uygulanır.

Kurallar (Canonical) metaveri anahtarları: `author`, `author_role` (DocumentInput şeması ile eşleşir).
Görüntüleme etiketleri (Yazar / Konuşmacı), DocumentTypeSpec.prefix_labels tarafından yönetilir.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class AuthorTransition:
    """Signals a new author segment starting at this atom."""

    author: str
    author_role: Optional[str] = None
    extra: dict = field(default_factory=dict)
    confidence: float = 1.0


@dataclass
class TaggedAtom:
    """Atom with propagated author state."""

    text: str
    label: str
    author: Optional[str]
    author_role: Optional[str]
    segment_index: int
    is_continuation: bool
    extra: dict = field(default_factory=dict)
    page: Optional[int] = None
    pages: list[int] = field(default_factory=list)


class AuthorSegmentExtractor(ABC):
    """Per-type author-transition detector."""

    @abstractmethod
    def detect_transition(self, atom_text: str) -> Optional[AuthorTransition]:
        """Return AuthorTransition if this atom starts a new author segment, else None."""
        ...

    def clean_transition(self, transition: AuthorTransition, text: str) -> AuthorTransition:
        """Override to add LLM-based correction for dirty transitions."""
        return transition


_CONFIDENCE_THRESHOLD = 0.85


def tag_atoms(
    atoms: list[dict],
    extractor: AuthorSegmentExtractor,
    initial_author: Optional[str] = None,
    initial_role: Optional[str] = None,
) -> list[TaggedAtom]:
    """Walk atoms, propagate author state via the extractor.

    Args:
        atoms: List of {"text": str, "label": str} from Docling.
        extractor: Per-type AuthorSegmentExtractor implementation.
        initial_author: Document-level author fallback (from DocumentInput).
        initial_role: Document-level author_role fallback.

    Returns:
        List[TaggedAtom] same length as atoms.
    """
    current_author = initial_author
    current_role = initial_role
    current_extra: dict = {}
    seg_idx = 0
    tagged: list[TaggedAtom] = []

    for atom in atoms:
        trans = extractor.detect_transition(atom["text"])
        is_cont = trans is None
        if not is_cont:
            if trans.confidence < _CONFIDENCE_THRESHOLD:
                trans = extractor.clean_transition(trans, atom["text"])
            current_author = trans.author
            current_role = trans.author_role
            current_extra = dict(trans.extra)
            seg_idx += 1
        tagged.append(
            TaggedAtom(
                text=atom["text"],
                label=atom.get("label", "unknown"),
                author=current_author,
                author_role=current_role,
                segment_index=seg_idx,
                is_continuation=is_cont,
                extra=dict(current_extra),
                page=atom.get("page"),
                pages=atom.get("pages", []),
            )
        )
    return tagged


def tag_chunks_post_hoc(
    chunks: list[dict],
    extractor: AuthorSegmentExtractor,
    initial_author: Optional[str] = None,
    initial_role: Optional[str] = None,
    initial_extra: Optional[dict] = None,
    paragraph_sep: str = "\n\n",
) -> list[dict]:
    """Walk chunks in order, propagate author state across chunk boundaries.

    Used when chunk boundaries are decided externally (e.g. HybridChunker) and
    we cannot do pre-chunk atom tagging. Splits each chunk text by paragraph_sep
    into pseudo-atoms and feeds them to the extractor. State carries chunk → chunk.

    Mutates chunk["metadata"] in-place with: author, author_role, authors_in_chunk,
    segment_indices, starts_mid_segment.
    """
    current_author = initial_author
    current_role = initial_role
    current_extra: dict = dict(initial_extra or {})
    seg_idx = 0

    for chunk in chunks:
        paragraphs = chunk["text"].split(paragraph_sep)
        char_counts: Counter[str] = Counter()
        role_by_author: dict[str, Optional[str]] = {}
        authors_seen: list[str] = []
        segment_indices: list[int] = []
        first_para_is_continuation = True

        for i, para in enumerate(paragraphs):
            trans = extractor.detect_transition(para)
            is_cont = trans is None
            if not is_cont:
                if trans.confidence < _CONFIDENCE_THRESHOLD:
                    trans = extractor.clean_transition(trans, para)
                current_author = trans.author
                current_role = trans.author_role
                current_extra = dict(trans.extra)
                seg_idx += 1
                if i == 0:
                    first_para_is_continuation = False

            if seg_idx not in segment_indices:
                segment_indices.append(seg_idx)
            if current_author:
                char_counts[current_author] += len(para)
                role_by_author.setdefault(current_author, current_role)
                if current_author not in authors_seen:
                    authors_seen.append(current_author)

        if char_counts:
            primary_author = char_counts.most_common(1)[0][0]
            primary_role = role_by_author.get(primary_author)
        else:
            primary_author = None
            primary_role = None
        starts_mid = first_para_is_continuation and current_author is not None

        meta = chunk.setdefault("metadata", {})
        meta.update(
            {
                "author": primary_author,
                "author_role": primary_role,
                "authors_in_chunk": authors_seen,
                "segment_indices": segment_indices,
                "starts_mid_segment": starts_mid,
            }
        )
        if current_extra:
            meta.setdefault("extra", {}).update(current_extra)

    return chunks
