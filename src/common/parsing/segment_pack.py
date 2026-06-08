"""Author-aware atom packer.

Like greedy_pack but:
  1. Soft-splits on author transition once chunk reaches min_chars
  2. Aggregates author metadata per chunk (primary_author, authors_in_chunk,
     segment_indices, starts_mid_segment)
  3. Optionally prepends a "[X devam ediyor]" prefix when a chunk starts mid-segment

Designed to replace greedy_pack for multi-author document types.
"""
from __future__ import annotations

from collections import Counter
from typing import Optional

from src.common.parsing.author_extractor import TaggedAtom


def segment_aware_pack(
    atoms: list[TaggedAtom],
    min_chars: int,
    max_chars: int,
    join_str: str = "\n\n",
    inject_continuation_prefix: bool = True,
) -> list[dict]:
    """Pack TaggedAtoms into chunks with chunk-level author metadata.

    Args:
        atoms: TaggedAtoms (output of tag_atoms).
        min_chars: Minimum chunk size before splitting.
        max_chars: Maximum chunk size — soft cap.
        join_str: Separator between atoms.
        inject_continuation_prefix: When chunk starts mid-segment, prepend
            "[Konuşmacı/Yazar X devam ediyor]\n" to chunk text.

    Returns:
        list[dict]: each chunk has {text, atoms (list[TaggedAtom]), metadata}.
        metadata fields:
            author, author_role,
            authors_in_chunk: list[str],
            segment_indices: list[int],
            starts_mid_segment: bool,
            continuation_prefix: str | None
    """
    if not atoms:
        return []

    chunks: list[dict] = []
    cur_atoms: list[TaggedAtom] = []
    cur_text = ""

    def _flush():
        if not cur_atoms:
            return
        chunks.append(_finalize_chunk(cur_atoms, cur_text, inject_continuation_prefix))

    for atom in atoms:
        if not cur_atoms:
            cur_atoms = [atom]
            cur_text = atom.text
            continue

        proposed_len = len(cur_text) + len(join_str) + len(atom.text)
        author_changed = (
            not atom.is_continuation
            and cur_atoms[-1].segment_index != atom.segment_index
        )

        soft_split = (
            len(cur_text) >= min_chars
            and author_changed
        )
        hard_split = (
            len(cur_text) >= min_chars
            and proposed_len > max_chars
        )

        if soft_split or hard_split:
            _flush()
            cur_atoms = [atom]
            cur_text = atom.text
        else:
            cur_atoms.append(atom)
            cur_text += join_str + atom.text

    _flush()
    return chunks


def _finalize_chunk(
    cur_atoms: list[TaggedAtom],
    cur_text: str,
    inject_continuation_prefix: bool,
) -> dict:
    """Build chunk dict with aggregated metadata."""
    # Char-weighted primary author
    char_counts: Counter[str] = Counter()
    role_by_author: dict[str, Optional[str]] = {}
    authors_seen: list[str] = []
    segment_indices: list[int] = []

    pages_seen: list[int] = []
    for a in cur_atoms:
        if a.segment_index not in segment_indices:
            segment_indices.append(a.segment_index)
        if a.author:
            char_counts[a.author] += len(a.text)
            role_by_author.setdefault(a.author, a.author_role)
            if a.author not in authors_seen:
                authors_seen.append(a.author)
        for p in getattr(a, "pages", []):
            if p not in pages_seen:
                pages_seen.append(p)

    pages_seen = sorted(pages_seen)
    primary_page = pages_seen[0] if pages_seen else None

    if char_counts:
        primary_author, _ = char_counts.most_common(1)[0]
        primary_role = role_by_author.get(primary_author)
    else:
        primary_author = None
        primary_role = None

    first_atom = cur_atoms[0]
    starts_mid = first_atom.is_continuation and first_atom.author is not None

    continuation_prefix: Optional[str] = None
    final_text = cur_text
    if inject_continuation_prefix and starts_mid:
        role_part = f" ({first_atom.author_role})" if first_atom.author_role else ""
        continuation_prefix = f"[{first_atom.author}{role_part} devam ediyor]"
        final_text = continuation_prefix + "\n" + cur_text

    return {
        "text": final_text,
        "atoms": cur_atoms,
        "metadata": {
            "author": primary_author,
            "author_role": primary_role,
            "authors_in_chunk": authors_seen,
            "segment_indices": segment_indices,
            "starts_mid_segment": starts_mid,
            "continuation_prefix": continuation_prefix,
            "page": primary_page,
            "pages": pages_seen,
        },
    }
