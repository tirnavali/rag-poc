"""Unit tests for CitationBuilder."""
from __future__ import annotations

from src.agent.citations import CitationBuilder
from src.agent.schemas import Chunk


def _chunk(i: int) -> Chunk:
    return Chunk(
        chunk_id=f"c{i}",
        document_id=f"d{i}",
        collection_name="col",
        doc_type="gazete",
        source_title=f"title-{i}",
        text=f"body-{i}",
        score=0.5,
        metadata={"year": 2020 + i},
    )


def test_citation_builder_produces_indexed_dicts():
    chunks = [_chunk(1), _chunk(2)]
    cites = CitationBuilder.build(chunks)
    assert len(cites) == 2
    assert cites[0]["index"] == 1
    assert cites[1]["index"] == 2
    assert cites[0]["chunk_id"] == "c1"
    assert cites[0]["collection_name"] == "col"
    assert cites[0]["doc_type"] == "gazete"
    assert cites[0]["source_title"] == "title-1"
    assert cites[0]["metadata"] == {"year": 2021}


def test_citation_builder_empty_returns_empty_list():
    assert CitationBuilder.build([]) == []


def _dated_chunk(chunk_id: str, date: str, author: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        document_id=f"doc-{chunk_id}",
        collection_name="tutanaklar",
        doc_type="tutanak",
        source_title="",
        text="...",
        score=0.5,
        metadata={"date": date, "author": author},
    )


def test_citation_builder_without_final_answer_has_no_cited_key():
    """Omitting final_answer preserves the old behavior exactly (no callers break)."""
    cites = CitationBuilder.build([_dated_chunk("c1", "2018-10-17", "ENGİN ÖZKOÇ")])
    assert "cited" not in cites[0]


def test_citation_builder_flags_chunks_referenced_in_answer():
    chunks = [
        _dated_chunk("c1", "2018-10-17", "ENGİN ÖZKOÇ"),
        _dated_chunk("c2", "2018-11-07", "BAŞKAN"),
    ]
    final_answer = (
        "Özkoç böyle dedi (Kaynak: TBMM Tutanakları, 2018-10-17, Engin Özkoç)."
    )
    cites = CitationBuilder.build(chunks, final_answer=final_answer)
    assert cites[0]["cited"] is True
    # Retrieved but never referenced in the prose — flagged, NOT dropped from the list.
    assert cites[1]["cited"] is False
    assert len(cites) == 2


def test_citation_builder_matches_citation_without_kaynak_prefix():
    """Müfettiş-mode prompt omits the "Kaynak:" prefix — must still match."""
    chunks = [_dated_chunk("c1", "2019-07-15", "ENGİN ÖZKOÇ")]
    final_answer = "Bu iddiayı öne sürmüştür (TBMM Tutanakları, 2019-07-15, Engin Özkoç)."
    cites = CitationBuilder.build(chunks, final_answer=final_answer)
    assert cites[0]["cited"] is True
