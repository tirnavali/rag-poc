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
