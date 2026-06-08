"""Unit tests for the Planning Agent tools (offline — no Chroma/Ollama).

Focuses on ContextBuilderTool, which is pure dict-processing and guards the
"context always empty" regression: build_context() zips documents with the
distances list, so an empty distances list silently drops the whole context.
"""
import types

from src.agent.tools import ContextBuilderTool


def _cfg(threshold=1.8, max_chars=4000, total=12000):
    retrieval = types.SimpleNamespace(
        distance_threshold=threshold,
        context_max_chars=max_chars,
        context_total_max_chars=total,
    )
    return types.SimpleNamespace(retrieval=retrieval)


def test_build_context_nonempty_with_distances():
    tool = ContextBuilderTool(_cfg())
    results = [{
        "documents": ["birinci belge", "ikinci belge"],
        "metadatas": [{"chunk_id": "1"}, {"chunk_id": "2"}],
        "distances": [0.5, 0.9],
    }]
    ctx, sources = tool.build(results)
    assert "birinci belge" in ctx
    assert "ikinci belge" in ctx
    assert len(sources) == 2


def test_build_context_threshold_filters_far_docs():
    tool = ContextBuilderTool(_cfg(threshold=1.0))
    results = [{
        "documents": ["yakin", "uzak"],
        "metadatas": [{"chunk_id": "1"}, {"chunk_id": "2"}],
        "distances": [0.5, 2.0],
    }]
    ctx, sources = tool.build(results)
    assert "yakin" in ctx
    assert "uzak" not in ctx
    assert [s["chunk_id"] for s in sources] == ["1"]


def test_build_context_sources_align_with_survivors():
    tool = ContextBuilderTool(_cfg(threshold=1.0))
    results = [{
        "documents": ["a", "b", "c"],
        "metadatas": [{"chunk_id": "1"}, {"chunk_id": "2"}, {"chunk_id": "3"}],
        "distances": [0.1, 5.0, 0.2],
    }]
    ctx, sources = tool.build(results)
    assert [s["chunk_id"] for s in sources] == ["1", "3"]


def test_build_context_merges_multiple_collections():
    tool = ContextBuilderTool(_cfg())
    results = [
        {"documents": ["x"], "metadatas": [{"chunk_id": "1"}], "distances": [0.1]},
        {"documents": ["y"], "metadatas": [{"chunk_id": "2"}], "distances": [0.2]},
    ]
    ctx, sources = tool.build(results)
    assert "x" in ctx and "y" in ctx
    assert len(sources) == 2
