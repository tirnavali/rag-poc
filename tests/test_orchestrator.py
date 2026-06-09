"""End-to-end orchestrator tests with mocked SearchTool and answering."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.agent.orchestrator import OrchestratorAgent
from src.agent.schemas import (
    CollectionSearchPlan,
    SearchPlan,
    SearchQueryDraft,
)
from src.common.llm_client_pool import LLMClientPool
from src.common.schemas import ExtractedFilterResponse, FilterCriteria
from src.config.pipeline_loader import load_pipeline_config


def _make_plan(*collections: str) -> SearchPlan:
    return SearchPlan(
        intent="factual",
        query_type="fact",
        resources=[
            CollectionSearchPlan(
                collection=c,
                query_drafts=[SearchQueryDraft(text="q", top_k=5)],
            )
            for c in collections
        ],
        reasoning="r",
    )


def _make_search_result(
    chunk_ids: list[str],
    doc_ids: list[str],
    collection: str,
) -> dict:
    return {
        "documents": [f"body-{i}" for i in chunk_ids],
        "metadatas": [
            {
                "chunk_id": cid,
                "document_id": did,
                "doc_type": "gazete",
                "source_title": f"t-{cid}",
                "collection": collection,
            }
            for cid, did in zip(chunk_ids, doc_ids)
        ],
        "distances": [0.1 for _ in chunk_ids],
    }


def _agent(
    monkeypatch,
    plan_collections=("gazete_arsivi",),
    result_chunks_by_collection=None,
):
    cfg = load_pipeline_config()
    pool = LLMClientPool.from_config(cfg)
    agent = OrchestratorAgent(cfg, pool)

    monkeypatch.setattr(
        agent._planner, "plan",
        lambda q, tracer=None: _make_plan(*plan_collections),
    )

    def _search(collection_key, query_text, filters=None, top_k=5):
        chunks = (result_chunks_by_collection or {}).get(collection_key, [])
        return _make_search_result(
            chunk_ids=[c["chunk_id"] for c in chunks],
            doc_ids=[c["document_id"] for c in chunks],
            collection=collection_key,
        )
    monkeypatch.setattr(agent._search_tool, "search", _search)

    monkeypatch.setattr(
        agent._answer_tool, "generate",
        lambda query, context, mufettis_mode=False: ("thinking", "Cevap metni."),
    )
    monkeypatch.setattr(
        agent._sanitizer, "validate",
        lambda *a, **kw: None,
    )
    return agent


def test_orchestrator_no_allowed_collections_returns_refuse(monkeypatch):
    agent = _agent(monkeypatch, plan_collections=("disallowed_collection",))
    out = agent.run("q", session_collections=["gazete_arsivi"])
    assert out.policy_result.allowed_collections == []
    # refuse path: short Turkish message, no streamed answer
    assert out.answer
    assert out.answer != "Cevap metni."


def test_orchestrator_happy_path_returns_answer(monkeypatch):
    chunks_a = [{"chunk_id": f"a{i}", "document_id": f"da{i}"} for i in range(3)]
    chunks_b = [{"chunk_id": f"b{i}", "document_id": f"db{i}"} for i in range(3)]
    agent = _agent(
        monkeypatch,
        plan_collections=("col_a", "col_b"),
        result_chunks_by_collection={"col_a": chunks_a, "col_b": chunks_b},
    )
    out = agent.run("q", session_collections=["col_a", "col_b"])
    assert out.answer == "Cevap metni."
    assert out.evidence_decision.action == "answer"
    assert out.evidence_decision.judge_type == "heuristic"
    assert len(out.sources) >= 4
    assert out.assembly
    assert out.policy_result.allowed_collections == ["col_a", "col_b"]


def test_orchestrator_zero_chunks_returns_clarify(monkeypatch):
    agent = _agent(
        monkeypatch,
        plan_collections=("col_a",),
        result_chunks_by_collection={"col_a": []},
    )
    out = agent.run("q", session_collections=["col_a"])
    assert out.evidence_decision.action == "clarify"


def test_orchestrator_expand_path_uses_reserves(monkeypatch):
    chunks = [{"chunk_id": f"k{i}", "document_id": f"d{i}"} for i in range(6)]
    agent = _agent(
        monkeypatch,
        plan_collections=("col_a",),
        result_chunks_by_collection={"col_a": chunks},
    )
    out = agent.run("q", session_collections=["col_a"])
    assert out.expanded is True


def test_orchestrator_single_collection_failure_continues(monkeypatch):
    chunks_a = [{"chunk_id": f"a{i}", "document_id": f"da{i}"} for i in range(3)]
    cfg = load_pipeline_config()
    pool = LLMClientPool.from_config(cfg)
    agent = OrchestratorAgent(cfg, pool)

    monkeypatch.setattr(
        agent._planner, "plan",
        lambda q, tracer=None: _make_plan("col_a", "col_b"),
    )

    def _search(collection_key, query_text, filters=None, top_k=5):
        if collection_key == "col_b":
            raise RuntimeError("boom")
        return _make_search_result(
            chunk_ids=[c["chunk_id"] for c in chunks_a],
            doc_ids=[c["document_id"] for c in chunks_a],
            collection=collection_key,
        )
    monkeypatch.setattr(agent._search_tool, "search", _search)
    monkeypatch.setattr(
        agent._answer_tool, "generate",
        lambda query, context, mufettis_mode=False: ("t", "ok"),
    )
    monkeypatch.setattr(agent._sanitizer, "validate", lambda *a, **kw: None)

    # Run does not raise even though col_b crashed
    out = agent.run("q", session_collections=["col_a", "col_b"])
    # col_a chunks are still assembled
    used_collections = {item.collection_name for item in (out.assembly or [])}
    assert "col_a" in used_collections


def test_orchestrator_emits_phase_trace_events(monkeypatch):
    chunks = [{"chunk_id": f"k{i}", "document_id": f"d{i}"} for i in range(4)]
    agent = _agent(
        monkeypatch,
        plan_collections=("col_a", "col_b"),
        result_chunks_by_collection={"col_a": chunks[:2], "col_b": chunks[2:]},
    )
    out = agent.run("q", session_collections=["col_a", "col_b"])
    phases = {e.phase for e in out.trace}
    assert "planning" in phases
    assert "policy" in phases
    assert "allocation" in phases
    assert "retrieval" in phases
    assert "assembly" in phases
    assert "judge" in phases
    assert "answering" in phases
    assert "citation" in phases


def test_orchestrator_propagates_extracted_filters_to_retrieval(monkeypatch):
    """FE → Planner.plan → allocator._collect_first_filters → _retrieve_all → search.

    _planner.plan'ı doğrudan patch'lemiyoruz (o _apply_filter_extractor'ı atlardı);
    bunun yerine _inner._generate_plan'ı sabit plana patch'leyip gerçek Planner.plan'ın
    FilterExtractor'ı çalıştırmasını sağlıyoruz.
    """
    mock_fe = MagicMock()
    mock_fe.model = "test-model"
    mock_fe.extract.return_value = ExtractedFilterResponse(
        refined_query="iklim", filters=FilterCriteria(year=2023)
    )

    cfg = load_pipeline_config()
    pool = LLMClientPool.from_config(cfg)
    agent = OrchestratorAgent(cfg, pool, filter_extractor=mock_fe)

    monkeypatch.setattr(
        agent._planner._inner, "_generate_plan",
        lambda q, tracer: _make_plan("col_a"),
    )

    chunks = [{"chunk_id": f"a{i}", "document_id": f"da{i}"} for i in range(3)]
    captured = {}

    def _search(collection_key, query_text, filters=None, top_k=5):
        captured["filters"] = filters
        return _make_search_result(
            chunk_ids=[c["chunk_id"] for c in chunks],
            doc_ids=[c["document_id"] for c in chunks],
            collection=collection_key,
        )
    monkeypatch.setattr(agent._search_tool, "search", _search)
    monkeypatch.setattr(
        agent._answer_tool, "generate",
        lambda query, context, mufettis_mode=False: ("t", "ok"),
    )
    monkeypatch.setattr(agent._sanitizer, "validate", lambda *a, **kw: None)

    agent.run("2023 iklim", session_collections=["col_a"])

    mock_fe.extract.assert_called_once_with("2023 iklim")
    # Orchestrator yolu artık allocator'da ChromaFilterTranslator ile çevrilmiş
    # geçerli Chroma where-dict'i geçirir (legacy planner yoluyla aynı sözleşme).
    assert captured["filters"] == {"year": {"$eq": 2023}}


def test_orchestrator_falls_back_to_refined_query_when_no_drafts(monkeypatch):
    """Draft'ı olmayan koleksiyon, ham sorgu yerine refined_query ile aranır."""
    mock_fe = MagicMock()
    mock_fe.model = "test-model"
    mock_fe.extract.return_value = ExtractedFilterResponse(
        refined_query="iklim", filters=FilterCriteria()
    )

    cfg = load_pipeline_config()
    pool = LLMClientPool.from_config(cfg)
    agent = OrchestratorAgent(cfg, pool, filter_extractor=mock_fe)

    # Draft'sız plan → orchestrator fallback yolu devreye girer.
    plan_no_drafts = SearchPlan(
        intent="factual",
        query_type="fact",
        resources=[CollectionSearchPlan(collection="col_a", query_drafts=[])],
        reasoning="r",
    )
    monkeypatch.setattr(
        agent._planner._inner, "_generate_plan",
        lambda q, tracer: plan_no_drafts,
    )

    captured = {}

    def _search(collection_key, query_text, filters=None, top_k=5):
        captured["query_text"] = query_text
        return _make_search_result(chunk_ids=["a0"], doc_ids=["da0"], collection=collection_key)
    monkeypatch.setattr(agent._search_tool, "search", _search)
    monkeypatch.setattr(
        agent._answer_tool, "generate",
        lambda query, context, mufettis_mode=False: ("t", "ok"),
    )
    monkeypatch.setattr(agent._sanitizer, "validate", lambda *a, **kw: None)

    agent.run("2023 iklim degisikligi", session_collections=["col_a"])

    assert captured["query_text"] == "iklim"


def test_orchestrator_runs_each_planner_draft_as_parallel_query(monkeypatch):
    """Bir koleksiyonun tüm draft'ları ayrı arama olarak çalışır (paralel)."""
    cfg = load_pipeline_config()
    pool = LLMClientPool.from_config(cfg)
    agent = OrchestratorAgent(cfg, pool)

    plan = SearchPlan(
        intent="factual",
        query_type="fact",
        resources=[
            CollectionSearchPlan(
                collection="col_a",
                query_drafts=[
                    SearchQueryDraft(text="draft-1", top_k=5),
                    SearchQueryDraft(text="draft-2", top_k=5),
                    SearchQueryDraft(text="draft-3", top_k=5),
                ],
            )
        ],
        reasoning="r",
    )
    monkeypatch.setattr(agent._planner, "plan", lambda q, tracer=None: plan)

    seen_queries = []

    def _search(collection_key, query_text, filters=None, top_k=5):
        seen_queries.append(query_text)
        # Each draft surfaces an overlapping chunk so RRF fusion has something to fuse.
        return _make_search_result(
            chunk_ids=["shared", query_text],
            doc_ids=["d-shared", f"d-{query_text}"],
            collection=collection_key,
        )
    monkeypatch.setattr(agent._search_tool, "search", _search)
    monkeypatch.setattr(
        agent._answer_tool, "generate",
        lambda query, context, mufettis_mode=False: ("t", "ok"),
    )
    monkeypatch.setattr(agent._sanitizer, "validate", lambda *a, **kw: None)

    out = agent.run("q", session_collections=["col_a"])

    # All three planner drafts were issued as separate searches.
    assert sorted(seen_queries) == ["draft-1", "draft-2", "draft-3"]
    # The chunk found by every draft is deduped (RRF), not triplicated.
    chunk_ids = [s["chunk_id"] for s in out.sources]
    assert chunk_ids.count("shared") == 1


def test_fuse_draft_chunks_ranks_shared_chunks_first():
    """RRF: birden çok draft'ın bulduğu chunk, tek draft'ın bulduğundan önce gelir."""
    from src.agent.schemas import Chunk

    def _c(cid: str) -> Chunk:
        return Chunk(
            chunk_id=cid, document_id=cid, collection_name="col",
            doc_type="gazete", source_title="t", text="b", score=1.0,
        )

    list_a = [_c("shared"), _c("only_a")]
    list_b = [_c("only_b"), _c("shared")]
    fused = OrchestratorAgent._fuse_draft_chunks([list_a, list_b])

    assert [c.chunk_id for c in fused][0] == "shared"
    assert sorted(c.chunk_id for c in fused) == ["only_a", "only_b", "shared"]


def test_fuse_draft_chunks_single_list_passthrough():
    """Tek draft → reranker sırası korunur (RRF devreye girmez)."""
    from src.agent.schemas import Chunk

    chunks = [
        Chunk(chunk_id=f"c{i}", document_id=f"d{i}", collection_name="col",
              doc_type="gazete", source_title="t", text="b", score=1.0)
        for i in range(3)
    ]
    fused = OrchestratorAgent._fuse_draft_chunks([chunks])
    assert [c.chunk_id for c in fused] == ["c0", "c1", "c2"]
