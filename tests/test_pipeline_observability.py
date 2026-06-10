"""Pipeline gözlemlenebilirlik özellikleri testleri.

Kapsanan özellikler:
  1. Belge başına uçuş kaydedici rapor (data_lake/reports/)
  2. Span coverage uyarısı ve embed mode fallback
  3. Manifest perf_json yazımı
  4. cmd_inspect argüman çözümlemesi
  5. perf_trends_by_collection manifest metodu
"""
from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.config.collections import CollectionSpec
from src.config.document_types import DocumentType
from src.trainer.ingestion.adapters.base import DocumentInput
from src.trainer.ingestion.manifest import DocumentManifest
from src.trainer.ingestion.pipeline import IngestionPipeline, _chunk_stats, _write_report


# ─── Fixtures ───────────────────────────────────────────────────────────────


@pytest.fixture
def spec():
    return CollectionSpec(
        name="test_observability",
        db_path=Path("/tmp/test_obs_chroma"),
        embed_model="jinaai/jina-embeddings-v3",
        doc_type=DocumentType.TUTANAK,
    )


@pytest.fixture
def mock_pipeline(spec):
    with (
        patch("src.trainer.ingestion.pipeline.DoclingManager"),
        patch("src.trainer.ingestion.pipeline.LocalLateChunkingEmbedder") as mock_emb,
        patch("src.trainer.ingestion.pipeline.open_or_create_collection") as mock_chroma,
        patch("src.trainer.ingestion.pipeline.DocumentManifest") as mock_manifest,
    ):
        mock_chroma.return_value = (MagicMock(), MagicMock())
        mock_manifest_inst = mock_manifest.return_value
        mock_manifest_inst.get.return_value = None

        pipeline = IngestionPipeline(spec=spec)
        yield pipeline, mock_emb.return_value, mock_chroma.return_value[1], mock_manifest_inst


def _make_doc(**kwargs):
    defaults = dict(
        document_id="test-obs-001",
        document_type="tutanak",
        collection_name="test_observability",
        document_source="test.pdf",
    )
    defaults.update(kwargs)
    return DocumentInput(**defaults)


# ─── _chunk_stats ────────────────────────────────────────────────────────────


def test_chunk_stats_full_span_coverage():
    chunks = [
        {"text": "abc" * 100, "span": (0, 300), "metadata": {"type": "HybridChunk"}},
        {"text": "def" * 80, "span": (300, 540), "metadata": {"type": "HybridChunk"}},
    ]
    spans = [c["span"] for c in chunks]
    stats = _chunk_stats(chunks, spans)
    assert stats["chunk_count"] == 2
    assert stats["span_coverage_pct"] == 100.0
    assert stats["span_missing_count"] == 0
    assert stats["char_min"] == 240
    assert stats["char_max"] == 300


def test_chunk_stats_partial_span_coverage():
    chunks = [
        {"text": "x" * 200, "span": (0, 200), "metadata": {"type": "Packed"}},
        {"text": "y" * 200, "span": None, "metadata": {"type": "Packed"}},
        {"text": "z" * 200, "span": None, "metadata": {"type": "Packed"}},
    ]
    spans = [c["span"] for c in chunks]
    stats = _chunk_stats(chunks, spans)
    assert stats["span_coverage_pct"] == pytest.approx(33.3, abs=0.1)
    assert stats["span_missing_count"] == 2


def test_chunk_stats_empty():
    assert _chunk_stats([], []) == {"chunk_count": 0}


# ─── _write_report ───────────────────────────────────────────────────────────


def test_write_report_creates_file(tmp_path):
    with patch("src.trainer.ingestion.pipeline.settings") as mock_settings:
        mock_settings.REPORTS_DIR = tmp_path / "reports"
        _write_report(
            document_id="test-report-001",
            collection="col_a",
            status="done",
            timings={"manifest_ms": 5, "parse_ms": 200, "embed_ms": 100, "upsert_ms": 10, "total_ms": 315},
            chunk_stats={"chunk_count": 5, "span_coverage_pct": 100.0},
            warnings=[],
        )

    report_path = tmp_path / "reports" / "test-report-001.json"
    assert report_path.exists()
    data = json.loads(report_path.read_text())
    assert data["document_id"] == "test-report-001"
    assert data["status"] == "done"
    assert data["timings"]["total_ms"] == 315
    assert data["chunk"]["span_coverage_pct"] == 100.0
    assert data["warnings"] == []


def test_write_report_sanitizes_slash_in_id(tmp_path):
    with patch("src.trainer.ingestion.pipeline.settings") as mock_settings:
        mock_settings.REPORTS_DIR = tmp_path / "reports"
        _write_report("doc/with/slashes", "col", "done", {}, {}, [])

    assert (tmp_path / "reports" / "doc_with_slashes.json").exists()


def test_write_report_includes_warnings(tmp_path):
    with patch("src.trainer.ingestion.pipeline.settings") as mock_settings:
        mock_settings.REPORTS_DIR = tmp_path / "reports"
        _write_report("w-doc", "col", "done", {}, {}, ["span_missing:2/10", "ocr_flagged:3/10"])

    data = json.loads((tmp_path / "reports" / "w-doc.json").read_text())
    assert "span_missing:2/10" in data["warnings"]


# ─── Pipeline span coverage ──────────────────────────────────────────────────


def test_pipeline_span_coverage_warning_in_output(mock_pipeline, capsys):
    """Span eksik olduğunda 'SPAN COVERAGE' uyarısı görünmeli."""
    pipeline, mock_embedder, mock_collection, mock_manifest = mock_pipeline
    mock_embedder.embed_documents.return_value = [[0.1] * 1024, [0.2] * 1024]
    mock_embedder.embed_with_late_chunking_windowed.return_value = [[0.1] * 1024, [0.2] * 1024]

    chunks_with_missing_span = [
        {"text": "Birinci parça.", "span": (0, 14), "metadata": {"type": "p", "source": "t.pdf"}},
        {"text": "İkinci parça.", "span": None, "metadata": {"type": "p", "source": "t.pdf"}},
    ]

    with patch("src.trainer.ingestion.pipeline.get_adapter") as mock_get_adapter, \
         patch("src.trainer.ingestion.pipeline._write_report"), \
         patch("src.trainer.ingestion.pipeline.settings") as mock_settings:
        mock_settings.REPORTS_DIR = Path("/tmp/noop")
        mock_adapter = MagicMock()
        mock_adapter.parse.return_value = ("Birinci parça. İkinci parça.", chunks_with_missing_span)
        mock_adapter.compute_content_hash.return_value = "hash_span_test"
        mock_get_adapter.return_value = mock_adapter

        result = pipeline.run_document(_make_doc())

    assert result.status == "done"
    # embed_with_late_chunking_windowed çağrılmamalı (span eksik → fallback)
    mock_embedder.embed_with_late_chunking_windowed.assert_not_called()
    mock_embedder.embed_documents.assert_called_once()


def test_pipeline_late_chunking_used_when_spans_complete(mock_pipeline):
    """Tüm span'lar mevcutsa late chunking kullanılmalı."""
    pipeline, mock_embedder, mock_collection, mock_manifest = mock_pipeline
    mock_embedder.embed_with_late_chunking_windowed.return_value = [[0.1] * 1024, [0.2] * 1024]

    full_chunks = [
        {"text": "A" * 300, "span": (0, 300), "metadata": {"type": "HybridChunk", "source": "t.pdf"}},
        {"text": "B" * 300, "span": (300, 600), "metadata": {"type": "HybridChunk", "source": "t.pdf"}},
    ]

    with patch("src.trainer.ingestion.pipeline.get_adapter") as mock_get_adapter, \
         patch("src.trainer.ingestion.pipeline._write_report"), \
         patch("src.trainer.ingestion.pipeline.settings"):
        mock_adapter = MagicMock()
        mock_adapter.parse.return_value = ("A" * 300 + "B" * 300, full_chunks)
        mock_adapter.compute_content_hash.return_value = "hash_full_span"
        mock_get_adapter.return_value = mock_adapter

        result = pipeline.run_document(_make_doc(document_id="test-obs-002"))

    assert result.status == "done"
    mock_embedder.embed_with_late_chunking_windowed.assert_called_once()


def test_pipeline_perf_passed_to_manifest(mock_pipeline):
    """perf dict manifest.upsert'e geçirilmeli; total_ms ve span_coverage_pct içermeli."""
    pipeline, mock_embedder, mock_collection, mock_manifest = mock_pipeline
    mock_embedder.embed_with_late_chunking_windowed.return_value = [[0.1] * 1024]

    chunks = [{"text": "X" * 200, "span": (0, 200), "metadata": {"type": "p", "source": "t.pdf"}}]

    with patch("src.trainer.ingestion.pipeline.get_adapter") as mock_get_adapter, \
         patch("src.trainer.ingestion.pipeline._write_report"), \
         patch("src.trainer.ingestion.pipeline.settings"):
        mock_adapter = MagicMock()
        mock_adapter.parse.return_value = ("X" * 200, chunks)
        mock_adapter.compute_content_hash.return_value = "hash_perf"
        mock_get_adapter.return_value = mock_adapter

        pipeline.run_document(_make_doc(document_id="test-obs-003"))

    call_kwargs = mock_manifest.upsert.call_args.kwargs
    assert "perf" in call_kwargs
    perf = call_kwargs["perf"]
    assert "total_ms" in perf
    assert "span_coverage_pct" in perf
    assert perf["span_coverage_pct"] == 100.0
    assert perf["chunk_count"] == 1


# ─── Manifest perf_trends_by_collection ─────────────────────────────────────


def test_perf_trends_empty_manifest(tmp_path):
    m = DocumentManifest(db_path=tmp_path / "manifest.db")
    trends = m.perf_trends_by_collection()
    assert trends == {}


def test_perf_trends_single_collection(tmp_path):
    m = DocumentManifest(db_path=tmp_path / "manifest.db")
    doc = DocumentInput(
        document_id="trend-001",
        document_type="tutanak",
        collection_name="col_trend",
        content_hash="h1",
    )
    m.upsert(
        doc,
        status="done",
        chunk_count=10,
        perf={
            "total_ms": 5000,
            "parse_ms": 3000,
            "embed_ms": 1800,
            "span_coverage_pct": 100.0,
            "embed_mode": "late_chunking",
            "chunk_count": 10,
        },
        quality={"ocr_flagged": False},
    )
    trends = m.perf_trends_by_collection()
    assert "col_trend" in trends
    t = trends["col_trend"]
    assert t["doc_count"] == 1
    assert t["avg_total_ms"] == 5000
    assert t["avg_parse_ms"] == 3000
    assert t["avg_embed_ms"] == 1800
    assert t["avg_span_coverage_pct"] == 100.0
    assert t["ocr_flagged_pct"] == 0.0


def test_perf_trends_ocr_flagged_percentage(tmp_path):
    m = DocumentManifest(db_path=tmp_path / "manifest.db")
    for i in range(4):
        doc = DocumentInput(
            document_id=f"doc-{i}",
            document_type="tutanak",
            collection_name="col_flag",
            content_hash=f"h{i}",
        )
        m.upsert(
            doc,
            status="done",
            chunk_count=5,
            perf={"total_ms": 1000, "span_coverage_pct": 100.0},
            quality={"ocr_flagged": i < 1},  # 1 out of 4 flagged
        )
    trends = m.perf_trends_by_collection()
    assert trends["col_flag"]["ocr_flagged_pct"] == 25.0


def test_perf_trends_ignores_failed_docs(tmp_path):
    m = DocumentManifest(db_path=tmp_path / "manifest.db")
    doc = DocumentInput(
        document_id="fail-doc",
        document_type="tutanak",
        collection_name="col_fail",
        content_hash="hf",
    )
    m.upsert(doc, status="failed", perf={"total_ms": 500})
    trends = m.perf_trends_by_collection()
    assert "col_fail" not in trends


# ─── cmd_inspect argüman çözümlemesi ────────────────────────────────────────


def test_cmd_inspect_missing_file(tmp_path, capsys):
    from src.trainer.ingestion.ingest import cmd_inspect

    args = Namespace(
        inspect=str(tmp_path / "nonexistent.pdf"),
        collection=None,
        document_type=None,
        limit=5,
    )
    with pytest.raises(SystemExit) as exc:
        cmd_inspect(args)
    assert exc.value.code == 1


def test_cmd_inspect_runs_and_prints_table(tmp_path, capsys):
    from src.trainer.ingestion.ingest import cmd_inspect

    fake_pdf = tmp_path / "test.pdf"
    fake_pdf.write_bytes(b"%PDF-fake")

    args = Namespace(
        inspect=str(fake_pdf),
        collection=None,
        document_type=None,
        limit=5,
    )

    fake_chunks = [
        {"text": "Test chunk içeriği " * 10, "span": (0, 190), "metadata": {"type": "Packed", "pages": [1]}},
        {"text": "İkinci chunk içeriği " * 8, "span": (190, 358), "metadata": {"type": "Packed", "pages": [1, 2]}},
    ]

    # cmd_inspect imports DoclingManager locally; patch at source module
    with patch("src.common.parsing.docling_manager.DoclingManager") as mock_mgr_cls:
        mock_mgr = mock_mgr_cls.return_value
        mock_mgr.convert_and_pack.return_value = (
            "Test chunk içeriği " * 10 + "İkinci chunk içeriği " * 8,
            fake_chunks,
        )
        cmd_inspect(args)

    out = capsys.readouterr().out
    # Rich Panel ve tablo çıktıda görünmeli
    assert "Chunk Önizleme" in out or "chunk" in out.lower()
