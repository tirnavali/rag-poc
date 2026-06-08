"""Tests for DocumentManifest SQLite operations."""
import pytest
import tempfile
from pathlib import Path

from src.trainer.ingestion.adapters.base import DocumentInput
from src.trainer.ingestion.manifest import DocumentManifest


@pytest.fixture
def manifest():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_manifest.db"
        yield DocumentManifest(db_path=db_path)


def test_empty_manifest(manifest):
    """Yeni manifest boş olmalı."""
    assert manifest.get("nonexistent", "default_collection") is None
    assert manifest.count_by_status() == {}


def test_upsert_and_get(manifest):
    """Belge ekleme ve okuma."""
    doc = DocumentInput(
        document_id="tbmm-20-1-1-19960108",
        document_type="tutanak",
        collection_name="tbmm_minutes_docling_jina_v3",
        document_source="test.pdf",
        content_hash="abc123",
        document_date="1996-01-08",
        period=20,
        legislative_year=1,
        session=1,
        author="Mustafa Kalemli",
        source_name="TBMM Tutanakları",
        title="6. Birleşim",
        metadata={"acilma_saati": "15:00"},
    )
    manifest.upsert(doc, status="done", chunk_count=5)
    
    record = manifest.get("tbmm-20-1-1-19960108", "tbmm_minutes_docling_jina_v3")
    assert record is not None
    assert record.document_id == "tbmm-20-1-1-19960108"
    assert record.status == "done"
    assert record.chunk_count == 5
    assert record.period == 20
    assert record.author == "Mustafa Kalemli"
    assert record.source_name == "TBMM Tutanakları"


def test_diff(manifest):
    """Diff: new, changed, unchanged detection."""
    existing = DocumentInput(
        document_id="existing-doc",
        document_type="tutanak",
        collection_name="tbmm_minutes_docling_jina_v3",
        content_hash="hash-v1",
    )
    manifest.upsert(existing, status="done")
    
    new_doc = DocumentInput(
        document_id="new-doc",
        document_type="tutanak",
        collection_name="tbmm_minutes_docling_jina_v3",
        content_hash="hash-new",
    )
    changed_doc = DocumentInput(
        document_id="existing-doc",
        document_type="tutanak",
        collection_name="tbmm_minutes_docling_jina_v3",
        content_hash="hash-v2",
    )
    unchanged_doc = DocumentInput(
        document_id="existing-doc",
        document_type="tutanak",
        collection_name="tbmm_minutes_docling_jina_v3",
        content_hash="hash-v1",
    )
    
    new, changed, unchanged = manifest.diff([new_doc, changed_doc, unchanged_doc])
    
    assert len(new) == 1 and new[0].document_id == "new-doc"
    assert len(changed) == 1 and changed[0].document_id == "existing-doc"
    assert len(unchanged) == 1 and unchanged[0].document_id == "existing-doc"


def test_delete(manifest):
    """Belge silme."""
    doc = DocumentInput(
        document_id="delete-me",
        document_type="tutanak",
        collection_name="tbmm_minutes_docling_jina_v3",
        content_hash="hash",
    )
    manifest.upsert(doc, status="done")
    assert manifest.get("delete-me", "tbmm_minutes_docling_jina_v3") is not None
    
    manifest.delete("delete-me", "tbmm_minutes_docling_jina_v3")
    assert manifest.get("delete-me", "tbmm_minutes_docling_jina_v3") is None


def test_count_by_status(manifest):
    """Durum bazında sayım."""
    for i, status in enumerate(["done", "done", "failed", "pending"]):
        doc = DocumentInput(
            document_id=f"doc-{i}",
            document_type="tutanak",
            collection_name="tbmm_minutes_docling_jina_v3",
            content_hash=f"hash-{i}",
        )
        manifest.upsert(doc, status=status)
    
    counts = manifest.count_by_status()
    assert counts.get("done") == 2
    assert counts.get("failed") == 1
    assert counts.get("pending") == 1


def test_list_by_collection(manifest):
    """Koleksiyon bazında listeleme."""
    doc1 = DocumentInput(
        document_id="doc-1",
        document_type="tutanak",
        collection_name="tbmm_minutes_docling_jina_v3",
        content_hash="h1",
    )
    doc2 = DocumentInput(
        document_id="doc-2",
        document_type="press_clip",
        collection_name="gazete_arsivi",
        content_hash="h2",
    )
    manifest.upsert(doc1, status="done")
    manifest.upsert(doc2, status="done")
    
    minutes = manifest.list_by_collection("tbmm_minutes_docling_jina_v3")
    assert len(minutes) == 1
    assert minutes[0].document_id == "doc-1"
    
    press = manifest.list_by_collection("gazete_arsivi")
    assert len(press) == 1
    assert press[0].document_id == "doc-2"


if __name__ == "__main__":
    pytest.main([__file__])
