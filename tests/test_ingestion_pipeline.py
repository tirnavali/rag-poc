import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

# Proje kök dizinini ekleyelim
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.config.collections import CollectionSpec
from src.config.document_types import DocumentType
from src.trainer.ingestion.adapters.base import DocumentInput
from src.trainer.ingestion.pipeline import IngestionPipeline


@pytest.fixture
def mock_pipeline_components():
    """Büyük bileşenleri (model yükleyen) mock'layarak testleri hızlandırır."""
    with patch('src.trainer.ingestion.pipeline.DoclingManager') as mock_docling, \
         patch('src.trainer.ingestion.pipeline.LocalLateChunkingEmbedder') as mock_embedder, \
         patch('src.trainer.ingestion.pipeline.open_or_create_collection') as mock_chroma, \
         patch('src.trainer.ingestion.pipeline.DocumentManifest') as mock_manifest:
        
        # ChromaDB mock'u
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_chroma.return_value = (mock_client, mock_collection)
        
        # Embedder mock'u (Jina v3 yüklemesini atlar)
        mock_embedder_inst = mock_embedder.return_value
        
        # Manifest mock'u
        mock_manifest_inst = mock_manifest.return_value
        mock_manifest_inst.get.return_value = None  # Hiçbir belge önceden işlenmemiş
        
        spec = CollectionSpec(
            name="test_collection",
            db_path=Path("/tmp/test_chroma"),
            embed_model="jinaai/jina-embeddings-v3-ctx1024",
            doc_type=DocumentType.TUTANAK,
        )
        pipeline = IngestionPipeline(spec=spec)
        yield pipeline, mock_docling.return_value, mock_embedder_inst, mock_collection, mock_manifest_inst


def test_pipeline_run_document_logic(mock_pipeline_components):
    """Pipeline'ın akış mantığını (parse -> embed -> store) test eder."""
    pipeline, mock_docling, mock_embedder, mock_collection, mock_manifest = mock_pipeline_components
    
    # 1. Mock Dönüş Değerleri
    mock_docling.convert_and_pack.return_value = (
        "Bu bir test metnidir. İkinci cümledir.",
        [
            {"text": "Bu bir test metnidir.", "span": (0, 20), "metadata": {"type": "p", "source": "test_document.pdf"}},
            {"text": "İkinci cümledir.", "span": (21, 37), "metadata": {"type": "p", "source": "test_document.pdf"}}
        ]
    )
    mock_embedder.embed_with_late_chunking_windowed.return_value = [[0.1] * 1024, [0.2] * 1024]

    # 2. DocumentInput oluştur
    doc = DocumentInput(
        document_id="test-doc-001",
        document_type="tutanak",
        collection_name="test_collection",
        document_source="test_document.pdf",
        document_date="1996-01-08",
        period=20,
    )
    
    # Mock adapter'ın compute_content_hash'ini override et
    with patch('src.trainer.ingestion.pipeline.get_adapter') as mock_get_adapter:
        mock_adapter = MagicMock()
        mock_adapter.parse.return_value = (
            "Bu bir test metnidir. İkinci cümledir.",
            [
                {"text": "Bu bir test metnidir.", "span": (0, 20), "metadata": {"type": "p"}},
                {"text": "İkinci cümledir.", "span": (21, 37), "metadata": {"type": "p"}}
            ]
        )
        mock_adapter.compute_content_hash.return_value = "testhash123"
        mock_get_adapter.return_value = mock_adapter
        
        result = pipeline.run_document(doc)

    # 3. Doğrulamalar
    assert result.status == "done"
    assert result.chunk_count == 2
    assert mock_adapter.parse.called
    assert mock_embedder.embed_with_late_chunking_windowed.called
    assert mock_collection.upsert.called
    
    # Metadata temizliğini kontrol et
    args, kwargs = mock_collection.upsert.call_args
    assert len(kwargs['ids']) == 2
    assert len(kwargs['embeddings']) == 2
    # Chunk ID'ler document_id prefix ile başlamalı
    assert kwargs['ids'][0].startswith("test-doc-001_")


def test_pipeline_skip_already_ingested(mock_pipeline_components):
    """Aynı content_hash ile işlenmiş belge atlanmalı."""
    pipeline, _, _, _, mock_manifest = mock_pipeline_components
    
    # Manifest'te zaten var
    existing = MagicMock()
    existing.content_hash = "samehash"
    existing.status = "done"
    mock_manifest.get.return_value = existing
    
    doc = DocumentInput(
        document_id="test-doc-002",
        document_type="tutanak",
        collection_name="test_collection",
        document_source="test_document.pdf",
        content_hash="samehash",
    )
    
    with patch('src.trainer.ingestion.pipeline.get_adapter') as mock_get_adapter:
        mock_adapter = MagicMock()
        mock_adapter.compute_content_hash.return_value = "samehash"
        mock_get_adapter.return_value = mock_adapter
        
        result = pipeline.run_document(doc)
    
    assert result.status == "skipped"
    assert result.reason == "already_ingested"


def test_batch_processing_logic(mock_pipeline_components):
    """Toplu dosya işleme mantığını test eder."""
    pipeline, _, _, _, _ = mock_pipeline_components
    
    docs = [
        DocumentInput(document_id=f"doc-{i}", document_type="tutanak", collection_name="test_collection")
        for i in range(3)
    ]
    
    with patch('src.trainer.ingestion.pipeline.get_adapter') as mock_get_adapter:
        mock_adapter = MagicMock()
        mock_adapter.parse.return_value = ("text", [
            {"text": "chunk", "span": None, "metadata": {}}
        ])
        mock_adapter.compute_content_hash.return_value = "hash"
        mock_get_adapter.return_value = mock_adapter
        
        results = pipeline.run_batch(docs)
    
    assert len(results) == 3
    for r in results:
        assert r.status in ("done", "skipped")


if __name__ == "__main__":
    pytest.main([__file__])
