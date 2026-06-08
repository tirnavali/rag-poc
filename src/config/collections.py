"""Collection registry for RAG experiments — loaded from models.yaml.

Provides a single source of truth that maps a collection name to:
- ChromaDB path & collection name
- The embedding model used to index it (critical: query-time model must match)
- Model-specific context window, overlap, and dimension
- Source type (gazete | minutes | custom)

All model specs and collection definitions live in models.yaml.
To add a new model or collection, edit models.yaml — no Python changes required.

------------------------------------------------------------------
ÖNEMLİ KAVRAMLAR
------------------------------------------------------------------

max_context_tokens
    Modelin BİR SEFERDE kaç token okuyabildiği. Bu "context window" veya
    "sequence length" denen şeydir.

    Örnek:
        • nomic-embed-text-v2-moe  →  512 token
        • jina-embeddings-v3       → 8192 token (8K)
        • jina-embeddings-v4       → 32768 token (32K)

    Eğer doküman bu sınırı aşıyorsa, "windowed late chunking" devreye girer:
    metin örtüşen pencerelere bölünür, her pencerede ayrı encode edilir,
    sonuçlar ortalanır.

embed_dim
    Modelin çıktı vektörünün boyutu. Her metin parçası için üretilen
    embedding kaç sayıdan oluşur?

    Örnek:
        • nomic-embed-text-v2-moe  → 768 boyut
        • jina-embeddings-v3/v4    → 1024 boyut

    embed_dim ile max_context_tokens BAĞIMSIZDIR:
    • Jina v3: 8K context, 1024 dim
    • Jina v4: 32K context, 1024 dim (context büyüdü ama dim aynı!)
    • Nomic v2: 512 context, 768 dim

supports_late_chunking
    Late chunking, modeli TÜM doküman üzerinden çalıştırıp her parçanın
    vektörünü o dokümanın bağlamından üretir. Bunu yapabilmek için model
    "offset_mapping" desteklemeli ve tüm dokümanı tek seferde işleyebilmeli.

    Nomic v2 (512 token) ile late chunking yapmak anlamsızdır — çünkü
    dokümanın %99'u pencere dışında kalır. Bu yüzden nomic için
    supports_late_chunking=False, her chunk bağımsız embed edilir.

overlap_tokens
    Pencereli late chunking'de ardışık pencereler arasındaki örtüşme.
    8K context için 128, 32K context için 256 gibi değerler makuldür.
    Daha büyük overlap → daha yavaş ama pencere kenarlarında daha az
    bilgi kaybı.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from src.config import settings
from src.config.document_types import DocumentType

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_MODELS_YAML = _PROJECT_ROOT / "models.yaml"


# ------------------------------------------------------------------
# Load models.yaml
# ------------------------------------------------------------------

def _load_models_yaml() -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], dict[str, str]]:
    """Load model_specs, collections, and defaults from models.yaml.

    Returns:
        (model_specs_raw, collections_raw, defaults_raw)
    """
    if not _MODELS_YAML.exists():
        raise FileNotFoundError(
            f"models.yaml not found at {_MODELS_YAML}. "
            "This file is required — it defines all embedding models and collections."
        )

    with open(_MODELS_YAML, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    model_specs = raw.get("model_specs", {})
    collections = raw.get("collections", {})
    defaults = raw.get("defaults", {})

    return model_specs, collections, defaults


_MODEL_SPECS_RAW, _COLLECTIONS_RAW, _DEFAULTS_RAW = _load_models_yaml()


# ------------------------------------------------------------------
# MODEL_SPECS — populated from models.yaml
# ------------------------------------------------------------------

MODEL_SPECS: dict[str, dict[str, Any]] = dict(_MODEL_SPECS_RAW)


# ------------------------------------------------------------------
# CollectionSpec
# ------------------------------------------------------------------

@dataclass(frozen=True)
class CollectionSpec:
    """Immutable descriptor for a ChromaDB collection + its indexing model.

    embed_model dışındaki model özellikleri (max_context_tokens, embed_dim,
    vb.) otomatik olarak MODEL_SPECS'ten çekilir.  Eğer MODEL_SPECS'te
    bulunamazsa constructor içinde KeyError atar — böylece bilinmeyen
    modelle koleksiyon oluşturulamaz.
    """

    name: str
    """ChromaDB koleksiyon adı."""

    db_path: Path
    """ChromaDB persistence dizini (mutlak path)."""

    embed_model: str
    """HuggingFace model id veya Ollama model adı."""

    max_context_tokens: int = field(init=False)
    """Modelin bir seferde işleyebileceği maksimum token sayısı.
    Otomatik olarak MODEL_SPECS'ten doldurulur."""

    overlap_tokens: int = field(init=False)
    """Pencereli late chunking'de ardışık pencereler arası örtüşme (token).
    Otomatik olarak MODEL_SPECS'ten doldurulur."""

    embed_dim: int = field(init=False)
    """Çıktı vektör boyutu.
    Otomatik olarak MODEL_SPECS'ten doldurulur."""

    supports_late_chunking: bool = field(init=False)
    """Late chunking desteği var mı?
    Otomatik olarak MODEL_SPECS'ten doldurulur."""

    doc_type: DocumentType = DocumentType.CUSTOM
    """Document source type: GAZETE, TUTANAK, ONERGE, CUSTOM."""

    min_chunk_chars: int = 400
    """Minimum chunk boyutu (karakter). Docling greedy_pack parametresi."""

    max_chunk_chars: int = 1500
    """Maksimum chunk boyutu (karakter). Docling greedy_pack parametresi."""

    max_chunk_tokens: int = 512
    """HybridChunker max token/chunk. embed_model tokenizer ile ölçülür (~1600 char Türkçe)."""

    min_chunk_tokens: int = 384
    """Post-process min token merge eşiği. Bu altındaki chunk'lar bir sonrakiyle birleşir."""

    context_weight: int = 5
    """LLM context'e kaç chunk katılacak (per-collection retrieval)."""

    def __post_init__(self):
        if self.embed_model not in MODEL_SPECS:
            raise KeyError(
                f"Model '{self.embed_model}' MODEL_SPECS'te tanımlı değil. "
                f"Önce models.yaml içinde model_specs'e ekleyin. "
                f"Mevcut modeller: {list(MODEL_SPECS.keys())}"
            )
        spec = MODEL_SPECS[self.embed_model]

        # If this spec is an alias for another model, use the base_model for loading
        if "base_model" in spec:
            object.__setattr__(self, "embed_model", spec["base_model"])

        object.__setattr__(self, "max_context_tokens", spec["max_context_tokens"])
        object.__setattr__(self, "overlap_tokens", spec["overlap_tokens"])
        object.__setattr__(self, "embed_dim", spec["embed_dim"])
        object.__setattr__(self, "supports_late_chunking", spec["supports_late_chunking"])

    def __repr__(self) -> str:
        return (
            f"CollectionSpec(name={self.name!r}, model={self.embed_model!r}, "
            f"context={self.max_context_tokens}, dim={self.embed_dim}, "
            f"late_chunking={self.supports_late_chunking}, doc_type={self.doc_type.value!r})"
        )


# ------------------------------------------------------------------
# Build COLLECTIONS from models.yaml
# ------------------------------------------------------------------

def _build_collections() -> dict[str, CollectionSpec]:
    """Build CollectionSpec instances from models.yaml collection definitions."""
    result = {}
    for key, cfg in _COLLECTIONS_RAW.items():
        doc_type_str = cfg.get("doc_type", "custom")
        try:
            doc_type = DocumentType(doc_type_str)
        except ValueError:
            doc_type = DocumentType.CUSTOM

        # Resolve paths: if relative, make absolute from PROJECT_ROOT
        chroma_path = Path(cfg["chroma_path"])
        if not chroma_path.is_absolute():
            chroma_path = _PROJECT_ROOT / chroma_path

        spec = CollectionSpec(
            name=cfg["collection_name"],
            db_path=chroma_path,
            embed_model=cfg["embed_model"],
            doc_type=doc_type,
            min_chunk_chars=cfg.get("min_chunk_chars", 400),
            max_chunk_chars=cfg.get("max_chunk_chars", 1500),
            max_chunk_tokens=cfg.get("max_chunk_tokens", 512),
            min_chunk_tokens=cfg.get("min_chunk_tokens", 384),
        )
        result[key] = spec
    return result


COLLECTIONS: dict[str, CollectionSpec] = _build_collections()


# ------------------------------------------------------------------
# Default collections for each document type
# ------------------------------------------------------------------

DEFAULT_COLLECTION_FOR_TYPE: dict[DocumentType, str] = {}
for dt in DocumentType:
    if dt.value in _DEFAULTS_RAW:
        DEFAULT_COLLECTION_FOR_TYPE[dt] = _DEFAULTS_RAW[dt.value]


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def get_spec(name: str) -> CollectionSpec:
    """Kayıtlı koleksiyonu ismiyle getir."""
    if name not in COLLECTIONS:
        raise KeyError(
            f"Collection '{name}' kayıtlı değil. "
            f"Mevcut: {list(COLLECTIONS.keys())}"
        )
    return COLLECTIONS[name]


def get_default_spec(doc_type: DocumentType) -> CollectionSpec:
    """Get default collection spec for a document type."""
    collection_name = DEFAULT_COLLECTION_FOR_TYPE.get(doc_type)
    if collection_name is None:
        raise KeyError(
            f"No default collection for document type '{doc_type.value}'. "
            f"Set it in models.yaml under defaults."
        )
    return get_spec(collection_name)


def register_spec(name: str, spec: CollectionSpec) -> None:
    """Runtime'da yeni koleksiyon ekle (ad-hoc deneyler için)."""
    COLLECTIONS[name] = spec


def get_model_specs() -> dict[str, dict[str, Any]]:
    """Return all model specs from models.yaml."""
    return dict(MODEL_SPECS)


def get_collection_names() -> list[str]:
    """Return all registered collection names."""
    return list(COLLECTIONS.keys())


def get_available_collections() -> list[dict[str, Any]]:
    """Return a list of all available collections with metadata and chunk counts.

    Each dict contains:
        - name: ChromaDB collection name
        - type: document type (gazete, tutanak, onerge, custom)
        - embedding_model: the embedding model used
        - count: number of chunks in the collection (0 if missing/broken)
        - spec: the CollectionSpec object

    Results are sorted alphabetically by name.
    """
    import logging
    from chromadb.errors import InvalidCollectionException, NotFoundError

    from src.common.chroma import open_collection

    logger = logging.getLogger(__name__)
    result = []

    for _, spec in COLLECTIONS.items():
        # Try to get chunk count from ChromaDB collection
        chunk_count = 0
        try:
            # ChromaDB client resource; open_collection returns a new client each call.
            # We intentionally discard it (assign to _) because PersistentClient holds no
            # long-lived handles that require explicit cleanup.
            _, collection = open_collection(spec.db_path, spec.name)
            chunk_count = collection.count()
        except (InvalidCollectionException, NotFoundError):
            # Collection does not exist or is inaccessible — default to 0
            pass
        except OSError as e:
            # DB path missing, permission denied, or other filesystem error
            logger.debug(f"Could not access collection {spec.name!r} at {spec.db_path}: {e}")
        except Exception as e:
            # Unexpected error (not a known chromadb/filesystem issue)
            logger.warning(f"Unexpected error accessing collection {spec.name!r}: {e}")

        result.append({
            "name": spec.name,
            "type": spec.doc_type.value,
            "embedding_model": spec.embed_model,
            "count": chunk_count,
            "spec": spec,
        })

    # Sort alphabetically by name
    result.sort(key=lambda x: x["name"])

    return result
