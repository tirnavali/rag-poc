"""Centralized configuration for the RAG pipeline.

All magic constants (paths, model names, thresholds) live here so the rest of
the codebase can stay free of environment-specific assumptions. Paths are
resolved absolutely from the project root, which prevents the relative-path
bug where ChromaDB silently creates an empty directory when a script is
invoked from an unexpected working directory.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

_logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_LAKE = PROJECT_ROOT / "data_lake"

# --- .env Dosyasını Yükleme (Sıfır Bağımlılık) ---
_env_file = PROJECT_ROOT / ".env"
if _env_file.exists():
    with open(_env_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, val = line.split("=", 1)
                os.environ.setdefault(key.strip(), val.strip())

# --- Model & Environment Configuration ---
# Options: 'local', 'remote'
RAG_ENV = os.environ.get("RAG_ENV", "local")

OLLAMA_ENVIRONMENTS = {
    "local": {
        "host": "http://localhost:11434",
        "llm": "gemma4:e2b",
        "embed": "nomic-embed-text-v2-moe",
    },
    "remote": {
        "host": os.environ.get("REMOTE_OLLAMA_HOST", "http://172.20.0.143:11434"),
        "llm": "gpt-oss:20b",
        "embed": "nomic-embed-text-v2-moe",
    }
}

_env_config = OLLAMA_ENVIRONMENTS.get(RAG_ENV, OLLAMA_ENVIRONMENTS["local"])

# --- Dynamic Load from pipeline.yaml as Single Source of Truth ---
_pipeline_llm = None
_pipeline_filter_llm = None
_pipeline_host = None

_pipeline_yaml_path = PROJECT_ROOT / "pipeline.yaml"
if _pipeline_yaml_path.exists():
    try:
        import yaml
        with open(_pipeline_yaml_path, "r", encoding="utf-8") as f:
            _raw_pipeline = yaml.safe_load(f)
        if isinstance(_raw_pipeline, dict):
            _blocks = _raw_pipeline.get("deployment_blocks", {})
            _agent_cfg = _raw_pipeline.get("agent", {})
            
            # 1. Resolve Answering Agent details
            _answering_cfg = _agent_cfg.get("answering", {})
            _answering_block_name = _answering_cfg.get("block", "gpu-01")
            _answering_model_key = _answering_cfg.get("model_key", "answer")
            _answering_block = _blocks.get(_answering_block_name, {})
            
            _pipeline_host = _answering_block.get("host")
            _answering_models = _answering_block.get("models", {})
            _pipeline_llm = _answering_models.get(_answering_model_key)
            
            # 2. Resolve Filter Extractor Agent details
            _filter_cfg = _agent_cfg.get("filter_extractor", {})
            _filter_block_name = _filter_cfg.get("block", "fast-01")
            _filter_model_key = _filter_cfg.get("model_key", "filter_extractor")
            _filter_block = _blocks.get(_filter_block_name, {})
            _filter_models = _filter_block.get("models", {})
            _pipeline_filter_llm = _filter_models.get(_filter_model_key)
    except Exception as e:
        _logger.debug(f"Failed to load pipeline.yaml configuration: {e}. Using environment/default fallback.")

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", _pipeline_host or _env_config["host"])
LLM_MODEL = os.environ.get("RAG_LLM_MODEL", _pipeline_llm or _env_config["llm"])
EMBED_MODEL = os.environ.get("RAG_EMBED_MODEL", _env_config["embed"])

# --- Filter Extraction Model ---
# FilterExtractor does structured JSON extraction (low reasoning need).
# Decoupled from LLM_MODEL so the heavyweight generation model is unaffected.
# qwen2.5:3b-instruct: non-reasoning instruction model — no thinking-mode latency.
FILTER_LLM_MODEL = os.environ.get("RAG_FILTER_LLM_MODEL", _pipeline_filter_llm or "qwen3.5:9b")

# --- Author Transition Cleaning Model ---
# llm_transition_cleaner.py: OCR error correction for speaker names/roles.
# Lightweight task (name/role extraction from short text) — llama3:8b sufficient.
# Decoupled from LLM_MODEL to avoid blocking generation on small corrections.
AUTHOR_TRANSITION_CLEAN_MODEL = os.environ.get("RAG_AUTHOR_CLEAN_MODEL", "llama3:8b")

PRESS_SQLITE = DATA_LAKE / "press_clips.db"
PRESS_CHROMA = DATA_LAKE / "press_clips_vectors"
PRESS_COLLECTION = "gazete_arsivi"
PRESS_CSV = PROJECT_ROOT / "gazete-rag-001.csv"

MINUTES_SQLITE = DATA_LAKE / "parliament_digital_born_minutes.db"
MINUTES_CHROMA = DATA_LAKE / "parliament_digital_born_minutes_vectors"
MINUTES_COLLECTION = "tbmm_minutes"
MINUTES_JSON_DIR = PROJECT_ROOT / "tutanak" / "extracted"

PRESS_CHUNK_SIZE = 1500
PRESS_CHUNK_OVERLAP = 150
MINUTES_CHUNK_SIZE = 1500
MINUTES_CHUNK_OVERLAP = 150
# Greedy speech-block packing thresholds (used by src/trainer/minutes/chunker.py).
# Short consecutive speaker turns are packed into one chunk so that a 5-word
# interjection is not stored as its own near-empty embedding.
MINUTES_MIN_CHUNK_CHARS = 400
MINUTES_TARGET_CHUNK_CHARS = 1500
MINUTES_PACK_CAP_CHARS = 1950
EMBED_BATCH_SIZE = 20

RRF_K = 60
DISTANCE_THRESHOLD = 1.8
RETRIEVE_TOP_K = 5
RETRIEVE_FETCH_K = 100
MUFETTIS_TOP_K = 40
MUFETTIS_FETCH_K = 150
FTS_LIMIT = 15

RETRIEVAL_MODE = os.environ.get("RETRIEVAL_MODE", "hybrid")  # "hybrid" | "vector"
USE_RERANKER = os.environ.get("USE_RERANKER", "1") == "1"
RERANK_MODEL = os.environ.get("RERANK_MODEL", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
RERANK_FETCH_K = int(os.environ.get("RERANK_FETCH_K", "100"))
RERANK_COARSE_K = int(os.environ.get("RERANK_COARSE_K", "20"))
RERANK_FINAL_K = int(os.environ.get("RERANK_FINAL_K", "5"))

CONTEXT_MAX_CHARS = 4000
CONTEXT_TOTAL_MAX = 12000
MUFETTIS_CONTEXT_MAX_CHARS = 8000
MUFETTIS_CONTEXT_TOTAL_MAX = 25000
CONTEXT_BUILD_DEFAULT_MAX = 4000
CONTEXT_BUILD_DEFAULT_TOTAL = 6000

WINDOW_SIZE = 800
WINDOW_MAX_TOTAL = 3000

LLM_NUM_CTX = 32768
LLM_TEMPERATURE_DEFAULT = 0.1
LLM_TEMPERATURE_MUFETTIS = 0.2
LLM_TEMPERATURE_EXPAND = 0.3
LLM_NUM_PREDICT_DEFAULT = 4096
LLM_NUM_PREDICT_MUFETTIS = 16384

DEBUG_RAG = os.environ.get("DEBUG_RAG", "0") == "1"

# --- Onerge (Kanun Teklifi) Configuration ---
ONERGE_CHROMA = DATA_LAKE / "onerge_vectors"
ONERGE_COLLECTION = "tbmm_onerge"

# --- Document Manifest Configuration ---
MANIFEST_DB = DATA_LAKE / "document_manifest.db"
PARSE_CACHE_DIR = DATA_LAKE / "parse_cache"
MARKDOWN_DIR = DATA_LAKE / "markdown"

# --- Downloaded Files ---
# Files fetched from URLs during ingestion are cached here per collection.
DOWNLOADS_DIR = DATA_LAKE / "downloads"

# --- Docling OCR Configuration ---
# Engine options: "easyocr", "tesseract", "mac"
# Override via env: OCR_ENGINE=tesseract python -m scripts.ingest ...
OCR_ENGINE = os.environ.get("OCR_ENGINE", "easyocr")

# GPU/CPU configuration for Docling
_use_gpu_env = os.environ.get("DOCLING_USE_GPU", "auto").lower()
if _use_gpu_env == "true":
    DOCLING_USE_GPU = True
elif _use_gpu_env == "false":
    DOCLING_USE_GPU = False
else:
    # "auto" or empty: autodetect CUDA capability
    try:
        import torch
        DOCLING_USE_GPU = torch.cuda.is_available()
    except ImportError:
        DOCLING_USE_GPU = False

# --- Local Late Chunking Configuration ---
#
# LATE CHUNKING NEDİR?
# Standart embedding: her chunk bağımsız olarak embed edilir → chunk, dokümanın
# geri kalanından habersizdir. Bağlamı kopuk, anlam kaybı yaşanır.
#
# Late chunking: önce TÜM doküman tek seferde modele verilir (full-context
# encoding). Ardından her chunk'a denk gelen token'ların embedding'leri
# mean-pool edilerek o chunk'ın vektörü hesaplanır. Sonuç: her chunk vektörü
# dokümanın bütününden gelen bağlamı taşır.
#
# "LOCAL" NE DEMEK?
# Jina v3'ün bulut API'si yerine modeli HuggingFace üzerinden local'de
# indirip çalıştırıyoruz. İnternet bağlantısı gerekmez (ilk indirmeden sonra),
# veri dışarı çıkmaz.
#
# NEDEN JİNA v3?
# • 8192 token context window (nomic 2048 token)
# • Türkçe dahil çok dilli destek
# • task_type parametresiyle retrieval/classification görevleri için optimize
#
# TRADE-OFF:
# • Yavaş: GPU yoksa CPU'da dakikalar sürer (default KAPALI bu yüzden)
# • Bellek: ~570MB model + batch tokenization RAM'i
# • Jina v4 (32k context) için: JINA_LOCAL_MODEL=jinaai/jina-embeddings-v4
#
# AKTİF ETMEK İÇİN:
#   USE_LOCAL_LATE_CHUNKING=1 python -m src.trainer.ingestion.ingest --request manifest.json
JINA_LOCAL_MODEL = os.environ.get("JINA_LOCAL_MODEL", "jinaai/jina-embeddings-v3")
USE_LOCAL_LATE_CHUNKING = os.environ.get("USE_LOCAL_LATE_CHUNKING", "0") == "1"
# NOT: max_context_tokens ve overlap_tokens artık model bazında
# src/config/collections.py içindeki MODEL_SPECS'ten çekiliyor.
# Bu sayede Jina v3 (8K), Jina v4 (32K), Nomic v2 (512) gibi farklı
# modeller aynı anda kullanılabilir — global tek bir değer yok.

MINUTES_KEYWORDS = [
    "tutanak", "tutanaklar", "meclis", "mecliste", "tbmm",
    "parlamento", "genel kurul", "milletvekili",
]
ONERGE_KEYWORDS = [
    "kanun teklifi", "kanun teklifleri",
    "önerge", "önergeler", "teklif", "teklifler",
    "yasa teklifi", "yasa teklifleri",
    "tbmm kanun teklifi", "tbmm önerge",
]
PUBLICATION_KEYWORDS = [
    "haber", "haberler", "gazete", "gazeteler",
    "basın", "basin", "medya", "basılı medya", "basin medya",
    "köşe yazısı", "kose yazisi", "köşe yazıları", "kose yazilari",
    "makale", "makaleler", "röportaj", "reportaj",
    "manşet", "manset", "manşetler", "mansetler",
    "kupür", "kupurler", "gazete kupürü",
    "muhabir", "editör", "editor", "yayın", "yayin",
    "yayınlar", "yayinlar", "dergi",
]

# --- Default collection for RAGService ---
# Used when RAGService() is instantiated without explicit spec.
# Override with RAG_DEFAULT_COLLECTION env var.
DEFAULT_COLLECTION = os.environ.get("RAG_DEFAULT_COLLECTION", "tbmm_tutanaklar_nomic_v2")

# --- Author Metadata Validator (LLM backstop) ---
# Runs only on chunks where regex-based author extraction failed.
# Disabled by default — enable with AUTHOR_VALIDATOR_ENABLED=1 for OCR-noisy docs.
AUTHOR_VALIDATOR_ENABLED = os.environ.get("AUTHOR_VALIDATOR_ENABLED", "0") == "1"
AUTHOR_VALIDATOR_PREV_CHARS = int(os.environ.get("AUTHOR_VALIDATOR_PREV_CHARS", "200"))

AUTHOR_VALIDATOR_PROMPTS: dict[str, str] = {
    "tutanak": (
        "Sen TBMM tutanak analisti uzmanısın. Aşağıdaki önceki bağlama ve mevcut "
        "chunk metnine bakarak, mevcut chunk'taki konuşan kişinin adını çıkar.\n\n"
        "Konuşmacı net değilse \"BİLİNMİYOR\" döndür.\n\n"
        "Önceki bağlam:\n{prev}\n\nMevcut chunk:\n{chunk}\n\n"
        "JSON formatında çıktı ver: "
        "{{\"author\": \"...\", \"author_role\": \"...\", \"confidence\": 0.0-1.0}}"
    ),
    "gazete": (
        "Sen gazete arşiv analisti uzmanısın. Aşağıdaki önceki bağlam ve mevcut "
        "küpür metnine bakarak yazar veya muhabir adını çıkar.\n\n"
        "Yazar net değilse \"BİLİNMİYOR\" döndür.\n\n"
        "Önceki bağlam:\n{prev}\n\nMevcut chunk:\n{chunk}\n\n"
        "JSON: {{\"author\": \"...\", \"author_role\": \"...\", \"confidence\": 0.0-1.0}}"
    ),
    "press_clip": (
        "Sen gazete arşiv analisti uzmanısın. Yazar veya muhabir adını çıkar.\n\n"
        "Önceki bağlam:\n{prev}\n\nMevcut chunk:\n{chunk}\n\n"
        "JSON: {{\"author\": \"...\", \"author_role\": \"...\", \"confidence\": 0.0-1.0}}"
    ),
    "onerge": (
        "Sen TBMM önerge analisti uzmanısın. Önerge sahibi veya imzacısını çıkar.\n\n"
        "Önceki bağlam:\n{prev}\n\nMevcut chunk:\n{chunk}\n\n"
        "JSON: {{\"author\": \"...\", \"author_role\": \"...\", \"confidence\": 0.0-1.0}}"
    ),
    "kanun_teklifi": (
        "Sen TBMM kanun teklifi analisti uzmanısın. Teklif sahibini çıkar.\n\n"
        "Önceki bağlam:\n{prev}\n\nMevcut chunk:\n{chunk}\n\n"
        "JSON: {{\"author\": \"...\", \"author_role\": \"...\", \"confidence\": 0.0-1.0}}"
    ),
}

AUTHOR_TRANSITION_CLEAN_PROMPTS: dict[str, str] = {
    "tutanak": (
        "Görev: TBMM tutanak konuşmacı başlığındaki OCR hatalarını düzelt.\n\n"
        "Girdi:\n"
        "Ham metin: {raw_head}\n"
        "Tespit edilen ad: {detected_name}\n"
        "Tespit edilen rol: {detected_role}\n\n"
        "Kurallar:\n"
        "1. OCR intra-kelime boşluklarını düzelt (örn. 'ERDO GAN' → 'ERDOĞAN')\n"
        "2. Unvanı addan ayır (unvan ad içine karışmışsa)\n"
        "3. Emin olmadıysan: author = 'BİLİNMİYOR', confidence = 0.0\n\n"
        "YANIT KESINLIKLE SADECE JSON OLACAK. BAŞKA HİÇBİR METIN EKLEME!\n\n"
        "Örnek çıktı:\n"
        "{{\"author\": \"FATIH ALTAYLI\", \"author_role\": \"gazeteci\", \"confidence\": 0.9}}\n\n"
        "Çıktı:"
    ),
}
