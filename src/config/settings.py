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

# Press clip chunking — TOKEN cinsinden (ingestion pipeline, token-tabanlı).
# Chunk boyutu koleksiyonun max_chunk_tokens'ından gelir; bu yalnız örtüşmedir.
PRESS_CHUNK_OVERLAP_TOKENS = 64
# Legacy/cosmetic (eski press_clips/index.py ve chunk_inspector karakter gösterimi):
PRESS_CHUNK_SIZE = 1500
PRESS_CHUNK_OVERLAP = 150
MINUTES_CHUNK_SIZE = 1500
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
RERANK_MODEL = os.environ.get("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
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
PARSE_CACHE_DIR = DATA_LAKE / "parse_cache"   # iç önbellek (opak MD5 anahtarlı)
# Aşama bazlı, okunabilir ({stem}__{hash8}) artefakt dizinleri — gözlemlenebilirlik/QC için.
MARKDOWN_DIR = DATA_LAKE / "markdown"          # 1 — tam markdown (.md)
ATOMS_DIR = DATA_LAKE / "atoms"                # 2 — docling atomları (_atoms.json)
PACKED_ATOMS_DIR = DATA_LAKE / "packed_atoms"  # 3 — paketlenmiş chunk'lar (_packed.json)
PAGES_DIR = DATA_LAKE / "pages"                # 4 — sayfa bazlı markdown (_pages.json)
REPORTS_DIR = DATA_LAKE / "reports"            # uçuş kaydedici + artifact index ({document_id}.json)

# --- Downloaded Files ---
# Files fetched from URLs during ingestion are cached here per collection.
DOWNLOADS_DIR = DATA_LAKE / "downloads"

# --- Tier-1 OCR Quality (src/common/parsing/quality.py) ---
# Sayfa başına atom sayısı bu eşiğin altındaysa "low_atom_density" bayrağı.
QUALITY_MIN_ATOMS_PER_PAGE = 3.0
# Sayfa başına karakter, aynı document_type ortalamasından bu oranın üzerinde
# sapıyorsa "char_count_outlier" bayrağı.
QUALITY_CHAR_DEVIATION_RATIO = 0.30
# Ortalama OCR güveni bu eşiğin altındaysa "low_ocr_confidence" bayrağı.
QUALITY_MIN_OCR_CONFIDENCE = 0.85
# Karakter sapması kontrolü için tip başına gereken minimum diğer belge sayısı.
QUALITY_STATS_MIN_DOCS = 3
# document_type bazlı karakter istatistiklerinin biriktiği dosya.
QUALITY_STATS_FILE = PARSE_CACHE_DIR / "quality_stats.json"

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

# --- VLM Table Extraction Configuration ---
# Taranmış/döndürülmüş tabloları (Docling'in TableFormer'ının yapı çıkaramadığı,
# OCR'ın çöp ürettiği tablolar) yerel bir Ollama görü-dil modeliyle (qwen2.5vl) okur.
# Yalnızca "bozuk" tablolarda otomatik devreye girer (low-quality tetikleyici);
# düzgün okunan tablolara dokunmaz. Akış: yüksek-çözünürlük kırp → dik çevir
# (Tesseract OSD) → gerekirse satır-bantlarına böl → VLM ile markdown'a çevir.
# Devre dışı bırakmak için: VLM_TABLE_EXTRACTION=0
VLM_TABLE_EXTRACTION = os.environ.get("VLM_TABLE_EXTRACTION", "1") not in ("0", "false", "False")
# Ollama'da kurulu görü-dil modeli. 7b, 32b kadar doğru (bütçe toplamları bire bir
# doğrulandı) ama bu makinede ~4× hızlı. Daha yüksek doğruluk için: VLM_TABLE_MODEL=qwen2.5vl:32b
VLM_TABLE_MODEL = os.environ.get("VLM_TABLE_MODEL", "qwen2.5vl:7b")
# Tablo bbox'ının PDF'ten kırpılırken kullanılacak SABİT render ölçeği — yalnızca
# adaptif zoom hesaplanamazsa fallback olarak kullanılır (aşağıya bkz).
VLM_TABLE_ZOOM = float(os.environ.get("VLM_TABLE_ZOOM", "3.0"))
# Adaptif zoom: kırpılan bölgenin uzun kenarını bu piksele getirecek ölçek hesaplanır
# (clamp MIN..MAX). Farklı çözünürlükteki/küçültülmüş tablolar normalize edilir → hücre
# başına yeterli piksel. uzun_kenar_pt > 0 değilse VLM_TABLE_ZOOM'a düşülür.
VLM_TABLE_TARGET_LONG_PX = int(os.environ.get("VLM_TABLE_TARGET_LONG_PX", "2400"))
VLM_TABLE_MIN_ZOOM = float(os.environ.get("VLM_TABLE_MIN_ZOOM", "2.0"))
VLM_TABLE_MAX_ZOOM = float(os.environ.get("VLM_TABLE_MAX_ZOOM", "6.0"))
# Kırpmadan önce tablo bbox'ı sayfa boyutunun bu kesiri kadar dışa genişletilir →
# cetvel başlığı/üst-başlık ve alt not kırpıntıya girer (aksi halde bbox onları keser).
VLM_TABLE_BBOX_PAD_FRAC = float(os.environ.get("VLM_TABLE_BBOX_PAD_FRAC", "0.07"))
# Ollama bağlam ve üretim limitleri. Varsayılan KvSize (8192) dev tablolarda
# taşıp 500 döndürdüğü için bağlam belirgin biçimde büyütülür (GPU unified memory).
# Tek çağrıda tipik bir tablonun tamamı (~50 satır) okunabilsin diye bağlam geniş;
# 16384 görsel+çıktıyı rahat alır ve varsayılan 8192 KV taşmasını (500) önler.
VLM_TABLE_NUM_CTX = int(os.environ.get("VLM_TABLE_NUM_CTX", "16384"))
# Üretim üst sınırı (cap; kullanılmazsa maliyet yok). ~50 satırlık tam tablo ölçümde
# ~2400 token üretti; 8192 truncation'ı önler.
VLM_TABLE_NUM_PREDICT = int(os.environ.get("VLM_TABLE_NUM_PREDICT", "8192"))
# Tek bir VLM çağrısı için HTTP zaman aşımı (sn). 32B model dilim başına dakikalar sürebilir.
VLM_TABLE_TIMEOUT = int(os.environ.get("VLM_TABLE_TIMEOUT", "900"))
# Kırpıntıyı VLM'e vermeden önce Tesseract OSD ile otomatik dik çevirme.
VLM_TABLE_AUTOROTATE = os.environ.get("VLM_TABLE_AUTOROTATE", "1") not in ("0", "false", "False")
# OSD yön güveni bu eşiğin altındaysa OSD açısına güvenilmez; aspect-ratio fallback'e düşülür.
VLM_TABLE_OSD_MIN_CONFIDENCE = float(os.environ.get("VLM_TABLE_OSD_MIN_CONFIDENCE", "1.0"))
# VARSAYILAN tek-çağrıdır (en tutarlı sütun hizası). Bu değer YALNIZCA --band ile
# bantlama zorlandığında bant yüksekliği olarak kullanılır (her bant ~bu kadar piksel).
VLM_TABLE_TILE_MAX_HEIGHT_PX = int(os.environ.get("VLM_TABLE_TILE_MAX_HEIGHT_PX", "400"))
# Bantlar arası dikey örtüşme (px) — satırların seam'de yarıdan kesilmesini önler.
VLM_TABLE_TILE_OVERLAP_PX = int(os.environ.get("VLM_TABLE_TILE_OVERLAP_PX", "50"))
# Bantlamada 2..N bantların üstüne eklenen başlık şeridi yüksekliği (px). Her bant
# sütun başlıklarını görür → başlıksız bantlardaki sütun kayması önlenir. 0 = kapalı.
VLM_TABLE_HEADER_STRIP_PX = int(os.environ.get("VLM_TABLE_HEADER_STRIP_PX", "95"))
# Paralel bant gönderimleri için iş parçacığı sayısı. Gerçek eşzamanlılık için
# Ollama'nın OLLAMA_NUM_PARALLEL değeri bu sayıya eşit veya daha büyük olmalı.
VLM_TABLE_MAX_WORKERS = int(os.environ.get("VLM_TABLE_MAX_WORKERS", "5"))

# --- Tesseract Table Extraction Configuration (klasik OCR backend) ---
# VLM'e (qwen2.5vl) alternatif, hızlı klasik backend: gömülü/döndürülmüş tabloları
# OpenCV + Tesseract ile okur. Detaylı plan: docs/tesseract_table_extraction_plan.md
# Backend seçimi: vlm | tesseract | paddleocr
TABLE_EXTRACTOR = os.environ.get("TABLE_EXTRACTOR", "vlm").strip().lower()
# OCR dili (kurulu: tur, eng, osd) ve motor (1 = LSTM).
TESS_TABLE_LANG = os.environ.get("TESS_TABLE_LANG", "tur+eng")
TESS_TABLE_OEM = int(os.environ.get("TESS_TABLE_OEM", "1"))
# Tablo bütünü için varsayılan PSM. 6 = uniform block (satır yapısını korur → grid
# reconstruction için en uygun). Alternatif: 4 (tek sütun), 11 (sparse text).
TESS_TABLE_PSM = int(os.environ.get("TESS_TABLE_PSM", "6"))
# Tek bir tesseract çağrısı için zaman aşımı (sn). VLM'in aksine saniyeler mertebesinde.
TESS_TABLE_TIMEOUT = int(os.environ.get("TESS_TABLE_TIMEOUT", "120"))
# Yön tespiti (orientation): kararlar bu uzun-kenar pikseline küçültülmüş "probe"
# üzerinde verilir (4× trial-OCR ucuz kalsın). Nihai dönüşüm tam çözünürlükte uygulanır.
TESS_TABLE_ORIENT_PROBE_LONG_PX = int(os.environ.get("TESS_TABLE_ORIENT_PROBE_LONG_PX", "1600"))
# Trial-OCR (yön oylaması) sırasında kullanılan PSM — hız için 6.
TESS_TABLE_TRIAL_PSM = int(os.environ.get("TESS_TABLE_TRIAL_PSM", "6"))
# OSD yön güveni bu eşiğin altındaysa OSD'ye tek başına güvenilmez (bu belgelerde OSD
# ~0.04 dönüyor); trial-OCR oylaması belirleyici olur. VLM eşiğinden (1.0) düşük.
TESS_TABLE_OSD_MIN_CONFIDENCE = float(os.environ.get("TESS_TABLE_OSD_MIN_CONFIDENCE", "0.5"))
# Hızlı yol: OSD güveni bu eşiği aşarsa (fitz-render'da ~2.3) 4× trial-OCR atlanır →
# yön tespiti ~11s'den ~3s'ye iner. Güven düşükse (ham/native görüntü ~0.04) yine
# trial-OCR oylamasına düşülür (güvenli). 0 = fast-path kapalı.
TESS_TABLE_OSD_TRUST_CONFIDENCE = float(os.environ.get("TESS_TABLE_OSD_TRUST_CONFIDENCE", "2.0"))

# --- PaddleOCR VL API Configuration ---
# OpenAI-uyumlu endpoint (LiteLLM proxy). --vlm paddleocr ile etkinleşir.
PADDLE_OCR_URL = os.environ.get("PADDLE_OCR_URL", "http://10.20.24.16:4000/v1")
PADDLE_OCR_MODEL = os.environ.get("PADDLE_OCR_MODEL", "paddleocr-vl-1.6")
PADDLE_OCR_API_KEY = os.environ.get("PADDLE_OCR_API_KEY", "none")
PADDLE_OCR_TIMEOUT = int(os.environ.get("PADDLE_OCR_TIMEOUT", "120"))
# Trial-OCR'da en iyi açı, ikinciyi bu orandan fazla geçerse "net kazanan" sayılır;
# aksi halde (belirsiz) OSD / aspect / çizgi ipuçları tie-breaker olur.
TESS_TABLE_ORIENT_MARGIN = float(os.environ.get("TESS_TABLE_ORIENT_MARGIN", "0.15"))

# --- İterasyon 2/3: yüksek-çözünürlük render + çizgi temizleme + grid ---
# Tablo bölgesi bu uzun-kenar pikseline gelecek ölçekte render edilir. fitz adaptif
# zoom'unun (uzun kenar 2400) aksine yüksek tutulur: geniş bütçe tablosu portre
# çerçeveye sıkıştığında sütunlar daralıp rakamlar bozuluyordu (ölçümde mean_conf
# 36→70 sadece çizgi temizleme + yüksek çözünürlükle).
TESS_TABLE_RENDER_LONG_PX = int(os.environ.get("TESS_TABLE_RENDER_LONG_PX", "5000"))
# Render ölçeği bu değerle sınırlanır (çok küçük bbox'ta aşırı büyümeyi önler).
TESS_TABLE_RENDER_MAX_ZOOM = float(os.environ.get("TESS_TABLE_RENDER_MAX_ZOOM", "16.0"))
# Çıktı biçimi: 1 = grid → markdown tablo (İter 3), 0 = satır-gruplu düz metin (İter 1).
TESS_TABLE_GRID = os.environ.get("TESS_TABLE_GRID", "1") not in ("0", "false", "False")
# Sütun sınırı tespiti: x-kapsama boşluğu sayfa genişliğinin bu kesirinden genişse
# sütun ayracı sayılır. Çok küçük → gürültü; çok büyük → komşu sütunları birleştirir.
TESS_TABLE_COL_MIN_GAP_FRAC = float(os.environ.get("TESS_TABLE_COL_MIN_GAP_FRAC", "0.012"))

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
USE_LOCAL_LATE_CHUNKING = os.environ.get("USE_LOCAL_LATE_CHUNKING", "1") == "1"
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
# Enumeration/exhaustive intent → 'comprehensive' query_type: gather many chunks
# across iterative retrieval rounds (single large context). Matched case-insensitively
# as substrings against the raw query.
COMPREHENSIVE_KEYWORDS = [
    "tüm", "tum", "bütün", "butun", "hepsi", "tamamı", "tamami",
    "listele", "liste halinde", "sırala", "sirala",
    "kaç tane", "kac tane", "kaç adet", "kac adet", "kaç defa", "kac defa",
    "hangileri", "hangi hangi", "her biri", "her bir",
    "tek tek", "madde madde", "say bakalım", "hepsini",
]

# --- Parliamentary Jargon Terms (colloquial → official-synonym expansion) ---
# Domain vocabulary the small planner/classifier LLMs sometimes miss. Example:
# 'kadük' (bir yasama döneminde sonuçlandırılamayan teklif) rarely matches the
# corpus's own phrasing ("hükümsüz sayılan kanun teklifleri") in embedding
# space — verified empirically: searching "kadük tüm listeyi ver" surfaces
# unrelated tables, while searching "hükümsüz sayılan kanun teklifleri" (same
# filters) correctly surfaces İçtüzük MADDE 77 and the actual lapsed-bill lists.
# PREPENDING the official synonym phrase to the search query (see
# src.common.text.expand_parliamentary_synonyms) closes this gap deterministically
# (no LLM dependency) — empirically, prepending clearly outperforms appending
# (0/8 relevant hits appended vs. 4-6/8 prepended in the same period-filtered
# search), and the full official phrase ("... kanun teklifleri") outperforms the
# bare word ("hükümsüz sayılan" alone scored 0/8 — needs the full noun phrase).
# The same keys double as a scope-classifier safety net
# (OrchestratorAgent._is_known_parliamentary_term): a known term in the query
# forces scope back to in_scope even when the classifier mistakes a jargon
# definition question ("kadük ne demek?") for an off-domain dictionary lookup.
PARLIAMENTARY_TERM_SYNONYMS: dict[str, list[str]] = {
    "kadük": ["hükümsüz sayılan kanun teklifleri", "hükümsüz sayılır"],
}

# --- Default collection for RAGService ---
# Used when RAGService() is instantiated without explicit spec.
# Override with RAG_DEFAULT_COLLECTION env var.
DEFAULT_COLLECTION = os.environ.get("RAG_DEFAULT_COLLECTION", "tutanaklar_ctx1024")

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
