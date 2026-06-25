import os
# Tüm transformers uyarılarını global düzeyde sustur
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
import contextlib
import threading
import torch
import torch.nn.functional as F
import warnings
import logging
from typing import List, Tuple, Dict, Any
from transformers import AutoModel, AutoTokenizer
from transformers.utils import logging as hf_logging
from langchain_core.embeddings import Embeddings

# Jina v3/v4 Flash Attention uyarılarını sustur
warnings.filterwarnings("ignore")
hf_logging.set_verbosity_error()
logging.getLogger("transformers_modules").setLevel(logging.ERROR)

# Thread-safe devnull: modül ömrü boyunca tek bir devnull dosyası açık kalır.
# Kaıpatılmadığı için başka thread'ler "I/O operation on closed file" almaz.
_DEVNULL = open(os.devnull, 'w')  # noqa: WPS515 — intentionally kept open
_SILENCE_LOCK = threading.Lock()


@contextlib.contextmanager
def _silence_stdout_stderr():
    """Jina gibi trust_remote_code modellerin print() ile bastığı
    anlamsız uyarıları (flash_attn is not installed vs.) susturur.
    Sadece model yükleme süresince aktif olur.

    Thread-güvenli: sys.stdout/stderr değiştirme bir kilit altında yapılır
    ve devnull hiç kapatılmaz; böylece eş zamanlı thread'ler kapatılmış
    dosyaya yazmaya çalışmaz.
    """
    with _SILENCE_LOCK:
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = _DEVNULL, _DEVNULL
        try:
            yield
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr

# Monkeypatch transformers DynamicCache to support get_usable_length in newer transformers versions
try:
    from transformers.cache_utils import DynamicCache
    if not hasattr(DynamicCache, "get_usable_length"):
        def get_usable_length(self, seq_len, layer_idx=0):
            return self.get_seq_length(layer_idx)
        DynamicCache.get_usable_length = get_usable_length
except Exception:
    pass


class LocalLateChunkingEmbedder(Embeddings):
    """Jina v3/v4 kullanarak Late Chunking gerçekleştiren yerel gömme sınıfı.

    Trainer katmanına ait: çevrimdışı belge indeksleme için tasarlandı.
    Tüm belge metnini işler ve her chunk span'ı için token gömme vektörlerini
    havuzlayarak bağlam-duyarlı (doküman genelini yansıtan) vektörler üretir.

    max_context_tokens ve overlap_tokens model bazında CollectionSpec
    üzerinden enjekte edilir. Böylece Jina v3 (8K), Jina v4 (32K) ve
    diğer modeller aynı anda kullanılabilir.

    Sorgu zamanı gömme için src.common.embeddings içindeki build_embedder() kullanın.
    """

    def __init__(
        self,
        model_name: str,
        max_context_tokens: int = 8192,
        overlap_tokens: int = 128,
        embed_dim: int | None = None,
    ):
        print(f"--- Yükleniyor: {model_name} (Bu işlem ilk seferde uzun sürebilir) ---")
        with _silence_stdout_stderr():
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, local_files_only=True)

        # Resolve target device up-front
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

        # Jina v3/v4 için özel ayarlar (trust_remote_code ve eager attention)
        load_kwargs = {
            "trust_remote_code": True,
            "low_cpu_mem_usage": False,
            "local_files_only": True,
        }
        if device == "cuda":
            load_kwargs["torch_dtype"] = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        if "jina" in model_name.lower():
            # Flash Attention, Jina'nın 'task' parametresiyle bazen çakışabiliyor.
            # Performans için 'eager' (standart) attention kullanarak task desteğini garantiye alıyoruz.
            load_kwargs["attn_implementation"] = "eager"

        if "mursit" in model_name.lower():
            # Mursit = ModernBERT. Varsayılan reference_compile=True torch.compile/Triton
            # çağırır; bazı ortamlarda (ör. aarch64) CUDA kernel derlemesi başarısız olur.
            load_kwargs["reference_compile"] = False

        try:
            with _silence_stdout_stderr():
                self.model = AutoModel.from_pretrained(model_name, **load_kwargs)
            self.model.eval()
            self.model.to(device)
        except Exception as e:
            if device == "cuda":
                print(f"CUDA'ya taşırken hata: {e}. CPU'da tutuluyor.")
                # If moving to CUDA fails (e.g. OOM), we reload it cleanly on CPU
                load_kwargs.pop("torch_dtype", None)
                with _silence_stdout_stderr():
                    self.model = AutoModel.from_pretrained(model_name, **load_kwargs)
                self.model.eval()
            else:
                raise e

        self.is_jina = "jina" in model_name.lower()
        self.is_nomic = "nomic" in model_name.lower()
        self.is_qwen = "qwen" in model_name.lower()
        # Qwen3-Embedding: son-token (last-token) havuzlama gerektirir (mean değil).
        self.is_qwen3 = "qwen3" in model_name.lower()
        # Jina v4: model(**inputs) yerine kendi encode_text() API'sini kullanır.
        self.is_jina_v4 = "jina-embeddings-v4" in model_name.lower()

        self.max_context_tokens = max_context_tokens
        self.overlap_tokens = overlap_tokens
        self.embed_dim = embed_dim

    @staticmethod
    def _last_token_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Son-token havuzlama (Qwen3-Embedding resmi yöntemi).

        Sağ veya sol padding'i otomatik algılar; her dizinin son padding'siz
        token'ının gizli durumunu döndürür.
        """
        left_padding = attention_mask[:, -1].sum().item() == attention_mask.shape[0]
        if left_padding:
            return last_hidden_state[:, -1]
        seq_lengths = attention_mask.sum(dim=1) - 1
        batch_idx = torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device)
        return last_hidden_state[batch_idx, seq_lengths]

    def _encode_text_jina_v4(self, texts: List[str], task: str) -> List[List[float]]:
        """Jina v4 kendi encode_text() API'siyle gömer (MRL truncate_dim destekli).

        Jina v4 (Qwen2.5-VL tabanlı, ~4B) genel model(**inputs) + mean-pool
        yoluyla kullanılamaz; encode_text task/prompt_name parametreleri ister.
        """
        from src.config import settings
        prompt_name = "query" if "query" in task else "passage"
        batch_size = max(1, settings.EMBED_BATCH_SIZE)
        all_embeddings: List[List[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            with torch.no_grad():
                embs = self.model.encode_text(
                    texts=batch,
                    task="retrieval",
                    prompt_name=prompt_name,
                    truncate_dim=self.embed_dim,
                )
            for e in embs:
                t = e if isinstance(e, torch.Tensor) else torch.as_tensor(e)
                t = F.normalize(t.float().unsqueeze(0), p=2, dim=1).squeeze(0)
                all_embeddings.append(t.cpu().tolist())
            if self.model.device.type == "cuda":
                torch.cuda.empty_cache()
        return all_embeddings

    def embed_documents(self, texts: List[str], task: str = "retrieval.passage") -> List[List[float]]:
        """Uyumluluk için standart gömme.

        Mini-batch'ler hâlinde işlenir: tüm chunk'lar tek forward pass'e
        verilirse (örn. 1817 parça) padding'li tensör ve attention matrisi
        belleği patlatıp OOM'a yol açar. EMBED_BATCH_SIZE ile sınırlanır.

        Havuzlama model bazlı: Jina v4 kendi API'si, Qwen3 son-token,
        diğerleri ortalama havuzlama.
        """
        if self.is_jina_v4:
            return self._encode_text_jina_v4(texts, task)

        if self.is_nomic:
            prefix = "search_query: " if "query" in task else "search_document: "
            texts = [prefix + text for text in texts]
        elif self.is_qwen:
            if "query" in task:
                prefix = "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: "
                texts = [prefix + text for text in texts]

        from src.config import settings
        batch_size = max(1, settings.EMBED_BATCH_SIZE)
        all_embeddings: List[List[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            inputs = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                # Jina v3 için task parametresi
                if self.is_jina:
                    outputs = self.model(**inputs, task=task)
                else:
                    outputs = self.model(**inputs)
            if self.is_qwen3:
                # Qwen3-Embedding son-token havuzlama ister (mean değil)
                embeddings = self._last_token_pool(outputs.last_hidden_state, inputs["attention_mask"])
            else:
                embeddings = outputs.last_hidden_state.mean(dim=1)
            if self.embed_dim is not None:
                embeddings = embeddings[:, :self.embed_dim]
            embeddings = F.normalize(embeddings, p=2, dim=1)
            all_embeddings.extend(embeddings.cpu().tolist())
            # Batch ara belleğini serbest bırak (unified memory'de kritik)
            del inputs, outputs, embeddings
            if self.model.device.type == "cuda":
                torch.cuda.empty_cache()
        return all_embeddings

    def embed_query(self, text: str, task: str = "retrieval.query") -> List[float]:
        return self.embed_documents([text], task=task)[0]

    def _standard_chunk_embeddings(
        self, full_text: str, spans: List[Tuple[int, int]], task: str
    ) -> List[List[float]]:
        """Late chunking uyumsuz modeller için chunk metinlerini bağımsız gömer.

        Qwen3 (son-token havuzlama) ve Jina v4 (encode_text API) span-bazlı
        token havuzlamayla uyumsuzdur; her chunk'ı kendi metniyle gömeriz.
        """
        chunk_texts = [full_text[start:end] for (start, end) in spans]
        return self.embed_documents(chunk_texts, task=task)

    def embed_with_late_chunking(self, full_text: str, spans: List[Tuple[int, int]],
                                task: str = "retrieval.passage") -> List[List[float]]:
        """Tek bir belge üzerinde late chunking uygular."""
        if self.is_qwen3 or self.is_jina_v4:
            return self._standard_chunk_embeddings(full_text, spans, task)
        if self.is_nomic:
            prefix = "search_query: " if "query" in task else "search_document: "
            shift = len(prefix)
            full_text = prefix + full_text
            spans = [(start + shift, end + shift) for start, end in spans]
        elif self.is_qwen:
            if "query" in task:
                prefix = "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: "
                shift = len(prefix)
                full_text = prefix + full_text
                spans = [(start + shift, end + shift) for start, end in spans]

        # Güvenlik kontrolü
        quick_check = self.tokenizer(full_text, return_tensors="pt", truncation=False)
        if quick_check["input_ids"].shape[1] > self.max_context_tokens:
            raise ValueError(
                f"Metin çok uzun ({quick_check['input_ids'].shape[1]} token, "
                f"limit={self.max_context_tokens}). "
                f"embed_with_late_chunking_windowed() kullanın."
            )

        inputs = self.tokenizer(full_text, return_tensors="pt", return_offsets_mapping=True).to(self.model.device)
        offsets = inputs.pop("offset_mapping")[0].cpu()

        with torch.no_grad():
            # Jina v3/v4 için task parametresi kritik
            if self.is_jina:
                outputs = self.model(**inputs, task=task)
            else:
                outputs = self.model(**inputs)

        token_embeddings = outputs.last_hidden_state[0]

        chunk_embeddings = []
        for start_char, end_char in spans:
            token_indices = []
            for i, (t_start, t_end) in enumerate(offsets):
                if t_start == t_end == 0:
                    continue  # [CLS] gibi özel token'ları atla
                if max(t_start, start_char) < min(t_end, end_char):
                    token_indices.append(i)

            if token_indices:
                chunk_vec = token_embeddings[token_indices].mean(dim=0)
            else:
                chunk_vec = token_embeddings.mean(dim=0)

            if self.embed_dim is not None:
                chunk_vec = chunk_vec[:self.embed_dim]

            chunk_vec = F.normalize(chunk_vec.unsqueeze(0), p=2, dim=1).squeeze(0)
            chunk_embeddings.append(chunk_vec.cpu().tolist())

        return chunk_embeddings

    def embed_with_late_chunking_windowed(
        self,
        full_text: str,
        spans: List[Tuple[int, int]],
        max_tokens: int | None = None,
        overlap_tokens: int | None = None,
        task: str = "retrieval.passage",
    ) -> List[List[float]]:
        """
        Pencereli late chunking — max_tokens'dan uzun belgeler için.

        Not: Bu yaklaşım, Late Chunking makalesini yazan araştırmacıların (Jina AI ekibi)
        makalenin "Section 3.1: Extended Algorithm for Long Documents" kısmında önerdiği
        Long Late Chunking mimarisi ile neredeyse birebir aynı mantıkta çalışmaktadır.
        Araştırmacılar da model limitini aşan çok uzun dokümanlarda makro pencereler
        (macro chunks) oluşturup overlap ile Late Chunking yapmayı önermektedir.

        Document link: https://openreview.net/notes/edits/attachment?id=7eUlqSx02t&name=pdf
        """
        if self.is_qwen3 or self.is_jina_v4:
            # Late chunking uyumsuz modeller: standart chunk-bazlı gömme
            return self._standard_chunk_embeddings(full_text, spans, task)

        if max_tokens is None:
            max_tokens = self.max_context_tokens
        if overlap_tokens is None:
            overlap_tokens = self.overlap_tokens

        enc = self.tokenizer(
            full_text,
            return_tensors="pt",
            return_offsets_mapping=True,
            truncation=False,
        )
        all_offsets = enc["offset_mapping"][0]  # (toplam_token, 2) — CPU'da
        total_tokens = all_offsets.shape[0]

        # Tokenizer'ın otomatik eklediği [CLS], [SEP] ve karakter sınırındaki
        # re-tokenizasyon kaymasına (boundary drift) yer açmak için pencereyi daralt.
        # "-3" yalnızca özel token'lar içindi; BPE boundary drift 3-4 ekstra token
        # üretebilir → 512-token modelde 513 > 512 hatasına neden olur.
        safe_max_tokens = max_tokens - 10

        if total_tokens <= safe_max_tokens:
            return self.embed_with_late_chunking(full_text, spans, task=task)

        # Örtüşen token pencerelerini oluştur
        stride = safe_max_tokens - overlap_tokens
        windows: List[Tuple[int, int]] = []  # (token_baş, token_son) — son hariç
        start = 0
        while start < total_tokens:
            end = min(start + safe_max_tokens, total_tokens)
            windows.append((start, end))
            if end == total_tokens:
                break
            start += stride

        # Her span için pencereler arası vektörleri biriktir
        span_accum: List[List[List[float]]] = [[] for _ in spans]

        for win_tok_start, win_tok_end in windows:
            win_char_start = int(all_offsets[win_tok_start][0])
            win_char_end = int(all_offsets[win_tok_end - 1][1])

            window_text = full_text[win_char_start:win_char_end]

            window_span_indices = []
            window_spans_local = []
            for idx, (s_start, s_end) in enumerate(spans):
                if s_end <= win_char_start or s_start >= win_char_end:
                    continue
                local_start = max(s_start, win_char_start) - win_char_start
                local_end = min(s_end, win_char_end) - win_char_start
                window_span_indices.append(idx)
                window_spans_local.append((local_start, local_end))

            if not window_spans_local:
                continue

            window_vecs = self.embed_with_late_chunking(window_text, window_spans_local, task=task)
            for idx, vec in zip(window_span_indices, window_vecs):
                span_accum[idx].append(vec)

        # Birden fazla pencerede geçen span'ların vektörlerini ortala
        result: List[List[float]] = []
        hidden_dim = len(span_accum[0][0]) if span_accum[0] else 0
        for idx, vecs in enumerate(span_accum):
            if vecs:
                avg = [sum(v[d] for v in vecs) / len(vecs) for d in range(len(vecs[0]))]
                result.append(avg)
            else:
                result.append([0.0] * hidden_dim)

        return result
