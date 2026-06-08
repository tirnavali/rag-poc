import os
# Tüm transformers uyarılarını global düzeyde sustur
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.nn.functional as F
import warnings
from typing import List, Tuple, Dict, Any
from transformers import AutoModel, AutoTokenizer, logging
from langchain_core.embeddings import Embeddings

# Jina v3/v4 Flash Attention uyarılarını sustur
warnings.filterwarnings("ignore")
logging.set_verbosity_error()


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
    ):
        print(f"--- Yükleniyor: {model_name} (Bu işlem ilk seferde uzun sürebilir) ---")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

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
            # transformers ≥ 4.45 may default low_cpu_mem_usage=True for
            # trust_remote_code models, leaving weights on the meta device.
            # The subsequent .to(device) call then fails with meta tensor error.
            # Forcing False keeps weights on CPU during load so .to(device) works.
            "low_cpu_mem_usage": False,
        }
        if "jina" in model_name.lower():
            # Flash Attention, Jina'nın 'task' parametresiyle bazen çakışabiliyor.
            # Performans için 'eager' (standart) attention kullanarak task desteğini garantiye alıyoruz.
            load_kwargs["attn_implementation"] = "eager"

        self.model = AutoModel.from_pretrained(model_name, **load_kwargs)
        self.model.eval()
        self.model.to(device)

        self.max_context_tokens = max_context_tokens
        self.overlap_tokens = overlap_tokens

    def embed_documents(self, texts: List[str], task: str = "retrieval.passage") -> List[List[float]]:
        """Uyumluluk için standart gömme (ortalama havuzlama kullanır)."""
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            # Jina v3/v4 için task parametresi
            outputs = self.model(**inputs, task=task)
        embeddings = outputs.last_hidden_state.mean(dim=1)
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings.cpu().tolist()

    def embed_query(self, text: str, task: str = "retrieval.query") -> List[float]:
        return self.embed_documents([text], task=task)[0]

    def embed_with_late_chunking(self, full_text: str, spans: List[Tuple[int, int]], 
                                task: str = "retrieval.passage") -> List[List[float]]:
        """Tek bir belge üzerinde late chunking uygular."""
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
            outputs = self.model(**inputs, task=task)

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
