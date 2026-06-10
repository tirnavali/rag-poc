"""Bu modül atomların chunk'lara paketlenmesinden sorumludur. PDF→Markdown
dönüşümü kapsam dışıdır — MarkdownConverter'a (KATMAN 1) delege edilir.

Mimari rolü: ingestion pipeline'ının KATMAN 2'si (CHUNK).
  Girdi : ParsedDocument (atoms + full_text)
  Çıktı : Late Chunking'e hazır chunk'lar (metin + char ofsetleri + metadata)

Üç paketleme yolu destekler: hybrid (HybridChunker, token-aware),
author-aware (segment_pack), greedy (basit min/max-char).
"""

import hashlib
import json
import os
from typing import Any, Dict, List, Tuple

from src.common.parsing.packer import greedy_pack  # noqa: F401 (backwards-compat re-export)
from src.common.parsing.markdown_converter import MarkdownConverter, ParsedDocument
from src.config import settings


def greedy_pack_atoms(
    atoms: List[Dict[str, Any]],
    min_chars: int,
    max_chars: int,
    join_str: str = "\n\n",
) -> List[Dict[str, Any]]:
    """Greedy pack list of atom dicts, aggregating page/pages metadata."""
    packed_chunks: List[Dict[str, Any]] = []
    current_chunk: List[Dict[str, Any]] = []
    current_len = 0

    for atom in atoms:
        atom_text = atom["text"]
        if not current_chunk:
            current_chunk = [atom]
            current_len = len(atom_text)
            continue

        proposed_len = current_len + len(join_str) + len(atom_text)

        if current_len >= min_chars and proposed_len > max_chars:
            merged_text = join_str.join([a["text"] for a in current_chunk])
            merged_pages = sorted(
                set(p for a in current_chunk for p in a.get("pages", []))
            )
            packed_chunks.append(
                {
                    "text": merged_text,
                    "label": "Packed",
                    "page": merged_pages[0] if merged_pages else None,
                    "pages": merged_pages,
                }
            )
            current_chunk = [atom]
            current_len = len(atom_text)
        else:
            current_chunk.append(atom)
            current_len = proposed_len

    if current_chunk:
        merged_text = join_str.join([a["text"] for a in current_chunk])
        merged_pages = sorted(
            set(p for a in current_chunk for p in a.get("pages", []))
        )
        packed_chunks.append(
            {
                "text": merged_text,
                "label": "Packed",
                "page": merged_pages[0] if merged_pages else None,
                "pages": merged_pages,
            }
        )

    return packed_chunks


class DoclingManager:
    """
    Karmaşık dokümanları (PDF, DOCX vb.) anlamsal parçalara ayıran ve
    akıllı paketleme yaparak Late Chunking'e hazır hale getiren yönetici sınıf.

    PDF → Markdown dönüşümü MarkdownConverter'a delege edilir.
    Bu sınıf packing (chunking) katmanından sorumludur.

    Kullanım:
        mgr = DoclingManager()
        full_text, chunks = mgr.convert_and_pack("belge.pdf")

    Ayrı adımlarla:
        parsed = mgr._converter.convert("belge.pdf")
        full_text, chunks = mgr.pack(parsed, "belge.pdf")

    OCR engine settings.OCR_ENGINE ile kontrol edilir (default: "easyocr").
    Override: DoclingManager(ocr_engine="tesseract") veya OCR_ENGINE=tesseract env.
    """

    def __init__(
        self,
        ocr_engine: str | None = None,
        do_ocr: bool = True,
        tokenizer_name: str | None = None,
        max_chunk_tokens: int = 400,
        min_chunk_tokens: int = 100,
    ):
        self._converter = MarkdownConverter(ocr_engine=ocr_engine, do_ocr=do_ocr)
        self.ocr_engine = self._converter.ocr_engine
        self.do_ocr = self._converter.do_ocr
        self.tokenizer_name = tokenizer_name
        self.max_chunk_tokens = max_chunk_tokens
        self.min_chunk_tokens = min_chunk_tokens

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def convert_and_pack(
        self,
        file_path: str,
        min_chars: int = 500,
        max_chars: int = 1500,
        do_pack: bool = True,
        document_type: str | None = None,
        initial_author: str | None = None,
        initial_role: str | None = None,
        quality_document_type: str | None = None,
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        PDF → chunk'lar.  MarkdownConverter.convert() + DoclingManager.pack() zinciri.

        İmza ve dönüş tipi değişmez — tüm çağıranlar (adapter'lar) güncelleme gerektirmez.

        quality_document_type: Yalnızca kalite (karakter sapması) karşılaştırması için
            kullanılan tip etiketi. Verilmezse author-aware'in tetiklenmemesini
            istediği halde kalite kontrolü yapmak isteyen adapter'lar için
            (ör. pdf_report). Boş bırakılırsa document_type kullanılır.
        """
        use_hybrid = bool(self.tokenizer_name)
        parsed = self._converter.convert(
            file_path,
            use_hybrid=use_hybrid,
            document_type=quality_document_type or document_type,
        )
        return self.pack(
            parsed,
            file_path,
            min_chars=min_chars,
            max_chars=max_chars,
            do_pack=do_pack,
            document_type=document_type,
            initial_author=initial_author,
            initial_role=initial_role,
        )

    def pack(
        self,
        parsed: ParsedDocument,
        file_path: str,
        min_chars: int = 500,
        max_chars: int = 1500,
        do_pack: bool = True,
        document_type: str | None = None,
        initial_author: str | None = None,
        initial_role: str | None = None,
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        ParsedDocument'ı chunk'lara paketler.

        Level-2 (chunk) önbelleği kullanır; cache hit'te doğrudan döner.
        Üç paketleme yolunu destekler: hybrid, author-aware, greedy.

        Args:
            parsed:         MarkdownConverter.convert() çıktısı.
            file_path:      Yalnızca metadata (kaynak adı) için kullanılır.
            min_chars / max_chars: Greedy / author-aware yollar için chunk sınırları.
            do_pack:        False ise atomlar chunk olarak olduğu gibi döner.
            document_type:  Author-aware yol için doküman tipi (ör. "tutanak").
            initial_author / initial_role: İlk konuşmacı bilgisi.

        Returns:
            (full_text, chunks) — pipeline'ın beklediği format.
        """
        use_hybrid = bool(self.tokenizer_name)
        atoms_data = parsed.atoms
        full_text = parsed.full_text
        ocr_flagged = bool((parsed.quality or {}).get("ocr_flagged", False))

        # Level-2 chunk cache key — aynı şema, mevcut cache dosyaları geçerli kalır
        if use_hybrid:
            author_tag = f"_author_{document_type}" if document_type else ""
            chunk_cache_key = hashlib.md5(
                f"{parsed.ocr_base}_hybrid_{self.tokenizer_name}_{self.max_chunk_tokens}"
                f"_{self.min_chunk_tokens}{author_tag}".encode()
            ).hexdigest()
        else:
            author_tag = f"_author_{document_type}" if document_type else ""
            chunk_cache_key = hashlib.md5(
                f"{parsed.ocr_base}_{min_chars}_{max_chars}_{do_pack}{author_tag}".encode()
            ).hexdigest()

        cache_dir = settings.PARSE_CACHE_DIR
        chunk_cache_file = cache_dir / f"{chunk_cache_key}.json"

        # Level-2 hit: chunk'lar daha önce hesaplanmış
        if chunk_cache_file.exists():
            try:
                with open(chunk_cache_file, "r", encoding="utf-8") as f:
                    cached_data = json.load(f)
                chunks_to_check = cached_data.get("chunks", [])
                has_page_meta = bool(chunks_to_check) and all(
                    "page" in c.get("metadata", {}) and "pages" in c.get("metadata", {})
                    for c in chunks_to_check
                )
                if has_page_meta:
                    print(f"  [CACHE] Chunk önbellekten okundu: {os.path.basename(file_path)}")
                    # ocr_flagged eski cache'lerde yok — cache geçerli kalır,
                    # bayrak güncel quality'den post-hoc enjekte edilir.
                    self._apply_ocr_flag(cached_data["chunks"], ocr_flagged)
                    return cached_data["full_text"], cached_data["chunks"]
                else:
                    print("  [CACHE] Önbellekte sayfa numarası eksik, yeniden oluşturuluyor.")
            except Exception as e:
                print(f"  [WARN] Chunk önbellek okuma hatası, devam ediliyor: {e}")

        join_str = "\n\n"

        # --- Hybrid path ---
        if use_hybrid and parsed.dl_doc is not None:
            full_text_hybrid, final_chunks = self._hybrid_pack(
                parsed.dl_doc,
                file_path,
                document_type=document_type,
                initial_author=initial_author,
                initial_role=initial_role,
            )
            self._apply_ocr_flag(final_chunks, ocr_flagged)
            self._save_chunk_cache(chunk_cache_file, full_text_hybrid, final_chunks)
            return full_text_hybrid, final_chunks

        # --- Author-aware path ---
        if document_type and do_pack:
            from src.common.parsing.author_extractor import tag_atoms
            from src.common.parsing.extractors import get_extractor
            from src.common.parsing.segment_pack import segment_aware_pack

            extractor = get_extractor(document_type)
            tagged = tag_atoms(
                atoms_data,
                extractor,
                initial_author=initial_author,
                initial_role=initial_role,
            )
            packed = segment_aware_pack(
                tagged,
                min_chars=min_chars,
                max_chars=max_chars,
                join_str=join_str,
                inject_continuation_prefix=False,
            )

            final_chunks = []
            current_search_pos = 0
            for chunk in packed:
                p_text = chunk["text"]
                start_idx = full_text.find(p_text, current_search_pos)
                span = None
                if start_idx != -1:
                    span = (start_idx, start_idx + len(p_text))
                    current_search_pos = start_idx + 1
                merged_meta = {
                    "source": os.path.basename(file_path),
                    "char_count": len(p_text),
                    "is_packed": True,
                    "type": "AuthorAwarePacked",
                    "ocr_engine": self.ocr_engine,
                    **chunk["metadata"],
                }
                final_chunks.append({"text": p_text, "span": span, "metadata": merged_meta})

            self._apply_ocr_flag(final_chunks, ocr_flagged)
            self._save_chunk_cache(chunk_cache_file, full_text, final_chunks)
            return full_text, final_chunks

        # --- Greedy path ---
        if do_pack:
            final_items = greedy_pack_atoms(
                atoms_data, min_chars=min_chars, max_chars=max_chars, join_str=join_str
            )
        else:
            final_items = atoms_data

        final_chunks = []
        current_search_pos = 0
        for item in final_items:
            p_text = item["text"]
            start_idx = full_text.find(p_text, current_search_pos)
            if start_idx != -1:
                final_chunks.append(
                    {
                        "text": p_text,
                        "span": (start_idx, start_idx + len(p_text)),
                        "metadata": {
                            "source": os.path.basename(file_path),
                            "char_count": len(p_text),
                            "is_packed": do_pack,
                            "type": item["label"],
                            "ocr_engine": self.ocr_engine,
                            "page": item.get("page"),
                            "pages": item.get("pages", []),
                        },
                    }
                )
                current_search_pos = start_idx + 1

        self._apply_ocr_flag(final_chunks, ocr_flagged)
        self._save_chunk_cache(chunk_cache_file, full_text, final_chunks)
        return full_text, final_chunks

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_ocr_flag(chunks: list, ocr_flagged: bool) -> None:
        """Tier-1 kalite bayrağını her chunk metadata'sına taşır (in-place)."""
        for chunk in chunks:
            chunk.setdefault("metadata", {})["ocr_flagged"] = ocr_flagged

    def _save_chunk_cache(self, cache_file, full_text: str, chunks: list) -> None:
        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump({"full_text": full_text, "chunks": chunks}, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"  [WARN] Chunk önbellek yazma hatası: {e}")

    def _hybrid_pack(
        self,
        dl_doc,
        file_path: str,
        document_type: str | None = None,
        initial_author: str | None = None,
        initial_role: str | None = None,
    ):
        """HybridChunker ile belgeyi parçala, charspan'ları kullan, min-token merge uygula."""
        from docling.chunking import HybridChunker
        from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
        from docling_core.transforms.serializer.markdown import MarkdownDocSerializer as _MDS

        tokenizer = HuggingFaceTokenizer.from_pretrained(
            model_name=self.tokenizer_name,
            max_tokens=self.max_chunk_tokens,
        )
        chunker = HybridChunker(tokenizer=tokenizer)

        # full_text Docling serializer'dan — charspan'lar buna göredir
        full_text = _MDS(doc=dl_doc).serialize().text

        chunks = []
        for hchunk in chunker.chunk(dl_doc):
            all_spans = [
                p.charspan
                for item in hchunk.meta.doc_items
                for p in getattr(item, "prov", [])
                if p.charspan != (0, 0)
            ]
            span = (
                (min(s[0] for s in all_spans), max(s[1] for s in all_spans))
                if all_spans
                else None
            )
            all_pages = sorted(
                {
                    p_no
                    for item in hchunk.meta.doc_items
                    for p in getattr(item, "prov", [])
                    for p_no in [getattr(p, "page_no", None)]
                    if p_no is not None
                }
            )
            primary_page = all_pages[0] if all_pages else None

            chunks.append(
                {
                    "text": hchunk.text,
                    "span": span,
                    "metadata": {
                        "source": os.path.basename(file_path),
                        "char_count": len(hchunk.text),
                        "is_packed": True,
                        "type": "HybridChunk",
                        "ocr_engine": self.ocr_engine,
                        "headings": hchunk.meta.headings or [],
                        "page": primary_page,
                        "pages": all_pages,
                    },
                }
            )

        print(f"  [HYBRID] {len(chunks)} chunk üretildi, min-token merge uygulanıyor...")
        chunks = self._min_token_merge(chunks, tokenizer)
        print(f"  [HYBRID] Merge sonrası: {len(chunks)} chunk")

        if document_type:
            from src.common.parsing.author_extractor import tag_chunks_post_hoc
            from src.common.parsing.extractors import get_extractor

            extractor = get_extractor(document_type)
            tag_chunks_post_hoc(
                chunks,
                extractor,
                initial_author=initial_author,
                initial_role=initial_role,
            )
            print(f"  [HYBRID] Author meta uygulandı ({document_type})")

        return full_text, chunks

    def _min_token_merge(self, chunks, tokenizer):
        """min_chunk_tokens altındaki chunk'ları bir sonrakiyle birleştir."""
        result = []
        current = None
        for chunk in chunks:
            if current is None:
                current = chunk
                continue
            cur_tokens = tokenizer.count_tokens(current["text"])
            if cur_tokens < self.min_chunk_tokens:
                merged_text = current["text"] + "\n\n" + chunk["text"]
                cur_span = current["span"]
                nxt_span = chunk["span"]
                if cur_span and nxt_span:
                    merged_span = (
                        min(cur_span[0], nxt_span[0]),
                        max(cur_span[1], nxt_span[1]),
                    )
                else:
                    merged_span = cur_span or nxt_span
                cur_pages = current["metadata"].get("pages", [])
                nxt_pages = chunk["metadata"].get("pages", [])
                merged_pages = sorted(set(cur_pages + nxt_pages))
                current = {
                    **current,
                    "text": merged_text,
                    "span": merged_span,
                    "metadata": {
                        **current["metadata"],
                        "char_count": len(merged_text),
                        "page": merged_pages[0] if merged_pages else None,
                        "pages": merged_pages,
                    },
                }
            else:
                result.append(current)
                current = chunk
        if current:
            result.append(current)
        return result

    def _get_file_hash(self, file_path: str) -> str:
        """MarkdownConverter._get_file_hash'e delege eder."""
        return self._converter._get_file_hash(file_path)
