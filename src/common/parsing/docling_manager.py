"""
Docling kütüphanesini kullanarak dokümanları parse eden ve Late Chunking için
uygun formatta (metin + ofsetler) çıktı veren modül.
"""

import os
from pathlib import Path
from typing import List, Tuple, Dict, Any

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    EasyOcrOptions,
    TesseractCliOcrOptions,
    OcrMacOptions,
)
from docling_core.transforms.serializer.markdown import MarkdownDocSerializer
from docling.document_converter import DocumentConverter, PdfFormatOption

from src.common.parsing.packer import greedy_pack
from src.config import settings


def greedy_pack_atoms(
    atoms: List[Dict[str, Any]], 
    min_chars: int, 
    max_chars: int, 
    join_str: str = "\n\n"
) -> List[Dict[str, Any]]:
    """Greedy pack list of atom dicts, aggregating page/pages metadata."""
    packed_chunks = []
    current_chunk = []
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
            merged_pages = sorted(list(set(p for a in current_chunk for p in a.get("pages", []))))
            packed_chunks.append({
                "text": merged_text,
                "label": "Packed",
                "page": merged_pages[0] if merged_pages else None,
                "pages": merged_pages
            })
            current_chunk = [atom]
            current_len = len(atom_text)
        else:
            current_chunk.append(atom)
            current_len = proposed_len
            
    if current_chunk:
        merged_text = join_str.join([a["text"] for a in current_chunk])
        merged_pages = sorted(list(set(p for a in current_chunk for p in a.get("pages", []))))
        packed_chunks.append({
            "text": merged_text,
            "label": "Packed",
            "page": merged_pages[0] if merged_pages else None,
            "pages": merged_pages
        })
        
    return packed_chunks



def _build_ocr_options(engine: str):
    """Map engine name → Docling OCR options object."""
    if engine == "easyocr":
        return EasyOcrOptions(lang=["tr"], use_gpu=False)
    elif engine == "tesseract":
        return TesseractCliOcrOptions(lang=["tur"])
    elif engine == "mac":
        return OcrMacOptions(lang=["tr-TR"])
    else:
        raise ValueError(
            f"Unknown OCR engine: {engine!r}. Valid options: easyocr, tesseract, mac"
        )


class DoclingManager:
    """
    Karmaşık dokümanları (PDF, DOCX vb.) anlamsal parçalara ayıran ve
    akıllı paketleme yaparak Late Chunking'e hazır hale getiren yönetici sınıf.

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
        engine = ocr_engine or settings.OCR_ENGINE
        self.ocr_engine = engine
        self.do_ocr = do_ocr
        self.tokenizer_name = tokenizer_name
        self.max_chunk_tokens = max_chunk_tokens
        self.min_chunk_tokens = min_chunk_tokens

        if do_ocr:
            ocr_options = _build_ocr_options(engine)
            pipeline_options = PdfPipelineOptions(do_ocr=True, ocr_options=ocr_options)
        else:
            pipeline_options = PdfPipelineOptions(do_ocr=False)
        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )

    def convert_and_pack(
        self,
        file_path: str,
        min_chars: int = 500,
        max_chars: int = 1500,
        do_pack: bool = True,
        document_type: str | None = None,
        initial_author: str | None = None,
        initial_role: str | None = None,
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Dokümanı okur, anlamsal parçalara ayırır ve (opsiyonel olarak) paketler.
        OCR sonuçlarını data_lake/parse_cache altında saklar.
        """
        import hashlib
        import json

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dosya bulunamadı: {file_path}")

        # Two-level cache:
        #   Level 1 — OCR/parse cache: keyed on file_hash + ocr_engine only.
        #             Stores full_text + raw atoms_data (before chunking).
        #             Reused across collections with different chunk params.
        #   Level 2 — Chunk cache: keyed on file_hash + ocr_engine + min/max/pack.
        #             Stores final packed chunks. Fast path — no OCR, no re-pack.
        file_hash = self._get_file_hash(file_path)
        # do_ocr=True is the historic default — omit tag to stay compatible
        # with existing cache files. Only tag when OCR is explicitly disabled.
        ocr_tag = "" if self.do_ocr else "_no_ocr"
        ocr_base = f"{file_hash}_{self.ocr_engine}{ocr_tag}"

        use_hybrid = bool(self.tokenizer_name)

        ocr_cache_key = hashlib.md5(ocr_base.encode()).hexdigest()
        author_tag = f"_author_{document_type}" if document_type else ""
        if use_hybrid:
            chunk_cache_key = hashlib.md5(
                f"{ocr_base}_hybrid_{self.tokenizer_name}_{self.max_chunk_tokens}_{self.min_chunk_tokens}{author_tag}".encode()
            ).hexdigest()
        else:
            chunk_cache_key = hashlib.md5(
                f"{ocr_base}_{min_chars}_{max_chars}_{do_pack}{author_tag}".encode()
            ).hexdigest()

        cache_dir = settings.PARSE_CACHE_DIR
        cache_dir.mkdir(parents=True, exist_ok=True)
        ocr_cache_file = cache_dir / f"{ocr_cache_key}_atoms.json"
        doc_cache_file = cache_dir / f"{ocr_cache_key}_doc.json"
        chunk_cache_file = cache_dir / f"{chunk_cache_key}.json"

        # Level 2 hit: full result cached
        if chunk_cache_file.exists():
            try:
                with open(chunk_cache_file, "r", encoding="utf-8") as f:
                    cached_data = json.load(f)
                
                # Check if cached chunks contain both "page" and "pages" metadata
                has_page_meta = True
                chunks_to_check = cached_data.get("chunks", [])
                if chunks_to_check:
                    for chk in chunks_to_check:
                        chk_meta = chk.get("metadata", {})
                        if "page" not in chk_meta or "pages" not in chk_meta:
                            has_page_meta = False
                            break
                else:
                    has_page_meta = False
                
                if has_page_meta:
                    print(f"  [CACHE] Chunk önbellekten okundu: {os.path.basename(file_path)}")
                    return cached_data["full_text"], cached_data["chunks"]
                else:
                    print(f"  [CACHE] Önbellekte sayfa numarası eksik, yeniden oluşturuluyor: {os.path.basename(file_path)}")
            except Exception as e:
                print(f"  [WARN] Chunk önbellek okuma hatası, devam ediliyor: {e}")

        # Level 1 hit: OCR done before, only re-pack
        atoms_data = None
        full_text = None
        dl_doc = None
        if ocr_cache_file.exists():
            try:
                with open(ocr_cache_file, "r", encoding="utf-8") as f:
                    ocr_cached = json.load(f)
                atoms_to_check = ocr_cached.get("atoms_data", [])
                
                # Check if cached atoms contain both "page" and "pages" keys
                has_page_meta = True
                if atoms_to_check:
                    for atom in atoms_to_check:
                        if "page" not in atom or "pages" not in atom:
                            has_page_meta = False
                            break
                else:
                    has_page_meta = False

                if has_page_meta:
                    atoms_data = ocr_cached["atoms_data"]
                    full_text = ocr_cached["full_text"]
                    print(f"  [CACHE] OCR önbellekten okundu, yeniden paketleniyor: {os.path.basename(file_path)}")
                else:
                    print(f"  [CACHE] OCR önbelleğinde sayfa numarası eksik, yeniden parse ediliyor: {os.path.basename(file_path)}")
            except Exception as e:
                print(f"  [WARN] OCR önbellek okuma hatası, yeniden parse ediliyor: {e}")
                atoms_data = None

        # Load DoclingDocument cache for hybrid path
        if use_hybrid and doc_cache_file.exists():
            try:
                from docling_core.types.doc.document import DoclingDocument as _DoclingDocument
                with open(doc_cache_file, "r", encoding="utf-8") as f:
                    dl_doc = _DoclingDocument.model_validate(json.load(f))
            except Exception as e:
                print(f"  [WARN] Doc önbellek okuma hatası, yeniden parse ediliyor: {e}")
                dl_doc = None

        if atoms_data is None or (use_hybrid and dl_doc is None):
            # Full miss: run OCR + parse
            print(f"  [PARSE] Docling çalıştırılıyor (OCR: {self.ocr_engine}): {os.path.basename(file_path)}")
            result = self.converter.convert(file_path)
            dl_doc = result.document

            atoms_data = []
            try:
                serializer = MarkdownDocSerializer(doc=dl_doc)
                for item, _ in dl_doc.iterate_items():
                    content = serializer.serialize(item=item).text
                    label = getattr(item, "label", "unknown")
                    if content.strip():
                        # Extract pages
                        pages = []
                        for p in getattr(item, "prov", []):
                            p_no = getattr(p, "page_no", None)
                            if p_no is not None:
                                pages.append(p_no)
                        pages = sorted(list(set(pages)))
                        primary_page = pages[0] if pages else None
                        atoms_data.append({
                            "text": content.strip(),
                            "label": str(label),
                            "page": primary_page,
                            "pages": pages
                        })
            except Exception as e:
                print(f"Uyarı: Gelişmiş Markdown dışa aktarma başarısız oldu, manuel yönteme geçiliyor: {e}")
                atoms_data = []
                for item, _ in dl_doc.iterate_items():
                    text = getattr(item, "text", "").strip()
                    label = getattr(item, "label", "unknown")
                    if not text:
                        continue
                    # Extract pages
                    pages = []
                    for p in getattr(item, "prov", []):
                        p_no = getattr(p, "page_no", None)
                        if p_no is not None:
                            pages.append(p_no)
                    pages = sorted(list(set(pages)))
                    primary_page = pages[0] if pages else None
                    if "heading" in str(label).lower():
                        level = str(label).split("_")[-1] if "_" in str(label) else "1"
                        atoms_data.append({
                            "text": f"{'#' * int(level if level.isdigit() else 1)} {text}",
                            "label": str(label),
                            "page": primary_page,
                            "pages": pages
                        })
                    else:
                        atoms_data.append({
                            "text": text,
                            "label": str(label),
                            "page": primary_page,
                            "pages": pages
                        })

            join_str = "\n\n"
            full_text = join_str.join([a["text"] for a in atoms_data])

            # Save Level 1 (OCR/atoms cache)
            try:
                with open(ocr_cache_file, "w", encoding="utf-8") as f:
                    json.dump({"full_text": full_text, "atoms_data": atoms_data}, f, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"  [WARN] OCR önbellek yazma hatası: {e}")

            # Save DoclingDocument cache for hybrid path
            if use_hybrid:
                try:
                    with open(doc_cache_file, "w", encoding="utf-8") as f:
                        json.dump(dl_doc.model_dump(mode="json"), f, ensure_ascii=False)
                except Exception as e:
                    print(f"  [WARN] Doc önbellek yazma hatası: {e}")

        join_str = "\n\n"

        # HybridChunker path
        if use_hybrid and dl_doc is not None:
            full_text_hybrid, final_chunks = self._hybrid_pack(
                dl_doc,
                file_path,
                document_type=document_type,
                initial_author=initial_author,
                initial_role=initial_role,
            )
            try:
                with open(chunk_cache_file, "w", encoding="utf-8") as f:
                    json.dump({"full_text": full_text_hybrid, "chunks": final_chunks}, f, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"  [WARN] Chunk önbellek yazma hatası: {e}")
            return full_text_hybrid, final_chunks

        # Author-aware pack path (multi-author documents)
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
                if start_idx != -1:
                    end_idx = start_idx + len(p_text)
                    span = (start_idx, end_idx)
                    current_search_pos = start_idx + 1
                else:
                    span = None
                merged_meta = {
                    "source": os.path.basename(file_path),
                    "char_count": len(p_text),
                    "is_packed": True,
                    "type": "AuthorAwarePacked",
                    "ocr_engine": self.ocr_engine,
                    **chunk["metadata"],
                }
                final_chunks.append({"text": p_text, "span": span, "metadata": merged_meta})

            try:
                with open(chunk_cache_file, "w", encoding="utf-8") as f:
                    json.dump({"full_text": full_text, "chunks": final_chunks}, f, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"  [WARN] Chunk önbellek yazma hatası: {e}")

            return full_text, final_chunks

        # Greedy pack path (original)
        if do_pack:
            final_items = greedy_pack_atoms(
                atoms_data,
                min_chars=min_chars,
                max_chars=max_chars,
                join_str=join_str
            )
        else:
            final_items = atoms_data

        # Compute char spans within full_text
        final_chunks = []
        current_search_pos = 0
        for item in final_items:
            p_text = item["text"]
            start_idx = full_text.find(p_text, current_search_pos)
            if start_idx != -1:
                end_idx = start_idx + len(p_text)
                final_chunks.append({
                    "text": p_text,
                    "span": (start_idx, end_idx),
                    "metadata": {
                        "source": os.path.basename(file_path),
                        "char_count": len(p_text),
                        "is_packed": do_pack,
                        "type": item["label"],
                        "ocr_engine": self.ocr_engine,
                        "page": item.get("page"),
                        "pages": item.get("pages", []),
                    }
                })
                current_search_pos = start_idx + 1

        # Save Level 2 (chunk cache)
        try:
            with open(chunk_cache_file, "w", encoding="utf-8") as f:
                json.dump({"full_text": full_text, "chunks": final_chunks}, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"  [WARN] Chunk önbellek yazma hatası: {e}")

        return full_text, final_chunks

    def _hybrid_pack(
        self,
        dl_doc,
        file_path: str,
        document_type: str | None = None,
        initial_author: str | None = None,
        initial_role: str | None = None,
    ):
        """HybridChunker ile belgeyi parçala, charspan'ları kullan, min token merge uygula."""
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
                if all_spans else None
            )
            all_pages = []
            for item in hchunk.meta.doc_items:
                for p in getattr(item, "prov", []):
                    p_no = getattr(p, "page_no", None)
                    if p_no is not None:
                        all_pages.append(p_no)
            all_pages = sorted(list(set(all_pages)))
            primary_page = all_pages[0] if all_pages else None

            chunks.append({
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
            })

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
                    merged_span = (min(cur_span[0], nxt_span[0]), max(cur_span[1], nxt_span[1]))
                else:
                    merged_span = cur_span or nxt_span
                cur_pages = current["metadata"].get("pages", [])
                nxt_pages = chunk["metadata"].get("pages", [])
                merged_pages = sorted(list(set(cur_pages + nxt_pages)))
                primary_page = merged_pages[0] if merged_pages else None
                current = {
                    **current,
                    "text": merged_text,
                    "span": merged_span,
                    "metadata": {
                        **current["metadata"],
                        "char_count": len(merged_text),
                        "page": primary_page,
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
        """Dosyanın SHA-256 hash'ini hesaplar."""
        import hashlib
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
