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
from pathlib import Path
from typing import Any, Dict, List, Tuple

from src.common.parsing.packer import greedy_pack  # noqa: F401 (backwards-compat re-export)
from src.common.parsing.markdown_converter import MarkdownConverter, ParsedDocument
from src.config import settings

# Chunklama token-tabanlıdır (HybridChunker). Bu karakter sınırları yalnızca
# tokenizer/dl_doc bulunamadığında devreye giren greedy güvenlik ağı içindir;
# normal üretim yolunda kullanılmaz.
_FALLBACK_MIN_CHARS = 500
_FALLBACK_MAX_CHARS = 1500


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


def _atom_char_spans(
    atoms: List[Dict[str, Any]], full_text: str, join_str: str = "\n\n"
) -> List[Tuple[int, int] | None]:
    """Her atomun `full_text` içindeki (start, end) char-aralığını döndürür.

    `full_text = join_str.join(a["text"] for a in atoms)` değişmezine dayanır
    (bkz. MarkdownConverter). Bu yüzden atom metni `full_text`'te birebir bulunur
    ve span deterministiktir — OCR bozulmasından etkilenmez. Sıralı bir imleçle
    `find` kullanılır (greedy güvenlik ağıyla aynı desen); bulunamayan atom için
    None döner (beklenmez).
    """
    spans: List[Tuple[int, int] | None] = []
    cursor = 0
    for atom in atoms:
        t = atom["text"]
        idx = full_text.find(t, cursor)
        if idx == -1:
            idx = full_text.find(t)
        if idx == -1:
            spans.append(None)
            continue
        end = idx + len(t)
        spans.append((idx, end))
        cursor = end
    return spans


def token_pack_atoms(
    atoms: List[Dict[str, Any]],
    full_text: str,
    count_tokens,
    max_tokens: int,
    min_tokens: int,
    join_str: str = "\n\n",
) -> List[Dict[str, Any]]:
    """Atomları **token-tabanlı** greedy paketler; span'ı `full_text`'ten türetir.

    greedy_pack_atoms ile aynı mantık, tek fark boyutun karakter yerine token
    cinsinden ölçülmesi. Her atom bölünmez (tablo = tek atom → asla ortadan
    kesilmez); tek başına `max_tokens`'ı aşan atom kendi chunk'ı olur. Chunk
    span'ı, paketlenen atomların `full_text` içindeki uç ofsetlerinden gelir;
    chunk metni `full_text[span]` ile birebir aynıdır (late chunking güvencesi).

    Args:
        count_tokens: metin → token sayısı (seçili embedding tokenizer'ı).
    Returns:
        list[dict]: {text, span, label, page, pages} — metadata pack() içinde tamamlanır.
    """
    atom_spans = _atom_char_spans(atoms, full_text, join_str=join_str)
    # Token sayımı atom başına önceden hesaplanır (toplamsal yaklaşım — subword
    # sınır birleşmeleri nedeniyle hafifçe fazla tahmin eder → chunk'lar limiti
    # aşmaz, güvenli yönde). _min_token_merge'deki O(N^2) yeniden sayımı önler.
    atom_tokens = [count_tokens(a["text"]) for a in atoms]

    chunks: List[Dict[str, Any]] = []
    cur_idx: List[int] = []  # mevcut chunk'a giren atom indeksleri
    cur_tokens = 0

    def _flush():
        if not cur_idx:
            return
        first, last = cur_idx[0], cur_idx[-1]
        s0 = atom_spans[first]
        s1 = atom_spans[last]
        if s0 is None or s1 is None:
            return  # span çıkarılamadı (beklenmez) — chunk'ı düşür
        start, end = s0[0], s1[1]
        merged_pages = sorted(
            {p for i in cur_idx for p in atoms[i].get("pages", [])}
        )
        chunks.append(
            {
                "text": full_text[start:end],
                "span": (start, end),
                "label": "Packed",
                "page": merged_pages[0] if merged_pages else None,
                "pages": merged_pages,
            }
        )

    for i, atom in enumerate(atoms):
        n = atom_tokens[i]
        if not cur_idx:
            cur_idx = [i]
            cur_tokens = n
            continue
        proposed = cur_tokens + n
        # min_tokens'a ulaştıysak ve bir sonrakini eklemek max'ı aşıyorsa kes.
        if cur_tokens >= min_tokens and proposed > max_tokens:
            _flush()
            cur_idx = [i]
            cur_tokens = n
        else:
            cur_idx.append(i)
            cur_tokens = proposed

    _flush()
    return chunks


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
        use_vlm: bool | None = None,
        images_scale: float = 1.0,
        ollama_model: str | None = None,
        ollama_url: str | None = None,
    ):
        # VLM tablo çıkarımı: argüman verilmezse settings.VLM_TABLE_EXTRACTION'dan
        # gelir; böylece adapter'lar/pipeline değişmeden ayarı miras alır.
        if use_vlm is None:
            use_vlm = settings.VLM_TABLE_EXTRACTION
        self._converter = MarkdownConverter(
            ocr_engine=ocr_engine,
            do_ocr=do_ocr,
            images_scale=images_scale,
            use_vlm=use_vlm,
            ollama_model=ollama_model,
            ollama_url=ollama_url,
        )
        self.ocr_engine = self._converter.ocr_engine
        self.do_ocr = self._converter.do_ocr
        self.tokenizer_name = tokenizer_name
        self.max_chunk_tokens = max_chunk_tokens
        self.min_chunk_tokens = min_chunk_tokens
        # Son pack() çağrısının ürettiği 4 aşamalık artefakt yolları (gözlemlenebilirlik
        # index'i — pipeline raporuna aktarılır). Her pack() çağrısında güncellenir.
        self.last_artifacts: Dict[str, Any] | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def convert_and_pack(
        self,
        file_path: str,
        do_pack: bool = True,
        document_type: str | None = None,
        initial_author: str | None = None,
        initial_role: str | None = None,
        quality_document_type: str | None = None,
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        PDF → chunk'lar.  MarkdownConverter.convert() + DoclingManager.pack() zinciri.

        Chunklama **token-tabanlıdır**: boyut, tokenizer_name + max_chunk_tokens /
        min_chunk_tokens ile belirlenir (HybridChunker). Karakter sınırı yoktur.

        quality_document_type: Yalnızca kalite (karakter sapması) karşılaştırması için
            kullanılan tip etiketi. Boş bırakılırsa document_type kullanılır.
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
            do_pack=do_pack,
            document_type=document_type,
            initial_author=initial_author,
            initial_role=initial_role,
        )

    def pack(
        self,
        parsed: ParsedDocument,
        file_path: str,
        do_pack: bool = True,
        document_type: str | None = None,
        initial_author: str | None = None,
        initial_role: str | None = None,
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        ParsedDocument'ı **token-tabanlı** chunk'lara paketler.

        Level-2 (chunk) önbelleği kullanır; cache hit'te doğrudan döner.

        Birincil yol atom-tabanlı token paketleme (`_atom_token_pack`): atomlar
        seçili tokenizer ile max_chunk_tokens'a kadar greedy paketlenir; span,
        atomların `full_text` içindeki ofsetlerinden türetilir (charspan
        provenance'a bağlı DEĞİL → OCR belgelerde de %100 span, late chunking
        aktif). Tablo tek atom olduğundan asla ortadan bölünmez. Yazar metadata'sı
        post-hoc atanır (tag_chunks_post_hoc).

        Yalnızca tokenizer yoksa basit bir karakter greedy güvenlik ağına düşülür
        — bu yol üretimde çalışmaz. (`_hybrid_pack` artık kullanılmıyor; HybridChunker
        OCR'da charspan=(0,0) → span=None ürettiği için bırakıldı.)

        Args:
            parsed:         MarkdownConverter.convert() çıktısı.
            file_path:      Yalnızca metadata (kaynak adı) için kullanılır.
            do_pack:        False ise atomlar chunk olarak olduğu gibi döner.
            document_type:  Author tagging için doküman tipi (ör. "tutanak").
            initial_author / initial_role: İlk konuşmacı bilgisi.

        Returns:
            (full_text, chunks) — pipeline'ın beklediği format.
        """
        use_hybrid = bool(self.tokenizer_name)
        atoms_data = parsed.atoms
        full_text = parsed.full_text
        ocr_flagged = bool((parsed.quality or {}).get("ocr_flagged", False))

        # Level-2 chunk cache key — token params (hybrid) veya fallback
        author_tag = f"_author_{document_type}" if document_type else ""
        if use_hybrid:
            chunk_cache_key = hashlib.md5(
                f"{parsed.ocr_base}_atompack_{self.tokenizer_name}_{self.max_chunk_tokens}"
                f"_{self.min_chunk_tokens}{author_tag}".encode()
            ).hexdigest()
        else:
            chunk_cache_key = hashlib.md5(
                f"{parsed.ocr_base}_fallback_{do_pack}{author_tag}".encode()
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
                    return self._finalize(
                        file_path, parsed, cached_data["full_text"], cached_data["chunks"]
                    )
                else:
                    print("  [CACHE] Önbellekte sayfa numarası eksik, yeniden oluşturuluyor.")
            except Exception as e:
                print(f"  [WARN] Chunk önbellek okuma hatası, devam ediliyor: {e}")

        join_str = "\n\n"

        # --- Atom token-pack (token-aware) — birincil yol ---
        # Span `full_text`'ten türetilir → OCR'da da %100 span, late chunking aktif.
        if use_hybrid and parsed.atoms:
            full_text_packed, final_chunks = self._atom_token_pack(
                parsed,
                file_path,
                document_type=document_type,
                initial_author=initial_author,
                initial_role=initial_role,
            )
            self._apply_ocr_flag(final_chunks, ocr_flagged)
            self._save_chunk_cache(chunk_cache_file, full_text_packed, final_chunks)
            return self._finalize(file_path, parsed, full_text_packed, final_chunks)

        # --- Karakter greedy güvenlik ağı (yalnız tokenizer/dl_doc yoksa) ---
        if use_hybrid:
            print("  [WARN] tokenizer var ama dl_doc yok — karakter greedy fallback'e düşülüyor.")
        if do_pack:
            final_items = greedy_pack_atoms(
                atoms_data,
                min_chars=_FALLBACK_MIN_CHARS,
                max_chars=_FALLBACK_MAX_CHARS,
                join_str=join_str,
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
        return self._finalize(file_path, parsed, full_text, final_chunks)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _finalize(
        self,
        file_path: str,
        parsed: ParsedDocument,
        full_text: str,
        chunks: List[Dict[str, Any]],
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """packed_atoms artefaktını yazar, last_artifacts index'ini doldurur, sonucu döner.

        pack()'in tüm dönüş yollarından (cache-hit + hybrid/author-aware/greedy)
        geçer; böylece okunabilir 'packed' artefaktı cache-hit'te de yazılır ve
        4 aşamalık yol haritası her zaman güncel kalır.
        """
        source_stem = Path(file_path).stem
        # ocr_base = "{file_hash}_{engine}{tag}" → file_hash'in ilk 8 hanesi
        file_hash8 = (parsed.ocr_base or "").split("_")[0][:8]
        packed_path = self._save_packed_artifact(source_stem, file_hash8, chunks)

        self.last_artifacts = {
            "source_stem": source_stem,
            "file_hash8": file_hash8,
            "markdown": parsed.markdown_path,
            "atoms": parsed.atoms_path,
            "packed_atoms": packed_path,
            "pages": parsed.pages_path,
        }
        return full_text, chunks

    def _save_packed_artifact(
        self, source_stem: str, file_hash8: str, chunks: List[Dict[str, Any]]
    ) -> str | None:
        """Paketlenmiş chunk'ları data_lake/packed_atoms/ altına okunabilir sidecar yazar.

        parse_cache/{md5}.json (chunk cache) ile aynı içerik; ama {stem}__{hash8}
        ile anahtarlı ve gözle incelenebilir (aşama 3 — packed_atoms).
        """
        if not file_hash8:
            return None
        try:
            packed_dir = settings.PACKED_ATOMS_DIR
            packed_dir.mkdir(parents=True, exist_ok=True)
            packed_path = packed_dir / f"{source_stem}__{file_hash8}_packed.json"
            if not packed_path.exists():
                packed_path.write_text(
                    json.dumps(
                        {
                            "source_stem": source_stem,
                            "file_hash8": file_hash8,
                            "chunk_count": len(chunks),
                            "chunks": chunks,
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                print(f"  [PACKED] Artefakt kaydedildi: {packed_path.name}")
            return str(packed_path)
        except Exception as e:
            print(f"  [WARN] Packed artefakt yazma hatası: {e}")
            return None

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

    def _atom_token_pack(
        self,
        parsed: ParsedDocument,
        file_path: str,
        document_type: str | None = None,
        initial_author: str | None = None,
        initial_role: str | None = None,
    ):
        """Atom granülünde token-aware paketleme — span `full_text`'ten türetilir.

        Birincil yol. HybridChunker'ın charspan provenance'ına bağımlı olmadığı
        için taranmış/OCR belgelerde de %100 span üretir (late chunking devre dışı
        kalmaz). Docling'in atom yapısı korunur (tablo = tek atom → bölünmez).
        """
        from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer

        full_text = parsed.full_text
        atoms = parsed.atoms

        tokenizer = HuggingFaceTokenizer.from_pretrained(
            model_name=self.tokenizer_name,
            max_tokens=self.max_chunk_tokens,
        )

        packed = token_pack_atoms(
            atoms,
            full_text,
            count_tokens=tokenizer.count_tokens,
            max_tokens=self.max_chunk_tokens,
            min_tokens=self.min_chunk_tokens,
        )

        chunks = []
        for p in packed:
            chunks.append(
                {
                    "text": p["text"],
                    "span": p["span"],
                    "metadata": {
                        "source": os.path.basename(file_path),
                        "char_count": len(p["text"]),
                        "is_packed": True,
                        "type": "AtomPacked",
                        "ocr_engine": self.ocr_engine,
                        "headings": [],
                        "page": p.get("page"),
                        "pages": p.get("pages", []),
                    },
                }
            )

        print(f"  [ATOMPACK] {len(atoms)} atom → {len(chunks)} chunk (token-aware, span %100)")

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
            print(f"  [ATOMPACK] Author meta uygulandı ({document_type})")

        return full_text, chunks

    def _hybrid_pack(
        self,
        dl_doc,
        file_path: str,
        document_type: str | None = None,
        initial_author: str | None = None,
        initial_role: str | None = None,
    ):
        """[ARTIK KULLANILMIYOR — _atom_token_pack ile değiştirildi]

        HybridChunker ile belgeyi parçala, charspan'ları kullan, min-token merge uygula.
        OCR belgelerde charspan=(0,0) → span=None ürettiği için bırakıldı; referans/
        karşılaştırma amacıyla korunuyor.
        """
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
