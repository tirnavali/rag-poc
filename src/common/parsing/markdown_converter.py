"""Bu modül PDF→Markdown dönüşümünden sorumludur. Chunking, embedding ve
ingestion orkestrasyon kapsam dışıdır.

Mimari rolü: ingestion pipeline'ının KATMAN 1'i (PARSE).
  Girdi : PDF (veya Docling'in desteklediği diğer formatlar)
  Çıktı : ParsedDocument(atoms, full_text, pages_by_number, quality)

Bağımsız CLI olarak da kullanılabilir — pipeline'a bağımlı değildir:
    python -m src.common.parsing.markdown_converter --file belge.pdf

Chunking için DoclingManager.pack() (KATMAN 2) kullanılır.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    EasyOcrOptions,
    OcrMacOptions,
    PdfPipelineOptions,
    TesseractCliOcrOptions,
)
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.transforms.serializer.markdown import MarkdownDocSerializer

from src.common.parsing.quality import compute_quality, extract_ocr_confidence
from src.config import settings


@dataclass
class ParsedDocument:
    """PDF → Markdown dönüşümünün sonucu. Packing adımına girdi olarak geçirilir."""

    full_text: str
    atoms: List[Dict[str, Any]]
    dl_doc: Any | None
    ocr_base: str               # Level-2 chunk önbelleği anahtar üretimi için
    markdown_path: str | None = field(default=None)
    atoms_path: str | None = field(default=None)
    pages_path: str | None = field(default=None)
    pages_by_number: List[Dict[str, Any]] = field(default_factory=list)
    quality: Dict[str, Any] = field(default_factory=dict)  # Tier-1 OCR kalite metrikleri


def _build_ocr_options(engine: str):
    """Engine adı → Docling OCR seçenekleri nesnesi."""
    if engine == "easyocr":
        return EasyOcrOptions(lang=["tr"], use_gpu=settings.DOCLING_USE_GPU)
    elif engine == "tesseract":
        return TesseractCliOcrOptions(lang=["tur"])
    elif engine == "mac":
        return OcrMacOptions(lang=["tr-TR"])
    else:
        raise ValueError(
            f"Bilinmeyen OCR engine: {engine!r}. Geçerli seçenekler: easyocr, tesseract, mac"
        )


class MarkdownConverter:
    """
    PDF/DOCX → Markdown dönüşümcüsü.

    Sadece parse katmanından sorumludur (OCR + atom çıkarma + markdown serileştirme).
    Chunking / packing bu sınıfın dışındadır — DoclingManager.pack() bunu üstlenir.

    İki seviyeli Level-1 önbellek (parse_cache/):
      {ocr_hash}_atoms.json  — full_text + atomlar
      {ocr_hash}_doc.json    — DoclingDocument (sadece hybrid path)
    """

    def __init__(
        self,
        ocr_engine: str | None = None,
        do_ocr: bool = True,
        images_scale: float = 1.0,
        use_vlm: bool = False,
        ollama_model: str | None = None,
        ollama_url: str | None = None,
        force_band: bool = False,
        paddle_url: str | None = None,
        paddle_model: str | None = None,
    ):
        engine = ocr_engine or settings.OCR_ENGINE
        self.ocr_engine = engine
        self.do_ocr = do_ocr
        self.images_scale = images_scale
        self.use_vlm = use_vlm
        self.ollama_model = ollama_model or settings.VLM_TABLE_MODEL
        self.ollama_url = ollama_url or settings.OLLAMA_HOST
        self.force_band = force_band
        self.paddle_url = paddle_url or settings.PADDLE_OCR_URL
        self.paddle_model = paddle_model or settings.PADDLE_OCR_MODEL

        if do_ocr:
            ocr_options = _build_ocr_options(engine)
            pipeline_options = PdfPipelineOptions(do_ocr=True, ocr_options=ocr_options, images_scale=self.images_scale)
        else:
            pipeline_options = PdfPipelineOptions(do_ocr=False, images_scale=self.images_scale)

        # PyPdfium backend: gömülü fontun ToUnicode/CMap eşlemesi bozuk PDF'lerde
        # Docling'in varsayılan backend'i native metin katmanını yanlış Unicode'a
        # çevirir ("Adalet" → "AGaOeW"). PyPdfium bu glyph-substitution bozulmasını
        # giderir. (4608ce5'te eklenmiş, MarkdownConverter refaktöründe kaybolmuştu.)
        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options,
                    backend=PyPdfiumDocumentBackend,
                )
            }
        )

    def convert(
        self,
        file_path: str,
        use_hybrid: bool = False,
        document_type: str | None = None,
    ) -> ParsedDocument:
        """
        Dosyayı parse et; markdown artefaktını diske yaz ve ParsedDocument döndür.

        Level-1 önbellek varsa Docling atlanır, atomlar diskten okunur.
        Her dönüşüm sonrası data_lake/markdown/ altına okunabilir bir .md dosyası kaydedilir.

        Args:
            file_path:     Parse edilecek dosya yolu.
            use_hybrid:    True ise DoclingDocument da döndürülür (HybridChunker için).
            document_type: Yalnızca kalite metrikleri için (tip bazlı karakter
                           sapması karşılaştırması). Parse davranışını değiştirmez.

        Returns:
            ParsedDocument — full_text, atoms, dl_doc (opsiyonel), ocr_base,
            markdown_path, quality.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dosya bulunamadı: {file_path}")

        file_hash = self._get_file_hash(file_path)
        ocr_tag = "" if self.do_ocr else "_no_ocr"
        # VLM açıkken cache yeniden anahtarlanır; aksi halde eski (VLM'siz) atomlar
        # yeniden kullanılır ve tablolar düzeltilmeden kalırdı.
        # Backend'e göre anahtarla: tesseract ve VLM çıktıları aynı cache'i paylaşmasın.
        if self.use_vlm:
            if settings.TABLE_EXTRACTOR == "tesseract":
                vlm_tag = "_tess"
            elif settings.TABLE_EXTRACTOR == "paddleocr":
                vlm_tag = f"_paddle-{self.paddle_model}"
            else:
                vlm_tag = f"_vlm-{self.ollama_model}"
        else:
            vlm_tag = ""
        # _pypdfium: PyPdfium backend'iyle üretilen (doğru-kodlanmış) atom'lar, eski
        # varsayılan-backend (bozuk) cache'inden ayrı anahtarlansın; aksi halde
        # reingest eski bozuk metni cache'ten okur.
        ocr_base = f"{file_hash}_{self.ocr_engine}{ocr_tag}_scale{self.images_scale}_pypdfium{vlm_tag}"
        ocr_cache_key = hashlib.md5(ocr_base.encode()).hexdigest()

        cache_dir = settings.PARSE_CACHE_DIR
        cache_dir.mkdir(parents=True, exist_ok=True)
        ocr_cache_file = cache_dir / f"{ocr_cache_key}_atoms.json"
        doc_cache_file = cache_dir / f"{ocr_cache_key}_doc.json"

        atoms_data = None
        full_text = None
        dl_doc = None
        ocr_mean_confidence = None

        # Level-1 hit: atomlar daha önce parse edilmiş
        if ocr_cache_file.exists():
            try:
                with open(ocr_cache_file, "r", encoding="utf-8") as f:
                    ocr_cached = json.load(f)
                atoms_to_check = ocr_cached.get("atoms_data", [])
                # schema_version >= 2: prov charspan'lerinden çok-sayfalı atom
                # bölme (page_texts) eklendi. v1 cache'ler sayfa sınırını aşan
                # blokları yanlış sayfaya atfettiği için geçersiz sayılır.
                schema_ok = ocr_cached.get("schema_version", 1) >= 2
                has_page_meta = (
                    schema_ok
                    and bool(atoms_to_check)
                    and all("page" in a and "pages" in a for a in atoms_to_check)
                )
                if has_page_meta:
                    atoms_data = ocr_cached["atoms_data"]
                    full_text = ocr_cached["full_text"]
                    # quality alanı eski cache'lerde yok — yokluğu cache'i geçersiz
                    # kılmaz; OCR güveni yeniden hesaplanamadığı için None kalır.
                    ocr_mean_confidence = (ocr_cached.get("quality") or {}).get(
                        "ocr_mean_confidence"
                    )
                    print(f"  [CACHE] OCR önbellekten okundu: {os.path.basename(file_path)}")
                else:
                    print("  [CACHE] OCR önbelleğinde sayfa numarası eksik, yeniden parse ediliyor.")
            except Exception as e:
                print(f"  [WARN] OCR önbellek okuma hatası, yeniden parse ediliyor: {e}")

        # DoclingDocument önbellekten yükle (hybrid path)
        if use_hybrid and doc_cache_file.exists():
            try:
                from docling_core.types.doc.document import DoclingDocument as _DoclingDocument

                with open(doc_cache_file, "r", encoding="utf-8") as f:
                    dl_doc = _DoclingDocument.model_validate(json.load(f))
            except Exception as e:
                print(f"  [WARN] Doc önbellek okuma hatası, yeniden parse ediliyor: {e}")
                dl_doc = None

        parsed_fresh = False
        if atoms_data is None or (use_hybrid and dl_doc is None):
            print(
                f"  [PARSE] Docling çalıştırılıyor (OCR: {self.ocr_engine}): "
                f"{os.path.basename(file_path)}"
            )
            result = self.converter.convert(file_path)
            dl_doc = result.document
            parsed_fresh = True

            atoms_data = self._extract_atoms(dl_doc)
            full_text = "\n\n".join(a["text"] for a in atoms_data)
            ocr_mean_confidence = extract_ocr_confidence(result) if self.do_ocr else None

            if use_hybrid and dl_doc is not None:
                try:
                    with open(doc_cache_file, "w", encoding="utf-8") as f:
                        json.dump(dl_doc.model_dump(mode="json"), f, ensure_ascii=False)
                except Exception as e:
                    print(f"  [WARN] Doc önbellek yazma hatası: {e}")

        # VLM table extraction — yalnızca "bozuk" tablolarda (TableFormer yapı
        # çıkaramamış veya OCR çöp); düzgün okunan tablolara dokunmaz.
        vlm_applied = False
        if self.use_vlm and atoms_data:
            from src.common.parsing.vlm_table_extractor import table_is_low_quality
            if any(table_is_low_quality(a) for a in atoms_data):
                if settings.TABLE_EXTRACTOR == "tesseract":
                    from src.common.parsing.tess_table_extractor import process_atoms_with_tesseract
                    print("  [TESS] Bozuk tablolar Tesseract+OpenCV ile yeniden okunuyor...")
                    atoms_data = process_atoms_with_tesseract(file_path, atoms_data)
                elif settings.TABLE_EXTRACTOR == "paddleocr":
                    from src.common.parsing.paddle_table_extractor import process_atoms_with_paddle
                    print(f"  [PADDLE] Bozuk tablolar PaddleOCR ({self.paddle_model}) ile yeniden okunuyor...")
                    atoms_data = process_atoms_with_paddle(
                        file_path, atoms_data, self.paddle_url, self.paddle_model,
                    )
                else:
                    from src.common.parsing.vlm_table_extractor import process_atoms_with_vlm
                    print(f"  [VLM] Bozuk tablolar {self.ollama_model} ile yeniden okunuyor"
                          f"{' (zorla bantlama)' if self.force_band else ''}...")
                    atoms_data = process_atoms_with_vlm(
                        file_path, atoms_data, self.ollama_url, self.ollama_model,
                        force_band=self.force_band,
                    )
                full_text = "\n\n".join(a["text"] for a in atoms_data)
                vlm_applied = True

        # Tier-1 kalite metrikleri — cache hit'te de yeniden hesaplanır
        # (tip ortalaması koleksiyon büyüdükçe değişir); yalnızca OCR güveni
        # taze parse gerektirir.
        quality = compute_quality(
            atoms_data,
            full_text,
            document_type=document_type,
            ocr_mean_confidence=ocr_mean_confidence,
            stats_key=file_hash,
        )
        if quality["ocr_flagged"]:
            print(
                f"  [WARN] OCR kalite bayrağı ({os.path.basename(file_path)}): "
                f"{', '.join(quality['flags'])}"
            )

        # Level-1 önbelleğe kaydet (quality alanı dahil)
        if parsed_fresh or vlm_applied:
            try:
                with open(ocr_cache_file, "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            # v3: tablo atomlarına table_num_rows/table_num_cols +
                            # opsiyonel extracted_by alanları eklendi. Okuma kapısı >=2.
                            "schema_version": 3,
                            "full_text": full_text,
                            "atoms_data": atoms_data,
                            "quality": quality,
                        },
                        f,
                        ensure_ascii=False,
                        indent=2,
                    )
            except Exception as e:
                print(f"  [WARN] OCR önbellek yazma hatası: {e}")

        overwrite_artifacts = parsed_fresh or vlm_applied
        markdown_path = self._save_markdown_artifact(file_path, file_hash, full_text, overwrite=overwrite_artifacts)
        atoms_path = self._save_atoms_artifact(file_path, file_hash, atoms_data, overwrite=overwrite_artifacts)
        pages_by_number = self._build_pages_by_number(atoms_data)
        pages_path = self._save_pages_artifact(file_path, file_hash, pages_by_number, overwrite=overwrite_artifacts)

        return ParsedDocument(
            full_text=full_text,
            atoms=atoms_data,
            dl_doc=dl_doc,
            ocr_base=ocr_base,
            markdown_path=markdown_path,
            atoms_path=atoms_path,
            pages_path=pages_path,
            pages_by_number=pages_by_number,
            quality=quality,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_atoms(self, dl_doc) -> List[Dict[str, Any]]:
        """DoclingDocument → atom listesi (metin + etiket + sayfa bilgisi + opsiyonel bbox)."""
        atoms_data = []
        try:
            serializer = MarkdownDocSerializer(doc=dl_doc)
            for item, _ in dl_doc.iterate_items():
                content = serializer.serialize(item=item).text
                label = getattr(item, "label", "unknown")
                if content.strip():
                    pages = self._extract_pages(item)
                    atom = {
                        "text": content.strip(),
                        "label": str(label),
                        "page": pages[0] if pages else None,
                        "pages": pages,
                    }
                    if str(label) == "table":
                        self._attach_table_meta(atom, item)
                    page_texts = self._extract_page_texts(item)
                    if page_texts:
                        atom["page_texts"] = page_texts
                    atoms_data.append(atom)
        except Exception as e:
            print(f"  [WARN] Gelişmiş Markdown dışa aktarma başarısız, manuel yönteme geçiliyor: {e}")
            atoms_data = []
            for item, _ in dl_doc.iterate_items():
                text = getattr(item, "text", "").strip()
                label = getattr(item, "label", "unknown")
                if not text:
                    continue
                pages = self._extract_pages(item)
                if "heading" in str(label).lower():
                    level_str = str(label).split("_")[-1] if "_" in str(label) else "1"
                    level = int(level_str) if level_str.isdigit() else 1
                    text = f"{'#' * level} {text}"
                atom = {
                    "text": text,
                    "label": str(label),
                    "page": pages[0] if pages else None,
                    "pages": pages,
                }
                if str(label) == "table":
                    self._attach_table_meta(atom, item)
                page_texts = self._extract_page_texts(item)
                if page_texts:
                    atom["page_texts"] = page_texts
                atoms_data.append(atom)
        return atoms_data

    @staticmethod
    def _attach_table_meta(atom: Dict[str, Any], item) -> None:
        """Tablo atomuna bbox + TableFormer satır/sütun sayısını ekler (VLM tetikleyici için).

        num_rows/num_cols == 0 → TableFormer yapı çıkaramadı (döndürülmüş/taranmış
        tablo göstergesi); bbox ise VLM kırpması için gerekli.
        """
        provs = getattr(item, "prov", [])
        if provs and getattr(provs[0], "bbox", None):
            bbox = provs[0].bbox
            atom["bbox"] = {
                "l": bbox.l,
                "t": bbox.t,
                "r": bbox.r,
                "b": bbox.b,
                "coord_origin": str(bbox.coord_origin),
            }
        data = getattr(item, "data", None)
        if data is not None:
            atom["table_num_rows"] = getattr(data, "num_rows", None)
            atom["table_num_cols"] = getattr(data, "num_cols", None)

    @staticmethod
    def _build_pages_by_number(atoms: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Atomları primary page'e göre gruplar; sorted [{sayfaNo, sayfa_markdown}] döner."""
        from collections import defaultdict

        page_buckets: dict[int, list[str]] = defaultdict(list)
        for atom in atoms:
            # Sayfa sınırını aşan atom: her parça prov charspan'ine göre doğru sayfaya.
            # (Cache'ten okunduğunda JSON anahtarları string olur → int'e normalize.)
            page_texts = atom.get("page_texts")
            if page_texts:
                for pno, txt in page_texts.items():
                    if txt.strip():
                        page_buckets[int(pno)].append(txt)
                continue
            primary_page = atom.get("page")
            if primary_page is None:
                continue
            page_buckets[primary_page].append(atom["text"])
        return [
            {"sayfa_no": page_no, "sayfa_markdown": "\n\n".join(page_buckets[page_no])}
            for page_no in sorted(page_buckets.keys())
        ]

    @staticmethod
    def _extract_pages(item) -> List[int]:
        return sorted(
            {
                p_no
                for p in getattr(item, "prov", [])
                for p_no in [getattr(p, "page_no", None)]
                if p_no is not None
            }
        )

    @staticmethod
    def _extract_page_texts(item) -> Dict[int, str]:
        """Sayfa sınırını aşan item'ı prov charspan'lerine göre {sayfa_no: metin}'e böler.

        Docling, fiziksel olarak iki sayfaya yayılan bir layout bloğunu (ör. çok-sütunlu
        yoklama isim listesi) tek item'da toplar ama her parçanın sayfasını prov'da ayrı
        charspan ile işaretler. Tek sayfaya ait item'larda boş dict döner (bölmeye gerek
        yok); bölünemeyen durumlarda (charspan yok/geçersiz, tek sayfa) yine {} döner ve
        çağıran primary-page davranışına geri düşer.
        """
        raw = getattr(item, "text", "") or ""
        provs = [
            p for p in getattr(item, "prov", []) if getattr(p, "page_no", None) is not None
        ]
        if not raw or len({p.page_no for p in provs}) <= 1:
            return {}

        from collections import defaultdict

        buckets: dict[int, list[str]] = defaultdict(list)
        for p in provs:
            cs = getattr(p, "charspan", None)
            if not cs or tuple(cs) == (0, 0):
                continue
            start, end = cs[0], cs[1]
            if 0 <= start < end <= len(raw):
                seg = raw[start:end].strip()
                if seg:
                    buckets[p.page_no].append(seg)

        if len(buckets) <= 1:
            return {}
        return {pno: " ".join(segs) for pno, segs in buckets.items()}

    def _save_pages_artifact(
        self, file_path: str, file_hash: str, pages_by_number: List[Dict[str, Any]], overwrite: bool = False
    ) -> str | None:
        """pages_by_number'ı data_lake/pages/ altına sidecar JSON olarak yazar."""
        try:
            pages_dir = settings.PAGES_DIR
            pages_dir.mkdir(parents=True, exist_ok=True)
            source_stem = Path(file_path).stem
            if self.use_vlm:
                if settings.TABLE_EXTRACTOR == "tesseract":
                    suffix = "__tess"
                elif settings.TABLE_EXTRACTOR == "paddleocr":
                    suffix = "__paddle"
                else:
                    suffix = "__vlm"
            else:
                suffix = ""
            pages_path = pages_dir / f"{source_stem}__{file_hash[:8]}{suffix}_pages.json"
            if overwrite or not pages_path.exists():
                pages_path.write_text(
                    json.dumps(pages_by_number, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                print(f"  [PAGES] Artefakt kaydedildi: {pages_path.name}")
            return str(pages_path)
        except Exception as e:
            print(f"  [WARN] Pages artefakt yazma hatası: {e}")
            return None

    def _save_atoms_artifact(
        self, file_path: str, file_hash: str, atoms_data: List[Dict[str, Any]], overwrite: bool = False
    ) -> str | None:
        """atom listesini data_lake/atoms/ altına okunabilir sidecar JSON olarak yazar.

        parse_cache/{md5}_atoms.json'ın {stem}__{hash8} ile anahtarlı, gözle
        incelenebilir yansımasıdır (aşama 2 — docling atomları).
        """
        try:
            atoms_dir = settings.ATOMS_DIR
            atoms_dir.mkdir(parents=True, exist_ok=True)
            source_stem = Path(file_path).stem
            if self.use_vlm:
                if settings.TABLE_EXTRACTOR == "tesseract":
                    suffix = "__tess"
                elif settings.TABLE_EXTRACTOR == "paddleocr":
                    suffix = "__paddle"
                else:
                    suffix = "__vlm"
            else:
                suffix = ""
            atoms_path = atoms_dir / f"{source_stem}__{file_hash[:8]}{suffix}_atoms.json"
            if overwrite or not atoms_path.exists():
                atoms_path.write_text(
                    json.dumps(
                        {
                            "source_stem": source_stem,
                            "file_hash8": file_hash[:8],
                            "atom_count": len(atoms_data),
                            "atoms": atoms_data,
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                print(f"  [ATOMS] Artefakt kaydedildi: {atoms_path.name}")
            return str(atoms_path)
        except Exception as e:
            print(f"  [WARN] Atoms artefakt yazma hatası: {e}")
            return None

    def _save_markdown_artifact(
        self, file_path: str, file_hash: str, full_text: str, overwrite: bool = False
    ) -> str | None:
        """full_text'i data_lake/markdown/ altına okunabilir .md dosyası olarak yazar."""
        try:
            md_dir = settings.MARKDOWN_DIR
            md_dir.mkdir(parents=True, exist_ok=True)
            source_stem = Path(file_path).stem
            if self.use_vlm:
                if settings.TABLE_EXTRACTOR == "tesseract":
                    suffix = "__tess"
                elif settings.TABLE_EXTRACTOR == "paddleocr":
                    suffix = "__paddle"
                else:
                    suffix = "__vlm"
            else:
                suffix = ""
            md_path = md_dir / f"{source_stem}__{file_hash[:8]}{suffix}.md"
            if overwrite or not md_path.exists():
                md_path.write_text(full_text, encoding="utf-8")
                print(f"  [MARKDOWN] Artefakt kaydedildi: {md_path.name}")
            return str(md_path)
        except Exception as e:
            print(f"  [WARN] Markdown artefakt yazma hatası: {e}")
            return None

    def _get_file_hash(self, file_path: str) -> str:
        """Dosyanın SHA-256 hash'ini hesaplar."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="PDF → Markdown dönüşümcüsü (chunk/embed olmadan).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Örnekler:
  python -m src.common.parsing.markdown_converter --file belge.pdf
  python -m src.common.parsing.markdown_converter --file belge.pdf --ocr-engine tesseract
  python -m src.common.parsing.markdown_converter --file belge.pdf --no-ocr
  python -m src.common.parsing.markdown_converter --file belge.pdf --use-vlm
  python -m src.common.parsing.markdown_converter --file belge.pdf --use-vlm --ollama-model qwen2.5vl:32b
  python -m src.common.parsing.markdown_converter --file belge.pdf --vlm paddleocr
  python -m src.common.parsing.markdown_converter --file belge.pdf --vlm paddleocr --paddle-url http://10.20.24.16:4000/v1
  python -m src.common.parsing.markdown_converter --file belge.pdf --use-tesseract
        """,
    )
    parser.add_argument("--file", required=True, help="Parse edilecek PDF dosyası")
    parser.add_argument(
        "--ocr-engine", default=None, help="OCR engine: easyocr | tesseract | mac"
    )
    parser.add_argument("--no-ocr", action="store_true", help="OCR'yi devre dışı bırak")
    parser.add_argument(
        "--pages-json", action="store_true", help="Sayfa bazlı JSON çıktısını stdout'a yaz"
    )
    parser.add_argument(
        "--images-scale", type=float, default=1.0, help="PDF sayfalarının render çözünürlük ölçeği (örn: 2.0 veya 3.0)"
    )
    parser.add_argument(
        "--force", action="store_true", help="Önbelleği yoksay ve dosyayı zorla yeniden parse et"
    )
    parser.add_argument(
        "--use-vlm", action="store_true", help="Tablolar için yerel Ollama VLM modelini kullan"
    )
    parser.add_argument(
        "--use-tesseract", action="store_true",
        help="Tablolar için VLM yerine klasik Tesseract+OpenCV backend'ini kullan (hızlı)"
    )
    parser.add_argument(
        "--vlm", metavar="BACKEND", default=None,
        help="Tablo çıkarma backend'i: paddleocr | ollama (--use-vlm ile aynı) | tesseract"
    )
    parser.add_argument(
        "--paddle-url", default=None,
        help=f"PaddleOCR API base URL (varsayılan: {settings.PADDLE_OCR_URL})"
    )
    parser.add_argument(
        "--paddle-model", default=None,
        help=f"PaddleOCR model adı (varsayılan: {settings.PADDLE_OCR_MODEL})"
    )
    parser.add_argument(
        "--band", action="store_true",
        help="Tüm tabloları zorla satır-bantlarına böl (A/B testi); adaptifi geçersiz kılar"
    )
    parser.add_argument(
        "--ollama-model", default=None,
        help=f"Ollama VLM modeli (varsayılan: {settings.VLM_TABLE_MODEL})"
    )
    parser.add_argument(
        "--ollama-url", default=None, help="Ollama API host adresi (örn: http://localhost:11434)"
    )
    args = parser.parse_args()

    # --vlm <backend>: tablo düzeltme katmanını aç + backend'i ayarla.
    # --use-tesseract / --use-vlm: eski flaglar, geriye uyumlu.
    vlm_backend = (args.vlm or "").strip().lower() if args.vlm else None
    if vlm_backend == "paddleocr":
        settings.TABLE_EXTRACTOR = "paddleocr"
    elif vlm_backend in ("tesseract",):
        settings.TABLE_EXTRACTOR = "tesseract"
    elif vlm_backend in ("ollama", "vlm"):
        settings.TABLE_EXTRACTOR = "vlm"
    elif args.use_tesseract:
        settings.TABLE_EXTRACTOR = "tesseract"

    use_table_layer = bool(vlm_backend) or args.use_vlm or args.use_tesseract

    conv = MarkdownConverter(
        ocr_engine=args.ocr_engine,
        do_ocr=not args.no_ocr,
        images_scale=args.images_scale,
        use_vlm=use_table_layer,
        ollama_model=args.ollama_model,
        ollama_url=args.ollama_url,
        force_band=args.band,
        paddle_url=args.paddle_url,
        paddle_model=args.paddle_model,
    )

    # If force is true, delete the matching cache file first (vlm anahtarı dahil)
    if args.force:
        import hashlib
        def _get_file_hash(fp):
            h = hashlib.sha256()
            with open(fp, "rb") as f:
                for b in iter(lambda: f.read(4096), b''): h.update(b)
            return h.hexdigest()
        fh = _get_file_hash(args.file)
        otag = "" if not args.no_ocr else "_no_ocr"
        engine = args.ocr_engine or settings.OCR_ENGINE
        if settings.TABLE_EXTRACTOR == "tesseract":
            vtag = "_tess"
        elif settings.TABLE_EXTRACTOR == "paddleocr":
            paddle_m = args.paddle_model or settings.PADDLE_OCR_MODEL
            vtag = f"_paddle-{paddle_m}"
        elif use_table_layer:
            vtag = f"_vlm-{args.ollama_model or settings.VLM_TABLE_MODEL}"
        else:
            vtag = ""
        obase = f"{fh}_{engine}{otag}_scale{args.images_scale}{vtag}"
        ckey = hashlib.md5(obase.encode()).hexdigest()
        cfile = settings.PARSE_CACHE_DIR / f"{ckey}_atoms.json"
        if cfile.exists():
            cfile.unlink()

    parsed = conv.convert(args.file)
    print(f"\nTamamlandı.")
    print(f"  Atom sayısı : {len(parsed.atoms)}")
    print(f"  Sayfa sayısı: {len(parsed.pages_by_number)}")
    print(f"  full_text   : {len(parsed.full_text):,} karakter")
    if parsed.quality.get("ocr_flagged"):
        print(f"  Kalite      : BAYRAKLI — {', '.join(parsed.quality['flags'])}")
    else:
        print(f"  Kalite      : temiz")
    if parsed.markdown_path:
        print(f"  Artefakt    : {parsed.markdown_path}")
    if parsed.pages_path:
        print(f"  Pages JSON  : {parsed.pages_path}")
    if args.pages_json:
        import json as _json
        print(_json.dumps(parsed.pages_by_number, ensure_ascii=False, indent=2))
    sys.exit(0)
