"""
PDF (ve diğer desteklenen formatlar) → Markdown dönüşümünü bağımsız bir modül olarak sağlar.

Sorumluluk: yalnızca parse katmanı.
  PDF → atoms (anlamsal parçalar) + full_text (markdown)

Chunking / packing için DoclingManager.pack() kullanın.
Bağımsız kullanım: python -m src.common.parsing.markdown_converter --file belge.pdf
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
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.transforms.serializer.markdown import MarkdownDocSerializer

from src.config import settings


@dataclass
class ParsedDocument:
    """PDF → Markdown dönüşümünün sonucu. Packing adımına girdi olarak geçirilir."""

    full_text: str
    atoms: List[Dict[str, Any]]
    dl_doc: Any | None
    ocr_base: str               # Level-2 chunk önbelleği anahtar üretimi için
    markdown_path: str | None = field(default=None)
    pages_path: str | None = field(default=None)
    pages_by_number: List[Dict[str, Any]] = field(default_factory=list)


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
    ):
        engine = ocr_engine or settings.OCR_ENGINE
        self.ocr_engine = engine
        self.do_ocr = do_ocr

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

    def convert(self, file_path: str, use_hybrid: bool = False) -> ParsedDocument:
        """
        Dosyayı parse et; markdown artefaktını diske yaz ve ParsedDocument döndür.

        Level-1 önbellek varsa Docling atlanır, atomlar diskten okunur.
        Her dönüşüm sonrası data_lake/markdown/ altına okunabilir bir .md dosyası kaydedilir.

        Args:
            file_path:   Parse edilecek dosya yolu.
            use_hybrid:  True ise DoclingDocument da döndürülür (HybridChunker için).

        Returns:
            ParsedDocument — full_text, atoms, dl_doc (opsiyonel), ocr_base, markdown_path.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dosya bulunamadı: {file_path}")

        file_hash = self._get_file_hash(file_path)
        ocr_tag = "" if self.do_ocr else "_no_ocr"
        ocr_base = f"{file_hash}_{self.ocr_engine}{ocr_tag}"
        ocr_cache_key = hashlib.md5(ocr_base.encode()).hexdigest()

        cache_dir = settings.PARSE_CACHE_DIR
        cache_dir.mkdir(parents=True, exist_ok=True)
        ocr_cache_file = cache_dir / f"{ocr_cache_key}_atoms.json"
        doc_cache_file = cache_dir / f"{ocr_cache_key}_doc.json"

        atoms_data = None
        full_text = None
        dl_doc = None

        # Level-1 hit: atomlar daha önce parse edilmiş
        if ocr_cache_file.exists():
            try:
                with open(ocr_cache_file, "r", encoding="utf-8") as f:
                    ocr_cached = json.load(f)
                atoms_to_check = ocr_cached.get("atoms_data", [])
                has_page_meta = bool(atoms_to_check) and all(
                    "page" in a and "pages" in a for a in atoms_to_check
                )
                if has_page_meta:
                    atoms_data = ocr_cached["atoms_data"]
                    full_text = ocr_cached["full_text"]
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

        if atoms_data is None or (use_hybrid and dl_doc is None):
            print(
                f"  [PARSE] Docling çalıştırılıyor (OCR: {self.ocr_engine}): "
                f"{os.path.basename(file_path)}"
            )
            result = self.converter.convert(file_path)
            dl_doc = result.document

            atoms_data = self._extract_atoms(dl_doc)
            full_text = "\n\n".join(a["text"] for a in atoms_data)

            # Level-1 önbelleğe kaydet
            try:
                with open(ocr_cache_file, "w", encoding="utf-8") as f:
                    json.dump(
                        {"full_text": full_text, "atoms_data": atoms_data},
                        f,
                        ensure_ascii=False,
                        indent=2,
                    )
            except Exception as e:
                print(f"  [WARN] OCR önbellek yazma hatası: {e}")

            if use_hybrid and dl_doc is not None:
                try:
                    with open(doc_cache_file, "w", encoding="utf-8") as f:
                        json.dump(dl_doc.model_dump(mode="json"), f, ensure_ascii=False)
                except Exception as e:
                    print(f"  [WARN] Doc önbellek yazma hatası: {e}")

        markdown_path = self._save_markdown_artifact(file_path, file_hash, full_text)
        pages_by_number = self._build_pages_by_number(atoms_data)
        pages_path = self._save_pages_artifact(file_path, file_hash, pages_by_number)

        return ParsedDocument(
            full_text=full_text,
            atoms=atoms_data,
            dl_doc=dl_doc,
            ocr_base=ocr_base,
            markdown_path=markdown_path,
            pages_path=pages_path,
            pages_by_number=pages_by_number,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_atoms(self, dl_doc) -> List[Dict[str, Any]]:
        """DoclingDocument → atom listesi (metin + etiket + sayfa bilgisi)."""
        atoms_data = []
        try:
            serializer = MarkdownDocSerializer(doc=dl_doc)
            for item, _ in dl_doc.iterate_items():
                content = serializer.serialize(item=item).text
                label = getattr(item, "label", "unknown")
                if content.strip():
                    pages = self._extract_pages(item)
                    atoms_data.append(
                        {
                            "text": content.strip(),
                            "label": str(label),
                            "page": pages[0] if pages else None,
                            "pages": pages,
                        }
                    )
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
                atoms_data.append(
                    {
                        "text": text,
                        "label": str(label),
                        "page": pages[0] if pages else None,
                        "pages": pages,
                    }
                )
        return atoms_data

    @staticmethod
    def _build_pages_by_number(atoms: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Atomları primary page'e göre gruplar; sorted [{sayfaNo, sayfa_markdown}] döner."""
        from collections import defaultdict

        page_buckets: dict[int, list[str]] = defaultdict(list)
        for atom in atoms:
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

    def _save_pages_artifact(
        self, file_path: str, file_hash: str, pages_by_number: List[Dict[str, Any]]
    ) -> str | None:
        """pages_by_number'ı data_lake/pages/ altına sidecar JSON olarak yazar."""
        try:
            pages_dir = settings.PAGES_DIR
            pages_dir.mkdir(parents=True, exist_ok=True)
            source_stem = Path(file_path).stem
            pages_path = pages_dir / f"{source_stem}__{file_hash[:8]}_pages.json"
            if not pages_path.exists():
                pages_path.write_text(
                    json.dumps(pages_by_number, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                print(f"  [PAGES] Artefakt kaydedildi: {pages_path.name}")
            return str(pages_path)
        except Exception as e:
            print(f"  [WARN] Pages artefakt yazma hatası: {e}")
            return None

    def _save_markdown_artifact(
        self, file_path: str, file_hash: str, full_text: str
    ) -> str | None:
        """full_text'i data_lake/markdown/ altına okunabilir .md dosyası olarak yazar."""
        try:
            md_dir = settings.MARKDOWN_DIR
            md_dir.mkdir(parents=True, exist_ok=True)
            source_stem = Path(file_path).stem
            md_path = md_dir / f"{source_stem}__{file_hash[:8]}.md"
            if not md_path.exists():
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


# ---------------------------------------------------------------------------
# CLI: python -m src.common.parsing.markdown_converter --file belge.pdf
# ---------------------------------------------------------------------------

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
  python -m src.common.parsing.markdown_converter --file belge.pdf --pages-json
  python -m src.common.parsing.markdown_converter --file belge.pdf --pages-json > pages.json
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
    args = parser.parse_args()

    conv = MarkdownConverter(ocr_engine=args.ocr_engine, do_ocr=not args.no_ocr)
    parsed = conv.convert(args.file)
    print(f"\nTamamlandı.")
    print(f"  Atom sayısı : {len(parsed.atoms)}")
    print(f"  Sayfa sayısı: {len(parsed.pages_by_number)}")
    print(f"  full_text   : {len(parsed.full_text):,} karakter")
    if parsed.markdown_path:
        print(f"  Artefakt    : {parsed.markdown_path}")
    if parsed.pages_path:
        print(f"  Pages JSON  : {parsed.pages_path}")
    if args.pages_json:
        import json as _json
        print(_json.dumps(parsed.pages_by_number, ensure_ascii=False, indent=2))
    sys.exit(0)
