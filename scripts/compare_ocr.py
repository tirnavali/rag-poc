#!/usr/bin/env python3
"""
Compare OCR engine output quality on a PDF.

Usage:
    python scripts/compare_ocr.py <pdf_path>
    python scripts/compare_ocr.py <pdf_path> easyocr,tesseract,mac

Output files written to output/ directory:
    output/<stem>_easyocr.txt
    output/<stem>_tesseract.txt
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.parsing.docling_manager import DoclingManager


def compare(pdf_path: str, engines: list[str]) -> None:
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    pdf_stem = Path(pdf_path).stem
    results = {}

    for engine in engines:
        print(f"\n{'='*50}")
        print(f"Engine: {engine.upper()}")
        print(f"{'='*50}")
        try:
            dm = DoclingManager(ocr_engine=engine)
            text, chunks = dm.convert_and_pack(pdf_path)
            out_path = output_dir / f"{pdf_stem}_{engine}.txt"
            out_path.write_text(text, encoding="utf-8")

            results[engine] = {"chars": len(text), "chunks": len(chunks), "path": out_path}
            print(f"Chars   : {len(text)}")
            print(f"Chunks  : {len(chunks)}")
            print(f"Saved   : {out_path}")
            print(f"\nPreview (first 500 chars):\n{text[:500]}")
        except Exception as e:
            results[engine] = {"error": str(e)}
            print(f"FAILED  : {e}")

    print(f"\n{'='*50}")
    print("SUMMARY")
    print(f"{'='*50}")
    for engine, r in results.items():
        if "error" in r:
            print(f"  {engine:12s} FAILED: {r['error']}")
        else:
            print(f"  {engine:12s} {r['chars']:>8,} chars  {r['chunks']:>4} chunks  → {r['path']}")
    print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    pdf = sys.argv[1]
    engines = sys.argv[2].split(",") if len(sys.argv) > 2 else ["easyocr", "tesseract"]
    compare(pdf, engines)
