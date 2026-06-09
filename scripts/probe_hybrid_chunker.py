"""Manuel Docling probe — HybridChunker charspan provenance'ını test eder.

Kullanım:
    python -m scripts.probe_hybrid_chunker <pdf_path>

Bu bir pytest testi DEĞİLDİR; elle çalıştırılan bir araştırma script'idir.
"""

import os
import sys
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions, EasyOcrOptions
from docling.datamodel.base_models import InputFormat
from docling.chunking import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from docling_core.transforms.serializer.markdown import MarkdownDocSerializer

if len(sys.argv) < 2:
    print("Kullanım: python -m scripts.probe_hybrid_chunker <pdf_path>")
    sys.exit(1)

pdf_path = sys.argv[1]

if not os.path.exists(pdf_path):
    print(f"Error: File not found at {pdf_path}")
    sys.exit(1)

print(f"Testing HybridChunker on: {os.path.basename(pdf_path)}")

ocr_options = EasyOcrOptions(lang=["tr"], use_gpu=False)
pipeline_options = PdfPipelineOptions(do_ocr=True, ocr_options=ocr_options)
converter = DocumentConverter(
    format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
)

print("Converting document...")
result = converter.convert(pdf_path)
dl_doc = result.document

# Docling'in kendi full_text'ini al (charspan bu koordinat sistemine göre)
serializer = MarkdownDocSerializer(doc=dl_doc)
full_text = serializer.serialize().text
print(f"full_text length: {len(full_text)} chars\n")

# Güncel API: deprecated string tokenizer yerine HuggingFaceTokenizer
tokenizer = HuggingFaceTokenizer.from_pretrained(
    model_name="jinaai/jina-embeddings-v3",
    max_tokens=8192,
)
chunker = HybridChunker(tokenizer=tokenizer)

total_chunks = 0
total_items = 0
missing_prov = 0
zero_span = 0
span_text_mismatch = 0
gap_chunks = 0  # multi-item chunk'larda gap var mı

for chunk in chunker.chunk(dl_doc):
    total_chunks += 1
    chunk_spans = []

    for item in chunk.meta.doc_items:
        total_items += 1
        prov = getattr(item, "prov", [])

        if not prov:
            missing_prov += 1
            continue

        for p in prov:
            cs = p.charspan
            if cs == (0, 0):
                zero_span += 1
                continue

            chunk_spans.append(cs)

            # KRİTİK: charspan koordinatı full_text'e karşılık geliyor mu?
            extracted = full_text[cs[0]:cs[1]]
            item_text = getattr(item, "text", "") or ""
            # Tam eşleşme zor (serializer whitespace ekleyebilir),
            # en azından item_text'in ilk 30 char'ı extracted içinde olmalı
            snippet = item_text.strip()[:30]
            if snippet and snippet not in extracted and snippet not in full_text[max(0,cs[0]-20):cs[1]+20]:
                span_text_mismatch += 1
                if span_text_mismatch <= 3:
                    print(f"[MISMATCH] charspan=({cs[0]},{cs[1]})")
                    print(f"  extracted : {extracted[:60]!r}")
                    print(f"  item.text : {item_text[:60]!r}")

    # Gap kontrolü: çoklu item'ların span'ları arasında boşluk var mı?
    if len(chunk_spans) > 1:
        chunk_spans_sorted = sorted(chunk_spans, key=lambda x: x[0])
        for i in range(len(chunk_spans_sorted) - 1):
            gap = chunk_spans_sorted[i+1][0] - chunk_spans_sorted[i][1]
            if gap > 50:  # 50 char'dan büyük gap
                gap_chunks += 1
                if gap_chunks <= 2:
                    print(f"[GAP] Chunk #{total_chunks}: gap={gap} chars between spans")
                    print(f"  Span A ends  : {full_text[max(0,chunk_spans_sorted[i][1]-30):chunk_spans_sorted[i][1]]!r}")
                    print(f"  Span B starts: {full_text[chunk_spans_sorted[i+1][0]:chunk_spans_sorted[i+1][0]+30]!r}")
                break

print(f"\n{'='*50}")
print(f"Total chunks       : {total_chunks}")
print(f"Total doc_items    : {total_items}")
print(f"Missing prov       : {missing_prov}  {'⚠️' if missing_prov else '✓'}")
print(f"Zero charspan      : {zero_span}  {'⚠️' if zero_span else '✓'}")
print(f"Span-text mismatch : {span_text_mismatch}  {'⚠️' if span_text_mismatch else '✓'}")
print(f"Gap chunks (>50ch) : {gap_chunks}  {'⚠️' if gap_chunks else '✓'}")

print(f"\n{'='*50}")
if missing_prov == 0 and zero_span == 0 and span_text_mismatch == 0:
    print("RESULT: HybridChunker charspan KULLANILABILIR → geçiş güvenli")
elif missing_prov > total_items * 0.1 or zero_span > total_items * 0.1:
    print("RESULT: Provenance EKSİK → OCR'd PDF için mevcut yöntem kalmalı")
else:
    print("RESULT: Kısmi sorun — detaylara bak")

if gap_chunks > 0:
    print(f"NOT: {gap_chunks} chunk'ta gap var → late chunking vektörü kirlenebilir")
