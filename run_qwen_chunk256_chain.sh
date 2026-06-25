#!/bin/bash
set -e
cd /home/tbmmai/RAG-poc

# Benchmark bitene kadar bekle
echo "[$(date)] benchmark bitmesini bekliyorum..."
while kill -0 2590572 2>/dev/null; do sleep 5; done
echo "[$(date)] benchmark bitti, ingest başlıyor..."

# qwen_chunk256 ingest
.venv/bin/python -m src.trainer.ingestion.ingest \
    --request ingest_qwen_chunk256.json \
    --force \
    2>&1 | tee artifacts/ingest_qwen_chunk256.log

echo "[$(date)] ingest tamamlandı."
