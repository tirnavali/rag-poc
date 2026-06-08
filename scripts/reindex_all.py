"""Re-run all ingestion pipelines from scratch.

Usage:
  python -m scripts.reindex_all
"""
from src.trainer.press_clips.load_csv import load_csv_to_sqlite
from src.trainer.press_clips.index import embed_and_index as embed_press
from src.trainer.minutes.index import embed_and_index as embed_minutes

if __name__ == "__main__":
    print("=== Step 1/3: Load press clips CSV ===")
    load_csv_to_sqlite()
    print("=== Step 2/3: Embed press clips ===")
    embed_press()
    print("=== Step 3/3: Embed minutes ===")
    embed_minutes()
    print("=== Reindex complete ===")
