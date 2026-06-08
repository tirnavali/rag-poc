"""Load newspaper CSV into SQLite (press_clips.db)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import sqlite3

from src.config import settings


def load_csv_to_sqlite(
    csv_file: Path = settings.PRESS_CSV,
    db_file: Path = settings.PRESS_SQLITE,
) -> None:
    print(f"--- {csv_file} okunuyor ---")
    if not csv_file.exists():
        print(f"HATA: {csv_file} bulunamadı!")
        return

    db_file.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_file)
    df = df.fillna("")

    if "TARIH" in df.columns:
        df["TARIH"] = pd.to_datetime(df["TARIH"], errors="coerce").dt.strftime("%Y-%m-%d")
        df["TARIH"] = df["TARIH"].fillna("")

    print(f"--- SQLite veritabanına bağlanılıyor: {db_file} ---")
    conn = sqlite3.connect(str(db_file))
    df.to_sql("kupurler", conn, if_exists="replace", index=False)
    print(f"--- Başarıyla {len(df)} kayıt SQLite'a taşındı ---")
    conn.close()


if __name__ == "__main__":
    load_csv_to_sqlite()
