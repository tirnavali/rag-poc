"""
SQLite smoke tests for the TBMM minutes database (parliament_digital_born_minutes.db).

Checks:
  1. DB file exists and is readable
  2. parliament_minutes table exists with expected columns
  3. parliament_minutes_fts virtual table exists
  4. Row count is non-zero
  5. All rows have a non-empty date
  6. All rows have a non-null speaker / speaker_name
  7. Rich speaker fields (speaker_role, city, party) are present in at least some rows
  8. FTS search by speaker name works (SELMAN OĞUZHAN ESER)
  9. FTS search by content keyword works
 10. Date-filtered query works (2026-03-11)
 11. speaker_name and party columns exist (new schema)
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

# Allow running from project root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import settings

DB_PATH = settings.MINUTES_SQLITE

PASS = "[PASS]"
FAIL = "[FAIL]"


def check(label: str, condition: bool, detail: str = "") -> bool:
    status = PASS if condition else FAIL
    suffix = f"  -> {detail}" if detail else ""
    print(f"  {status}  {label}{suffix}")
    return condition


def run_all() -> int:
    failures = 0

    print(f"\n{'='*60}")
    print(f"  DB : {DB_PATH}")
    print(f"{'='*60}\n")

    # 1. DB file exists
    exists = DB_PATH.exists()
    if not check("DB file exists", exists, str(DB_PATH)):
        print("\n  Cannot continue – DB file missing.\n")
        return 1

    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # 2. Table exists
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='parliament_minutes'")
    failures += not check("parliament_minutes table exists", cur.fetchone() is not None)

    # 3. FTS virtual table exists
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='parliament_minutes_fts'")
    failures += not check("parliament_minutes_fts FTS table exists", cur.fetchone() is not None)

    # 4. Row count > 0
    cur.execute("SELECT COUNT(*) FROM parliament_minutes")
    count = cur.fetchone()[0]
    failures += not check("Row count > 0", count > 0, f"{count:,} rows")

    # 5. Expected columns present (new schema)
    cur.execute("PRAGMA table_info(parliament_minutes)")
    cols = {row["name"] for row in cur.fetchall()}
    for col in ("id", "term", "date", "speaker", "content",
                "speaker_name", "speaker_role", "city", "party",
                "section_title", "speech_order", "chunk_header"):
        failures += not check(f"Column '{col}' exists", col in cols)

    # 6. No row with empty tarih
    cur.execute("SELECT COUNT(*) FROM parliament_minutes WHERE date IS NULL OR date = ''")
    null_date = cur.fetchone()[0]
    failures += not check("All rows have non-empty date", null_date == 0,
                          f"{null_date} rows with empty date")

    # 7. No row with empty speaker_name
    cur.execute("SELECT COUNT(*) FROM parliament_minutes WHERE speaker_name IS NULL OR speaker_name = ''")
    null_spk = cur.fetchone()[0]
    failures += not check("All rows have non-empty speaker_name", null_spk == 0,
                          f"{null_spk} rows with empty speaker_name")

    # 8. At least some rows have a party value (rich parsing)
    cur.execute("SELECT COUNT(*) FROM parliament_minutes WHERE party IS NOT NULL AND party != ''")
    with_party = cur.fetchone()[0]
    failures += not check("At least some rows have party info", with_party > 0,
                          f"{with_party} rows with party")

    # 9. FTS search: SELMAN OĞUZHAN ESER
    try:
        cur.execute(
            "SELECT COUNT(*) FROM parliament_minutes_fts WHERE speaker_name MATCH ?",
            ("ESER",)
        )
        eser_hits = cur.fetchone()[0]
        failures += not check(
            "FTS author search 'ESER' returns results",
            eser_hits > 0,
            f"{eser_hits} rows"
        )
    except Exception as e:
        failures += not check("FTS author search 'ESER'", False, str(e))

    # 10. FTS search: content keyword
    try:
        cur.execute(
            "SELECT COUNT(*) FROM parliament_minutes_fts WHERE content MATCH ?",
            ("milli",)
        )
        kw_hits = cur.fetchone()[0]
        failures += not check(
            "FTS content search 'milli' returns results",
            kw_hits > 0,
            f"{kw_hits} rows"
        )
    except Exception as e:
        failures += not check("FTS content search 'milli'", False, str(e))

    # 11. Date-filtered query: 2026-03-11
    cur.execute(
        "SELECT COUNT(*) FROM parliament_minutes WHERE date = ?",
        ("2026-03-11",)
    )
    date_hits = cur.fetchone()[0]
    failures += not check(
        "Date-filtered query (2026-03-11) returns results",
        date_hits > 0,
        f"{date_hits} rows"
    )

    # 12. Sample data preview
    print("\n  --- Sample rows (top 3) ---")
    cur.execute(
        "SELECT id, date, speaker_name, speaker_role, party, city, LENGTH(content) as clen "
        "FROM parliament_minutes ORDER BY id LIMIT 3"
    )
    for row in cur.fetchall():
        print(f"    id={row['id']} | {row['tarih']} | {row['speaker_name']!r} "
              f"| role={row['speaker_role']} | party={row['party']} "
              f"| city={row['city']} | content_len={row['clen']}")

    # 13. ESER rows detail
    print("\n  --- SELMAN OĞUZHAN ESER rows ---")
    cur.execute(
        "SELECT id, date, speaker_name, party, city, speech_order "
        "FROM parliament_minutes WHERE speaker_name LIKE '%ESER%' LIMIT 5"
    )
    rows = cur.fetchall()
    if rows:
        for row in rows:
            print(f"    id={row['id']} | {row['tarih']} | {row['speaker_name']!r} "
                  f"| party={row['party']} | city={row['city']} | order={row['speech_order']}")
    else:
        print("    (no ESER rows found)")

    conn.close()

    print(f"\n{'='*60}")
    if failures:
        print(f"  {FAIL}  {failures} test(s) failed.")
    else:
        print(f"  {PASS}  All tests passed.")
    print(f"{'='*60}\n")

    return failures


if __name__ == "__main__":
    sys.exit(run_all())
