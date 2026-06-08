"""SQLite helpers for opening read-only connections."""
from __future__ import annotations

import sqlite3
from pathlib import Path


def connect(db_path: Path | str) -> sqlite3.Connection:
    return sqlite3.connect(str(db_path), check_same_thread=False)
