import sqlite3
import json
import os
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

DB_PATH = Path("data_lake/rag_sessions.db")

def get_connection():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_connection()
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            title TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            role TEXT,
            content TEXT,
            sources TEXT,
            trace TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES sessions (id)
        )
    ''')
    # Safe migration to add memory column if it's missing
    try:
        c.execute("ALTER TABLE messages ADD COLUMN memory TEXT")
    except sqlite3.OperationalError:
        pass
    # Safe migration to add suggestions (rabbit-hole drill-downs) column
    try:
        c.execute("ALTER TABLE messages ADD COLUMN suggestions TEXT")
    except sqlite3.OperationalError:
        pass
    # Global (not session-scoped) queue of LLM-discovered vocabulary-synonym
    # hypotheses (e.g. "kadük" -> "hükümsüz sayılan kanun teklifleri") awaiting
    # human approve/reject via the "Öğrenilen Terimler" web UI panel.
    c.execute('''
        CREATE TABLE IF NOT EXISTS term_candidates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            term TEXT NOT NULL,
            hypothesis TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            times_seen INTEGER NOT NULL DEFAULT 1,
            last_source_query TEXT,
            last_session_id TEXT,
            first_seen_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_seen_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            reviewed_at TIMESTAMP,
            UNIQUE(term, hypothesis)
        )
    ''')
    conn.commit()
    conn.close()

def create_session(session_id: str, title: str = "Yeni Sohbet") -> str:
    conn = get_connection()
    c = conn.cursor()
    c.execute("INSERT INTO sessions (id, title) VALUES (?, ?)", (session_id, title))
    conn.commit()
    conn.close()
    return session_id

def get_sessions() -> List[Dict[str, Any]]:
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT id, title, created_at FROM sessions ORDER BY created_at DESC")
    rows = c.fetchall()
    conn.close()
    return [dict(row) for row in rows]

def add_message(
    session_id: str,
    role: str,
    content: str,
    sources: Optional[List[Dict]] = None,
    trace: Optional[List[Dict]] = None,
    memory: Optional[List[Dict]] = None,
    suggestions: Optional[List[str]] = None
):
    conn = get_connection()
    c = conn.cursor()
    
    # Update title if it's the first user message
    if role == "user":
        c.execute("SELECT COUNT(*) FROM messages WHERE session_id = ?", (session_id,))
        if c.fetchone()[0] == 0:
            title = content[:30] + "..." if len(content) > 30 else content
            c.execute("UPDATE sessions SET title = ? WHERE id = ?", (title, session_id))

    sources_json = json.dumps(sources) if sources else None
    trace_json = json.dumps(trace) if trace else None
    memory_json = json.dumps(memory) if memory else None
    suggestions_json = json.dumps(suggestions) if suggestions else None

    c.execute('''
        INSERT INTO messages (session_id, role, content, sources, trace, memory, suggestions)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (session_id, role, content, sources_json, trace_json, memory_json, suggestions_json))
    conn.commit()
    conn.close()

def get_messages(session_id: str) -> List[Dict[str, Any]]:
    conn = get_connection()
    c = conn.cursor()
    c.execute('''
        SELECT role, content, sources, trace, memory, suggestions, created_at
        FROM messages
        WHERE session_id = ?
        ORDER BY id ASC
    ''', (session_id,))
    rows = c.fetchall()
    conn.close()

    messages = []
    for r in rows:
        msg = dict(r)
        msg['sources'] = json.loads(msg['sources']) if msg['sources'] else None
        msg['trace'] = json.loads(msg['trace']) if msg['trace'] else None
        msg['memory'] = json.loads(msg['memory']) if msg['memory'] else None
        msg['suggestions'] = json.loads(msg['suggestions']) if msg['suggestions'] else None
        messages.append(msg)
    return messages

# --------------------------------------------------------------- term_candidates

def upsert_term_candidate(
    term: str,
    hypothesis: str,
    source_query: Optional[str] = None,
    source_session_id: Optional[str] = None,
) -> None:
    """Record a successful re-query term hypothesis for human review.

    Global, not session-scoped — the same (term, hypothesis) pair discovered
    independently by different queries/sessions increments `times_seen` rather
    than creating duplicate rows, so a reviewer can see "this guess proved
    itself N times" as a confidence signal (not an auto-approval).
    """
    conn = get_connection()
    c = conn.cursor()
    c.execute('''
        INSERT INTO term_candidates (term, hypothesis, last_source_query, last_session_id)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(term, hypothesis) DO UPDATE SET
            times_seen = times_seen + 1,
            last_seen_at = CURRENT_TIMESTAMP,
            last_source_query = excluded.last_source_query,
            last_session_id = excluded.last_session_id
    ''', (term, hypothesis, source_query, source_session_id))
    conn.commit()
    conn.close()


def list_term_candidates(status: str = "pending") -> List[Dict[str, Any]]:
    conn = get_connection()
    c = conn.cursor()
    c.execute('''
        SELECT id, term, hypothesis, status, times_seen, last_source_query,
               last_session_id, first_seen_at, last_seen_at, reviewed_at
        FROM term_candidates
        WHERE status = ?
        ORDER BY times_seen DESC, last_seen_at DESC
    ''', (status,))
    rows = c.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def set_term_candidate_status(candidate_id: int, status: str) -> bool:
    """Returns False if no row with this id exists (caller can 404)."""
    conn = get_connection()
    c = conn.cursor()
    c.execute(
        "UPDATE term_candidates SET status = ?, reviewed_at = CURRENT_TIMESTAMP WHERE id = ?",
        (status, candidate_id),
    )
    updated = c.rowcount > 0
    conn.commit()
    conn.close()
    return updated


def list_approved_term_synonyms() -> Dict[str, List[str]]:
    """Approved (term -> [hypothesis, ...]) map for live synonym expansion —
    the counterpart to settings.PARLIAMENTARY_TERM_SYNONYMS, but DB-backed so
    an approval takes effect without a code change/redeploy."""
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT term, hypothesis FROM term_candidates WHERE status = 'approved'")
    rows = c.fetchall()
    conn.close()
    result: Dict[str, List[str]] = {}
    for row in rows:
        result.setdefault(row["term"], []).append(row["hypothesis"])
    return result


def list_rejected_term_hypotheses() -> List[Dict[str, str]]:
    """Rejected {term, hypothesis} pairs, fed back into the re-query prompt as
    a negative constraint so the same debunked guess isn't proposed again."""
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT term, hypothesis FROM term_candidates WHERE status = 'rejected'")
    rows = c.fetchall()
    conn.close()
    return [{"term": row["term"], "hypothesis": row["hypothesis"]} for row in rows]


# Initialize on import
init_db()
