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
    memory: Optional[List[Dict]] = None
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
    
    c.execute('''
        INSERT INTO messages (session_id, role, content, sources, trace, memory)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (session_id, role, content, sources_json, trace_json, memory_json))
    conn.commit()
    conn.close()

def get_messages(session_id: str) -> List[Dict[str, Any]]:
    conn = get_connection()
    c = conn.cursor()
    c.execute('''
        SELECT role, content, sources, trace, memory, created_at 
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
        messages.append(msg)
    return messages

# Initialize on import
init_db()
