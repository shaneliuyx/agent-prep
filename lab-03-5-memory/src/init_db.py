# src/init_db.py
import sqlite3, os
from pathlib import Path
from dotenv import load_dotenv; load_dotenv()

Path(os.getenv("SQLITE_PATH")).parent.mkdir(exist_ok=True)
conn = sqlite3.connect(os.getenv("SQLITE_PATH"))
conn.executescript("""
CREATE TABLE IF NOT EXISTS user_facts (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id    TEXT NOT NULL,
    key        TEXT NOT NULL,       -- e.g. 'location', 'diet', 'name'
    value      TEXT NOT NULL,
    confidence REAL DEFAULT 1.0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    archived   INTEGER DEFAULT 0
    -- NO UNIQUE(user_id, key, archived) — that constraint forbids multiple
    -- archived rows per (user_id, key), which breaks SCD-2 history under
    -- repeated contradictions (location: Osaka → Tokyo → Kyoto). Partial
    -- unique index below enforces uniqueness ONLY on the live (archived=0)
    -- row, allowing unbounded archived history.
);
CREATE INDEX IF NOT EXISTS idx_user_facts_live ON user_facts(user_id, archived);
CREATE UNIQUE INDEX IF NOT EXISTS idx_live_unique
    ON user_facts(user_id, key) WHERE archived = 0;
""")
conn.commit(); conn.close()
print("SQLite initialised")