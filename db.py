"""
ScamShield - SQLite storage shared by the streaming engine and the dashboard.

Why SQLite?  Because it is in the Python standard library, needs no server,
and in WAL mode it lets the dashboard READ while the stream keeps WRITING -
which is the whole point of a live demo.  On LinuxONE that means zero extra
installation.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS transactions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    seq         INTEGER NOT NULL,          -- position in the replayed stream
    step        INTEGER NOT NULL,          -- PaySim simulated hour
    type        TEXT    NOT NULL,
    amount      REAL    NOT NULL,
    name_orig   TEXT    NOT NULL,
    name_dest   TEXT    NOT NULL,
    score       REAL    NOT NULL,          -- fraud probability 0..1
    decision    TEXT    NOT NULL,          -- ALLOW | REVIEW | BLOCK
    reasons     TEXT    NOT NULL,          -- JSON list of top contributions
    summary     TEXT    NOT NULL,          -- plain-language sentence
    latency_ms  REAL    NOT NULL,          -- features + score, per transaction
    is_fraud    INTEGER NOT NULL,          -- ground truth, demo only
    created_at  TEXT    NOT NULL,
    decision_mode TEXT  NOT NULL DEFAULT 'MODEL',  -- MODEL | DEGRADED
    expected_loss REAL  NOT NULL DEFAULT 0,        -- score * amount
    ring_size   INTEGER NOT NULL DEFAULT 1         -- mule cluster size at scoring time
);

CREATE INDEX IF NOT EXISTS idx_txn_id       ON transactions(id DESC);
CREATE INDEX IF NOT EXISTS idx_txn_decision ON transactions(decision);
CREATE INDEX IF NOT EXISTS idx_txn_created  ON transactions(created_at);
-- The review queue is ordered by money at risk, so give it an index.
CREATE INDEX IF NOT EXISTS idx_txn_exploss  ON transactions(expected_loss DESC);

CREATE TABLE IF NOT EXISTS analyst_actions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    txn_id      INTEGER NOT NULL,
    action      TEXT    NOT NULL,          -- APPROVE | BLOCK | ESCALATE
    note        TEXT    NOT NULL DEFAULT '',
    created_at  TEXT    NOT NULL,
    FOREIGN KEY (txn_id) REFERENCES transactions(id)
);

CREATE INDEX IF NOT EXISTS idx_action_txn ON analyst_actions(txn_id);

-- Free-form key/value facts about the current run (platform, thresholds, ...)
CREATE TABLE IF NOT EXISTS stream_meta (
    key        TEXT PRIMARY KEY,
    value      TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""


# Columns added after the first version of the schema shipped.  CREATE TABLE
# IF NOT EXISTS will not add them to a database that already exists, so a
# database recorded by an earlier run (for example the LinuxONE evidence file)
# has to be nudged forward.  ALTER TABLE ... ADD COLUMN with a default is
# instant and does not touch any existing row's data.
_MIGRATIONS = [
    ("decision_mode", "TEXT NOT NULL DEFAULT 'MODEL'"),
    ("expected_loss", "REAL NOT NULL DEFAULT 0"),
    ("ring_size", "INTEGER NOT NULL DEFAULT 1"),
]


def _migrate(conn: sqlite3.Connection) -> None:
    """Add any missing columns to an older transactions table."""
    try:
        existing = {r["name"] for r in conn.execute("PRAGMA table_info(transactions)")}
    except sqlite3.Error:
        return
    if not existing:
        return
    for column, ddl in _MIGRATIONS:
        if column not in existing:
            try:
                conn.execute(f"ALTER TABLE transactions ADD COLUMN {column} {ddl}")
            except sqlite3.Error:
                # A read-only file or a concurrent writer: the dashboard
                # degrades to not showing that column rather than crashing.
                pass


def connect(path: Path | str = config.DB_PATH, *, read_only: bool = False) -> sqlite3.Connection:
    """
    Open the database in WAL mode.

    WAL is what lets Streamlit poll the table twice a second without ever
    blocking the writer (and without the writer blocking the dashboard).
    """
    conn = sqlite3.connect(str(path), timeout=30.0, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    # NORMAL is the right trade-off here: durable across process crashes,
    # and it does not fsync on every single commit (which would cap us at
    # a few hundred transactions per second on spinning storage).
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=30000")
    if not read_only:
        # Order matters.  _migrate() must run FIRST, because SCHEMA contains an
        # index over expected_loss: on a database recorded before that column
        # existed, CREATE INDEX fails with "no such column" and takes the whole
        # connection down before the migration ever gets a chance to add it.
        # _migrate() is a no-op on a brand-new file (no transactions table yet),
        # so running it first is safe in both directions.
        _migrate(conn)
        conn.executescript(SCHEMA)
        conn.commit()
    return conn


def reset(path: Path | str = config.DB_PATH) -> None:
    """Drop everything and recreate an empty database (start of a fresh demo)."""
    p = Path(path)
    for suffix in ("", "-wal", "-shm"):
        f = Path(str(p) + suffix)
        if f.exists():
            f.unlink()
    conn = connect(p)
    conn.close()


def set_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
    from datetime import datetime, timezone
    conn.execute(
        "INSERT INTO stream_meta(key, value, updated_at) VALUES (?, ?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
        (key, str(value), datetime.now(timezone.utc).isoformat(timespec="seconds")),
    )


def get_meta(conn: sqlite3.Connection) -> dict[str, str]:
    try:
        return {r["key"]: r["value"] for r in conn.execute("SELECT key, value FROM stream_meta")}
    except sqlite3.Error:
        return {}
