"""
Opening a database recorded by an OLDER version of ScamShield must still work.

This is not hypothetical.  `linuxone_results/scamshield.db` is the evidence
file from the IBM LinuxONE run, recorded before decision_mode, expected_loss
and ring_size existed.  Pointing the dashboard at it crashed with
"no such column: expected_loss", because connect() ran the schema script -
which creates an index over expected_loss - BEFORE the migration that adds
the column.  The demo most worth keeping was the one that broke.
"""
from __future__ import annotations

import sqlite3

import db as dbmod

# The transactions table exactly as the first released version wrote it.
OLD_SCHEMA = """
CREATE TABLE transactions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    seq         INTEGER NOT NULL,
    step        INTEGER NOT NULL,
    type        TEXT    NOT NULL,
    amount      REAL    NOT NULL,
    name_orig   TEXT    NOT NULL,
    name_dest   TEXT    NOT NULL,
    score       REAL    NOT NULL,
    decision    TEXT    NOT NULL,
    reasons     TEXT    NOT NULL,
    summary     TEXT    NOT NULL,
    latency_ms  REAL    NOT NULL,
    is_fraud    INTEGER NOT NULL,
    created_at  TEXT    NOT NULL
);
CREATE TABLE analyst_actions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    txn_id      INTEGER NOT NULL,
    action      TEXT    NOT NULL,
    note        TEXT    NOT NULL DEFAULT '',
    created_at  TEXT    NOT NULL
);
CREATE TABLE stream_meta (
    key        TEXT PRIMARY KEY,
    value      TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""

NEW_COLUMNS = {"decision_mode", "expected_loss", "ring_size"}


def _write_old_database(path) -> None:
    conn = sqlite3.connect(str(path))
    conn.executescript(OLD_SCHEMA)
    conn.execute(
        "INSERT INTO transactions (seq, step, type, amount, name_orig, "
        "name_dest, score, decision, reasons, summary, latency_ms, is_fraud, "
        "created_at) VALUES (1, 7, 'TRANSFER', 4200.0, 'C1', 'C2', 0.97, "
        "'BLOCK', '[]', 'Looked wrong.', 0.42, 1, '2026-01-01T00:00:00+00:00')"
    )
    conn.execute(
        "INSERT INTO stream_meta (key, value, updated_at) "
        "VALUES ('machine', 's390x', '2026-01-01T00:00:00+00:00')"
    )
    conn.commit()
    conn.close()


def test_old_database_opens_and_gains_the_new_columns(tmp_path):
    path = tmp_path / "old.db"
    _write_old_database(path)

    conn = dbmod.connect(path)          # must not raise
    columns = {r["name"] for r in conn.execute("PRAGMA table_info(transactions)")}
    assert NEW_COLUMNS <= columns

    # The existing row survives untouched and picks up sane defaults.
    row = conn.execute("SELECT * FROM transactions").fetchone()
    assert row["decision"] == "BLOCK"
    assert row["score"] == 0.97
    assert row["decision_mode"] == "MODEL"
    assert row["expected_loss"] == 0
    assert row["ring_size"] == 1

    # And the index the new schema wants now exists.
    indexes = {r[1] for r in conn.execute("PRAGMA index_list(transactions)")}
    assert "idx_txn_exploss" in indexes

    assert dbmod.get_meta(conn)["machine"] == "s390x"
    conn.close()


def test_the_queries_the_dashboard_runs_work_after_migration(tmp_path):
    """Opening is not enough - the dashboard's own SQL has to run too."""
    path = tmp_path / "old.db"
    _write_old_database(path)
    conn = dbmod.connect(path)

    # The three queries that touch the added columns.
    conn.execute("SELECT SUM(decision_mode = 'DEGRADED'), MAX(ring_size) "
                 "FROM transactions").fetchone()
    conn.execute("SELECT id FROM transactions WHERE decision IN ('REVIEW','BLOCK') "
                 "ORDER BY expected_loss DESC LIMIT 5").fetchall()
    conn.execute("SELECT name_dest FROM transactions WHERE ring_size > 2 "
                 "GROUP BY name_dest").fetchall()
    conn.close()


def test_migrating_twice_is_harmless(tmp_path):
    """connect() runs on every dashboard refresh, so it must be idempotent."""
    path = tmp_path / "old.db"
    _write_old_database(path)
    for _ in range(3):
        conn = dbmod.connect(path)
        conn.close()
    conn = dbmod.connect(path)
    assert conn.execute("SELECT COUNT(*) FROM transactions").fetchone()[0] == 1
    conn.close()


def test_a_brand_new_database_is_unaffected(tmp_path):
    """The migration must not get in the way of creating a fresh database."""
    path = tmp_path / "fresh.db"
    conn = dbmod.connect(path)
    columns = {r["name"] for r in conn.execute("PRAGMA table_info(transactions)")}
    assert NEW_COLUMNS <= columns
    assert conn.execute("SELECT COUNT(*) FROM transactions").fetchone()[0] == 0
    conn.close()
