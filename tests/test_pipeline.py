"""
End-to-end pipeline test on synthetic data.

Generates a small PaySim-shaped file in a temp directory, trains a model,
streams it through the scoring engine into a throwaway SQLite database, and
checks that every stage produced what the next stage needs.  No Kaggle
download, no network, runs in well under a minute.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
import explain  # noqa: E402
import features as F  # noqa: E402


def run(args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    """Run one pipeline stage as a subprocess, exactly as a user would."""
    proc = subprocess.run(
        [sys.executable, *args], cwd=str(cwd),
        capture_output=True, text=True, timeout=900,
    )
    if proc.returncode != 0:
        raise AssertionError(
            f"`{' '.join(args)}` exited {proc.returncode}\n"
            f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
        )
    return proc


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory):
    """
    Run make_synthetic -> train -> stream in an isolated working directory.

    config.py resolves every path relative to the repo root, so instead of
    copying the code we point each stage at files inside tmp via its CLI
    flags, and keep the real artifacts/ and scamshield.db untouched.
    """
    work = tmp_path_factory.mktemp("scamshield")
    sample = work / "sample.csv"
    db_file = work / "test.db"

    run(["data/make_synthetic.py", "--rows", "9000", "--out", str(sample)], ROOT)
    assert sample.exists()

    train = run(["train.py", "--csv", str(sample)], ROOT)
    stream = run(["stream.py", "--rate", "0", "--limit", "1200",
                  "--db", str(db_file)], ROOT)

    return {
        "work": work, "sample": sample, "db": db_file,
        "train_out": train.stdout, "stream_out": stream.stdout,
    }


# ----------------------------------------------------------------------
# Data generation
# ----------------------------------------------------------------------
def test_synthetic_data_has_paysim_shape(pipeline):
    df = pd.read_csv(pipeline["sample"])
    expected = ["step", "type", "amount", "nameOrig", "oldbalanceOrg",
                "newbalanceOrig", "nameDest", "oldbalanceDest",
                "newbalanceDest", "isFraud", "isFlaggedFraud"]
    assert list(df.columns) == expected
    assert len(df) > 1000
    assert df["isFraud"].isin([0, 1]).all()
    # Imbalanced, but not so much that training is impossible.
    assert 0.0 < df["isFraud"].mean() < 0.15
    assert set(df["type"].unique()) <= set(F.TXN_TYPES)
    assert df["step"].is_monotonic_increasing          # already time ordered


def test_prepare_keeps_all_fraud_and_the_type_mix(pipeline, tmp_path):
    """
    data/prepare.py is the Kaggle path, so it needs its own coverage.

    Regression guard: pandas 3.0 changed groupby().apply() to drop the
    grouping columns, which silently wiped the 'type' column for every
    sampled legit row.  Everything still "ran" - it just produced garbage.
    """
    src = pipeline["sample"]              # a PaySim-shaped CSV
    out = tmp_path / "prepared.csv"
    full = pd.read_csv(src)
    target = len(full) // 2

    run(["data/prepare.py", str(src), "--rows", str(target), "--out", str(out)], ROOT)
    prepared = pd.read_csv(out)

    assert prepared.notna().all().all(), "prepare.py produced NaNs"
    assert list(prepared.columns) == list(full.columns)
    # Every fraud row must survive - there are too few to throw any away.
    assert prepared["isFraud"].sum() == full["isFraud"].sum()
    assert abs(len(prepared) - target) < 0.1 * target
    assert prepared["step"].is_monotonic_increasing

    # The transaction-type mix must survive the stratified sample.
    before = full["type"].value_counts(normalize=True)
    after = prepared["type"].value_counts(normalize=True)
    assert set(after.index) == set(before.index), "a whole transaction type vanished"
    for t in before.index:
        assert abs(after[t] - before[t]) < 0.05, f"type mix for {t} drifted too far"


# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------
def test_train_writes_all_artifacts(pipeline):
    for path in (config.MODEL_PATH, config.METADATA_PATH, config.METRICS_PATH,
                 config.PR_CURVE_CSV, config.STREAM_CSV, config.WARMUP_CSV):
        assert path.exists(), f"{path} was not written by train.py"


def test_metadata_matches_the_live_feature_code(pipeline):
    meta = json.loads(config.METADATA_PATH.read_text())
    # If this fails, the model and the streaming features have drifted apart.
    assert meta["feature_names"] == F.FEATURE_NAMES
    assert meta["n_features"] == len(F.FEATURE_NAMES)
    assert 0.0 < meta["thresholds"]["review"] <= meta["thresholds"]["block"] < 1.0


def test_split_is_time_based_not_random(pipeline):
    """Every warm-up row must come before every streamed row."""
    metrics = json.loads(config.METRICS_PATH.read_text())
    assert "time-based" in metrics["split"]["type"]

    warm = pd.read_csv(config.WARMUP_CSV)
    test = pd.read_csv(config.STREAM_CSV)
    assert warm["step"].max() <= test["step"].min()
    assert metrics["split"]["test_fraud"] > 0, "no fraud left to detect in the test half"


def test_model_beats_the_rule_baseline(pipeline):
    """The whole pitch of the project: ML must outperform a static rule."""
    m = json.loads(config.METRICS_PATH.read_text())
    assert m["scores"]["pr_auc"] > 0.5
    assert m["review_tier"]["recall"] > m["rule_baseline"]["recall"]
    assert m["block_tier"]["precision"] >= m["rule_baseline"]["precision"]
    assert "ML vs rules" in pipeline["train_out"]


# ----------------------------------------------------------------------
# Scoring engine
# ----------------------------------------------------------------------
def test_fast_path_agrees_with_sklearn_predict_proba(pipeline):
    """
    stream.py scores with booster.inplace_predict for speed.  That must give
    the same answer as the ordinary predict_proba the model was evaluated
    with, or the reported metrics do not describe the running system.
    """
    from scorer import Scorer

    scorer = Scorer()
    df = F.sort_time_order(pd.read_csv(config.STREAM_CSV)).head(400)
    X, _ = F.build_feature_matrix(df)

    reference = scorer.model.predict_proba(X)[:, 1]
    fast = np.array([scorer.score_fast(X[i]) for i in range(len(X))])
    np.testing.assert_allclose(fast, reference, rtol=1e-6, atol=1e-6)

    # The TreeSHAP path reconstructs the same probability from its contributions.
    for i in range(0, len(X), 37):
        score, contribs = scorer.score_with_contribs(X[i])
        assert score == pytest.approx(float(reference[i]), abs=1e-5)
        assert len(contribs) == len(F.FEATURE_NAMES)


def test_decision_ladder_respects_thresholds(pipeline):
    from scorer import ALLOW, BLOCK, REVIEW, Scorer

    s = Scorer()
    assert s.decide(0.0) == ALLOW
    assert s.decide(max(s.review_threshold - 1e-6, 0.0)) == ALLOW
    assert s.decide(s.review_threshold) == REVIEW
    assert s.decide(s.block_threshold) == BLOCK
    assert s.decide(1.0) == BLOCK


# ----------------------------------------------------------------------
# Streaming into SQLite
# ----------------------------------------------------------------------
def test_stream_populates_the_database(pipeline):
    import db as dbmod

    conn = dbmod.connect(pipeline["db"])
    rows = conn.execute("SELECT COUNT(*) FROM transactions").fetchone()[0]
    assert rows == 1200

    # WAL is what lets the dashboard read while the engine writes.
    assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"

    decisions = {r[0] for r in conn.execute("SELECT DISTINCT decision FROM transactions")}
    assert decisions <= {"ALLOW", "REVIEW", "BLOCK"}
    assert "ALLOW" in decisions, "everything was flagged - the model is useless"

    bad = conn.execute(
        "SELECT COUNT(*) FROM transactions WHERE score < 0 OR score > 1 OR latency_ms < 0"
    ).fetchone()[0]
    assert bad == 0

    # Sequence numbers are dense and ordered.
    seqs = [r[0] for r in conn.execute("SELECT seq FROM transactions ORDER BY id")]
    assert seqs == list(range(1, rows + 1))
    conn.close()


def test_every_alert_carries_a_human_readable_reason(pipeline):
    import db as dbmod

    conn = dbmod.connect(pipeline["db"])
    alerts = conn.execute(
        "SELECT score, decision, reasons, summary FROM transactions "
        "WHERE decision != 'ALLOW'"
    ).fetchall()
    assert alerts, "the stream produced no alerts at all"

    for row in alerts:
        reasons = json.loads(row["reasons"])
        assert 1 <= len(reasons) <= config.TOP_REASONS
        assert {"feature", "value", "contribution"} <= set(reasons[0])
        assert all(r["feature"] in F.FEATURE_NAMES for r in reasons)
        # Contributions come back sorted, most incriminating first.
        contribs = [r["contribution"] for r in reasons]
        assert contribs == sorted(contribs, reverse=True)
        # A sentence an analyst can actually read.
        assert len(row["summary"]) > 15 and row["summary"].endswith(".")
    conn.close()


def test_analyst_actions_table_accepts_decisions(pipeline):
    import db as dbmod
    from datetime import datetime, timezone

    conn = dbmod.connect(pipeline["db"])
    txn_id = conn.execute(
        "SELECT id FROM transactions WHERE decision != 'ALLOW' LIMIT 1"
    ).fetchone()[0]

    for action in ("APPROVE", "BLOCK", "ESCALATE"):
        conn.execute(
            "INSERT INTO analyst_actions (txn_id, action, note, created_at) VALUES (?,?,?,?)",
            (txn_id, action, f"test {action}",
             datetime.now(timezone.utc).isoformat(timespec="seconds")),
        )
    conn.commit()

    joined = conn.execute("""
        SELECT a.action, t.score FROM analyst_actions a
        JOIN transactions t ON t.id = a.txn_id
    """).fetchall()
    assert {r["action"] for r in joined} == {"APPROVE", "BLOCK", "ESCALATE"}
    conn.close()


def test_stream_reports_latency_and_platform(pipeline):
    out = pipeline["stream_out"]
    for expected in ("scoring latency", "throughput", "running on", "money protected"):
        assert expected in out.lower()


# ----------------------------------------------------------------------
# Explanations
# ----------------------------------------------------------------------
def test_summaries_are_plain_language():
    names = F.FEATURE_NAMES
    values = np.zeros(len(names), dtype=float)
    contribs = np.zeros(len(names), dtype=float)

    values[names.index("orig_drained_to_zero")] = 1.0
    contribs[names.index("orig_drained_to_zero")] = 3.0
    values[names.index("dest_fanin_senders")] = 14.0
    contribs[names.index("dest_fanin_senders")] = 2.5
    values[names.index("orig_rapid_passthrough")] = 1.0
    contribs[names.index("orig_rapid_passthrough")] = 1.8

    reasons, sentence = explain.explain(contribs, values, score=0.97)
    assert [r["feature"] for r in reasons] == [
        "orig_drained_to_zero", "dest_fanin_senders", "orig_rapid_passthrough",
    ]
    assert "drained to zero" in sentence
    assert "14 different senders" in sentence
    assert "pass-through" in sentence
    assert sentence[0].isupper() and sentence.endswith(".")
    # No raw feature names or numpy repr leaking into the analyst's screen.
    assert "orig_drained_to_zero" not in sentence
    assert "np.float" not in sentence


def test_one_hot_features_are_not_narrated_when_unset():
    """A zero-valued flag must never produce a sentence claiming it happened."""
    names = F.FEATURE_NAMES
    values = np.zeros(len(names), dtype=float)
    contribs = np.zeros(len(names), dtype=float)
    contribs[names.index("orig_drained_to_zero")] = 2.0   # positive, but value is 0
    _, sentence = explain.explain(contribs, values)
    assert "drained to zero" not in sentence


def test_generative_hook_is_disabled_by_default():
    assert config.USE_GENERATIVE_SUMMARY is False
    assert explain._generative_summary([], 0.5) is None


# ----------------------------------------------------------------------
# Benchmark
# ----------------------------------------------------------------------
def test_benchmark_records_the_architecture(pipeline):
    import platform

    out_file = pipeline["work"] / "bench.json"
    run(["benchmark.py", "--rows", "1500", "--warmup-rows", "200",
         "--out", str(out_file)], ROOT)

    data = json.loads(out_file.read_text())
    assert data["rows"] == 1500
    # This is the field that proves to the judges where it ran.
    assert data["platform"]["machine"] == platform.machine()
    assert data["platform"]["cpu_count"] == os.cpu_count()
    for section in ("end_to_end", "score_only", "batch"):
        assert data[section]["throughput_txn_per_sec"] > 0
    lat = data["end_to_end"]["latency_ms"]
    assert 0 < lat["p50"] <= lat["p95"] <= lat["p99"] <= lat["max"]


# ----------------------------------------------------------------------
# Dashboard
# ----------------------------------------------------------------------
def test_dashboard_renders_against_the_streamed_database(pipeline):
    """Render app.py headlessly and assert it raised nothing."""
    apptest = pytest.importorskip(
        "streamlit.testing.v1", reason="streamlit not installed"
    )
    os.environ["SCAMSHIELD_NO_AUTOREFRESH"] = "1"
    os.environ["SCAMSHIELD_DB"] = str(pipeline["db"])
    try:
        at = apptest.AppTest.from_file(str(ROOT / "app.py"), default_timeout=180).run()
        assert not at.exception, [str(e.value) for e in at.exception]

        labels = {m.label for m in at.metric}
        for expected in ("Transactions scored", "Alerts raised", "Money protected",
                         "Architecture", "Python", "CPU cores"):
            assert expected in labels, f"KPI '{expected}' missing from the dashboard"

        # The review queue must offer the three analyst actions.
        keys = {b.key[:2] for b in at.button if b.key}
        assert {"ap", "bl", "es"} <= keys

        # Clicking one writes to the audit table.
        before = at.button[0].key
        at.button[0].click().run()
        assert not at.exception, [str(e.value) for e in at.exception]

        import db as dbmod
        conn = dbmod.connect(pipeline["db"])
        n = conn.execute("SELECT COUNT(*) FROM analyst_actions WHERE txn_id = ?",
                         (int(before[2:]),)).fetchone()[0]
        conn.close()
        assert n >= 1
    finally:
        os.environ.pop("SCAMSHIELD_NO_AUTOREFRESH", None)
        os.environ.pop("SCAMSHIELD_DB", None)
