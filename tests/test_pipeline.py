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


def run(args: list[str], cwd: Path, home: Path | None = None) -> subprocess.CompletedProcess:
    """Run one pipeline stage as a subprocess, exactly as a user would."""
    env = dict(os.environ)
    if home is not None:
        env["SCAMSHIELD_HOME"] = str(home)
    proc = subprocess.run(
        [sys.executable, *args], cwd=str(cwd), env=env,
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
    Run make_synthetic -> train -> stream against an isolated SCAMSHIELD_HOME.

    Every stage writes its artifacts and generated feeds under a temp
    directory, so running the test suite can never overwrite the model,
    feeds or database you built for the demo.  (It used to: `pytest` would
    retrain into artifacts/ with 9k rows and silently replace a full-size
    model, which then made stream.py replay the wrong feed.)
    """
    work = tmp_path_factory.mktemp("scamshield")
    home = work / "home"
    (home / "data").mkdir(parents=True)
    (home / "artifacts").mkdir(parents=True)
    sample = work / "sample.csv"
    db_file = work / "test.db"

    run(["data/make_synthetic.py", "--rows", "9000", "--out", str(sample)], ROOT, home)
    assert sample.exists()

    train = run(["train.py", "--csv", str(sample)], ROOT, home)
    stream = run(["stream.py", "--rate", "0", "--limit", "1200",
                  "--db", str(db_file)], ROOT, home)

    # Point this process's own config at the same isolated home, so the
    # assertions below read the artifacts the subprocesses just wrote.
    import importlib
    os.environ["SCAMSHIELD_HOME"] = str(home)
    importlib.reload(config)

    yield {
        "work": work, "home": home, "sample": sample, "db": db_file,
        "train_out": train.stdout, "stream_out": stream.stdout,
    }

    os.environ.pop("SCAMSHIELD_HOME", None)
    importlib.reload(config)


def test_tests_do_not_touch_the_real_artifacts(pipeline):
    """Guard the guard: everything this suite wrote must live under tmp."""
    home = pipeline["home"]
    for path in (config.MODEL_PATH, config.METRICS_PATH,
                 config.STREAM_CSV, config.WARMUP_CSV):
        assert home in path.parents or str(path).startswith(str(home)), (
            f"{path} escaped the isolated SCAMSHIELD_HOME"
        )
        assert ROOT not in path.parents, f"{path} would clobber the real repo"


# ----------------------------------------------------------------------
# Data generation
# ----------------------------------------------------------------------
def test_synthetic_data_has_paysim_shape(pipeline):
    df = pd.read_csv(pipeline["sample"])
    expected = ["step", "type", "amount", "nameOrig", "oldbalanceOrg",
                "newbalanceOrig", "nameDest", "oldbalanceDest",
                "newbalanceDest", "isFraud", "isFlaggedFraud"]
    # The PaySim columns must come first and be exactly right.  Our generator
    # appends one extra column, fraud_type, naming the pattern that produced
    # each row; ablation.py reports recall per pattern from it.  Nothing in
    # the model pipeline reads it, and the real Kaggle file does not have it.
    assert list(df.columns)[:len(expected)] == expected
    assert set(df.columns) - set(expected) <= {"fraud_type"}
    assert (df.loc[df["isFraud"] == 1, "fraud_type"] != "").all()
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
                 config.PR_CURVE_CSV, config.STREAM_CSV, config.WARMUP_CSV,
                 config.DRIFT_REFERENCE_PATH):
        assert path.exists(), f"{path} was not written by train.py"


def test_cost_analysis_prices_the_model_against_the_rule(pipeline):
    """
    Precision and recall cannot be put in a budget; cost can.  The training
    report has to price all three options, and ScamShield has to come out
    cheaper than the rule a bank runs today - otherwise there is no business
    case, whatever the PR-AUC says.
    """
    cost = json.loads(config.METRICS_PATH.read_text())["cost_analysis"]
    total = cost["total_cost"]
    assert total["do_nothing"] > 0
    assert total["scamshield"] < total["rule_baseline"] < total["do_nothing"]
    # The sweep cannot do worse than the thresholds we ship with.
    assert total["scamshield_cost_optimal"] <= total["scamshield"] + 1e-6
    assert cost["saving_vs_rule"] > 0
    assert "COST OF OPERATING" in pipeline["train_out"]


def test_review_queue_stays_inside_analyst_capacity(pipeline):
    """
    A queue nobody can work is the same as no queue at all.

    Some fraud genuinely carries no signal, so chasing the recall target
    drives the threshold to the floor.  The capacity cap is what stops that,
    and this test is what stops the cap being quietly removed.
    """
    m = json.loads(config.METRICS_PATH.read_text())
    flagged_rate = m["review_tier"]["flagged_rate"]
    # A little slack: ties in the score distribution mean the cap cannot
    # always be hit exactly.
    assert flagged_rate <= config.MAX_ALERT_RATE * 2.5, (
        f"{100 * flagged_rate:.1f}% of traffic queued for review - "
        "no team could work that"
    )
    assert m["capacity"]["threshold_set_by"] in ("analyst capacity", "recall target")


def test_ablation_groups_cover_every_feature():
    """
    The ablation study is only honest if its groups describe the real feature
    set.  If someone adds a feature and forgets to put it in a group, the
    study would silently stop measuring it.
    """
    import ablation

    grouped = [f for group in ablation.GROUPS.values() for f in group]
    assert len(grouped) == len(set(grouped)), "a feature is in two groups"
    assert set(ablation.BASELINE_FEATURES) | set(grouped) == set(F.FEATURE_NAMES), (
        "ablation.py does not account for every feature in features.py"
    )
    assert not set(ablation.BASELINE_FEATURES) & set(grouped)


def test_drift_reference_describes_the_training_window(pipeline):
    import monitor

    reference = monitor.load_reference(config.DRIFT_REFERENCE_PATH)
    assert reference is not None
    assert reference["score"]["n"] > 0
    assert len(reference["score"]["pct"]) >= 2
    assert abs(sum(reference["score"]["pct"]) - 1.0) < 1e-6
    # The reference must cover features that actually exist in the live code.
    for name in reference["features"]:
        assert name in F.FEATURE_NAMES


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


def test_stream_records_exposure_rings_and_mode(pipeline):
    """The columns the dashboard prioritises and explains work from."""
    import db as dbmod

    conn = dbmod.connect(pipeline["db"])
    bad = conn.execute("""
        SELECT COUNT(*) FROM transactions
        WHERE expected_loss < 0
           OR ring_size < 1
           OR decision_mode NOT IN ('MODEL', 'DEGRADED')
    """).fetchone()[0]
    assert bad == 0

    # expected_loss is score x amount, which is what orders the queue.
    row = conn.execute(
        "SELECT score, amount, expected_loss FROM transactions "
        "WHERE decision != 'ALLOW' ORDER BY expected_loss DESC LIMIT 1"
    ).fetchone()
    assert row["expected_loss"] == pytest.approx(row["score"] * row["amount"], rel=1e-3)

    # A clean run must be entirely model-scored.
    assert conn.execute(
        "SELECT COUNT(*) FROM transactions WHERE decision_mode = 'DEGRADED'"
    ).fetchone()[0] == 0
    conn.close()


def test_stream_keeps_deciding_when_the_model_fails(pipeline, tmp_path):
    """
    The availability guarantee, end to end.

    We make the model raise for a stretch of transactions and assert that
    (a) every single transaction still gets a decision, (b) those decisions
    are labelled DEGRADED rather than passed off as model output, (c) nothing
    is auto-blocked on the strength of the fallback rule, and (d) the engine
    goes back to using the model afterwards.
    """
    import db as dbmod

    # The run has to outlast the breaker's cooldown, or it would finish while
    # the breaker is still open and we could not prove it recovers.
    limit = 1200
    rate = 300
    assert limit / rate > config.BREAKER_COOLDOWN_SECONDS * 1.5

    db_file = tmp_path / "degraded.db"
    proc = run(["stream.py", "--rate", str(rate), "--limit", str(limit),
                "--inject-failure", "150:12", "--db", str(db_file)],
               ROOT, pipeline["home"])
    assert proc.returncode == 0, proc.stdout

    conn = dbmod.connect(db_file)
    total = conn.execute("SELECT COUNT(*) FROM transactions").fetchone()[0]
    assert total == limit, "transactions went missing while the model was down"

    modes = dict(conn.execute(
        "SELECT decision_mode, COUNT(*) FROM transactions GROUP BY 1"))
    assert modes.get("DEGRADED", 0) > 0, "the breaker never tripped"
    assert modes.get("MODEL", 0) > 0, "the engine never recovered"

    # Degraded decisions may ask a human, never block on their own.
    assert conn.execute(
        "SELECT COUNT(*) FROM transactions "
        "WHERE decision_mode = 'DEGRADED' AND decision = 'BLOCK'"
    ).fetchone()[0] == 0

    # Every row has a decision, no matter which path produced it.
    assert conn.execute(
        "SELECT COUNT(*) FROM transactions WHERE decision IS NULL OR decision = ''"
    ).fetchone()[0] == 0

    # The engine recovered: model-scored rows continue after the degraded run.
    last_degraded = conn.execute(
        "SELECT MAX(id) FROM transactions WHERE decision_mode = 'DEGRADED'"
    ).fetchone()[0]
    after = conn.execute(
        "SELECT COUNT(*) FROM transactions WHERE id > ? AND decision_mode = 'MODEL'",
        (last_degraded,),
    ).fetchone()[0]
    assert after > 0, "the breaker stayed open for the rest of the run"

    meta = dbmod.get_meta(conn)
    health = json.loads(meta["health"])
    assert health["breaker_trips"] >= 1
    assert health["scored_degraded"] > 0
    conn.close()

    assert "injecting" in proc.stdout
    assert "breaker" in proc.stdout.lower()


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
