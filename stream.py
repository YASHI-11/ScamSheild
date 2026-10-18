"""
ScamShield - the real-time streaming engine.

Replays the held-out (later-in-time) transactions as if they were arriving
live, scores each one, decides ALLOW / REVIEW / BLOCK, and writes the result
to SQLite so the Streamlit dashboard can show it within a couple of seconds.

    python stream.py                 # 50 txn/sec, fresh database
    python stream.py --rate 200      # faster replay for a short demo
    python stream.py --rate 0        # as fast as the machine allows
    python stream.py --limit 2000    # stop after 2000 transactions
    python stream.py --keep          # append to the existing database

Warm-up
-------
The model was trained on rows that already had account history behind them,
so the engine first replays the TRAINING rows through AccountState without
scoring them.  That takes a couple of seconds and makes the live features
identical to the ones the model learned on.  Skip it with --no-warmup if you
only want a quick smoke test.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import config
import db
import features as F
from scorer import ALLOW, BLOCK, REVIEW, Scorer

INSERT_SQL = """
INSERT INTO transactions
    (seq, step, type, amount, name_orig, name_dest, score, decision,
     reasons, summary, latency_ms, is_fraud, created_at,
     decision_mode, expected_loss, ring_size)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

_stop = False


def _handle_sigint(signum, frame):  # noqa: ARG001
    global _stop
    _stop = True
    print("\n[stream] stopping after the current transaction ...")


def main() -> int:
    ap = argparse.ArgumentParser(description="Replay transactions through ScamShield")
    ap.add_argument("--rate", type=float, default=config.STREAM_RATE,
                    help="transactions per second (0 = unthrottled)")
    ap.add_argument("--limit", type=int, default=0, help="stop after N transactions (0 = all)")
    ap.add_argument("--csv", type=str, default=str(config.STREAM_CSV))
    ap.add_argument("--warmup-csv", type=str, default=str(config.WARMUP_CSV))
    ap.add_argument("--no-warmup", action="store_true",
                    help="skip replaying the training rows into AccountState")
    ap.add_argument("--keep", action="store_true", help="append to the existing database")
    ap.add_argument("--db", type=str, default=str(config.DB_PATH))
    ap.add_argument("--loop", action="store_true", help="restart from the top when the feed ends")
    ap.add_argument("--inject-failure", type=str, default="",
                    help="demo the circuit breaker: START:COUNT simulates a "
                         "model outage lasting COUNT transactions, beginning "
                         "at transaction START, e.g. --inject-failure 500:20")
    args = ap.parse_args()

    # Parse the failure-injection spec up front so a typo fails fast.
    fail_at, fail_count = 0, 0
    if args.inject_failure:
        try:
            fail_at_s, fail_count_s = args.inject_failure.split(":")
            fail_at, fail_count = int(fail_at_s), int(fail_count_s)
        except ValueError:
            print("[stream] ERROR: --inject-failure wants START:COUNT, "
                  "e.g. 500:40", file=sys.stderr)
            return 1

    signal.signal(signal.SIGINT, _handle_sigint)

    feed_path = Path(args.csv)
    if not feed_path.exists():
        print(f"[stream] ERROR: {feed_path} not found - run  python train.py  first.",
              file=sys.stderr)
        return 1

    # ---- load the feed ------------------------------------------------
    feed = F.sort_time_order(pd.read_csv(feed_path))
    if args.limit:
        feed = feed.head(args.limit)
    print(f"[stream] feed: {len(feed):,} transactions from {feed_path.name}")

    # ---- warm the account state up ------------------------------------
    state = F.AccountState()
    warmup_path = Path(args.warmup_csv)
    if not args.no_warmup and warmup_path.exists():
        t0 = time.perf_counter()
        warm = F.sort_time_order(pd.read_csv(warmup_path))
        state = F.warm_up(warm, state)
        print(f"[stream] warmed AccountState on {len(warm):,} earlier transactions "
              f"in {time.perf_counter() - t0:.1f}s")
    else:
        print("[stream] cold start - no account history (features will be weaker)")

    scorer = Scorer(state=state)
    print(f"[stream] thresholds: REVIEW >= {scorer.review_threshold:.4f}, "
          f"BLOCK >= {scorer.block_threshold:.4f}")

    # ---- database ------------------------------------------------------
    if not args.keep:
        db.reset(args.db)
    conn = db.connect(args.db)
    db.set_meta(conn, "machine", platform.machine())
    db.set_meta(conn, "platform", platform.platform())
    db.set_meta(conn, "python", platform.python_version())
    db.set_meta(conn, "processor", platform.processor() or platform.machine())
    db.set_meta(conn, "cpu_count", str(os.cpu_count()))
    db.set_meta(conn, "review_threshold", f"{scorer.review_threshold:.6f}")
    db.set_meta(conn, "block_threshold", f"{scorer.block_threshold:.6f}")
    db.set_meta(conn, "feed_rows", str(len(feed)))
    db.set_meta(conn, "target_rate", str(args.rate))
    db.set_meta(conn, "latency_budget_ms", str(config.LATENCY_BUDGET_MS))
    # Stamp which model produced these rows.  The dashboard refuses to compute
    # drift unless this matches the model that wrote the drift reference.
    db.set_meta(conn, "n_features", str(len(F.FEATURE_NAMES)))
    db.set_meta(conn, "model_trained_at", str(scorer.metadata.get("trained_at", "")))
    db.set_meta(conn, "started_at", datetime.now(timezone.utc).isoformat(timespec="seconds"))
    db.set_meta(conn, "status", "running")
    conn.commit()

    # Where the stream appends to an existing run, keep the sequence going.
    seq = int(conn.execute("SELECT COALESCE(MAX(seq), 0) FROM transactions").fetchone()[0])

    interval = (1.0 / args.rate) if args.rate and args.rate > 0 else 0.0
    counts = {ALLOW: 0, REVIEW: 0, BLOCK: 0}
    latencies: list[float] = []
    money_blocked = 0.0
    money_at_risk = 0.0            # expected loss sitting in the review queue
    slo_breaches = 0               # transactions slower than the latency budget
    budget_ms = config.LATENCY_BUDGET_MS
    largest_ring = 1
    pending = 0
    started = time.perf_counter()
    next_due = started
    last_report = started
    last_commit = started

    print(f"[stream] running on {platform.machine()} at "
          f"{'max speed' if interval == 0 else f'{args.rate:g} txn/s'} - Ctrl+C to stop\n")

    rows = list(F.iter_txn_dicts(feed))
    truth = feed["isFraud"].to_numpy()
    n = len(rows)
    i = 0
    processed = 0

    try:
        while not _stop:
            if i >= n:
                if args.loop:
                    i = 0
                else:
                    break

            txn = rows[i]
            is_fraud = int(truth[i])
            i += 1

            # Demo hook: trip the circuit breaker on cue so the degraded-mode
            # behaviour can be shown live rather than just described.
            if fail_count and processed == fail_at:
                scorer.inject_failures = fail_count
                print(f"[stream] !! injecting {fail_count} scoring failures "
                      f"at transaction {processed} - watch the breaker trip")

            # ---- the measured real-time path --------------------------
            t0 = time.perf_counter()
            result = scorer.process(txn)
            latency_ms = (time.perf_counter() - t0) * 1000.0
            # -----------------------------------------------------------

            seq += 1
            processed += 1
            decision = result["decision"]
            counts[decision] += 1
            latencies.append(latency_ms)
            if latency_ms > budget_ms:
                slo_breaches += 1
            if decision == BLOCK and is_fraud:
                money_blocked += float(txn["amount"])
            if decision != ALLOW:
                money_at_risk += result["expected_loss"]
            if result["ring_size"] > largest_ring:
                largest_ring = result["ring_size"]

            conn.execute(INSERT_SQL, (
                seq,
                int(txn["step"]),
                str(txn["type"]),
                float(txn["amount"]),
                str(txn["nameOrig"]),
                str(txn["nameDest"]),
                float(result["score"]),
                decision,
                scorer.reasons_json(result["reasons"]),
                result["summary"],
                round(latency_ms, 4),
                is_fraud,
                datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
                result["mode"],
                round(result["expected_loss"], 2),
                result["ring_size"],
            ))
            pending += 1

            now = time.perf_counter()
            # Commit often enough that the dashboard feels live, rarely
            # enough that we are not fsync-bound.
            if pending >= 25 or (now - last_commit) >= 0.4:
                conn.commit()
                pending = 0
                last_commit = now

            if now - last_report >= 2.0:
                elapsed = now - started
                arr = np.asarray(latencies[-5000:])
                health = scorer.health()
                breaker = health["breaker_state"]
                flag = "" if breaker == "closed" else f" | BREAKER {breaker.upper()}"
                print(f"[stream] {processed:>7,} txns | {processed / elapsed:7.1f} txn/s | "
                      f"p50 {np.percentile(arr, 50):.2f}ms p95 {np.percentile(arr, 95):.2f}ms | "
                      f"ALLOW {counts[ALLOW]:,} REVIEW {counts[REVIEW]:,} "
                      f"BLOCK {counts[BLOCK]:,}{flag}")
                # Publish health so the dashboard can show it without having
                # to infer anything from the transaction rows.
                db.set_meta(conn, "health", json.dumps(health))
                db.set_meta(conn, "slo_breaches", str(slo_breaches))
                db.set_meta(conn, "largest_ring", str(largest_ring))
                db.set_meta(conn, "money_at_risk", f"{money_at_risk:.2f}")
                last_report = now

            # ---- rate limiting (deadline based, so it does not drift) --
            if interval:
                next_due += interval
                sleep_for = next_due - time.perf_counter()
                if sleep_for > 0:
                    time.sleep(sleep_for)
                elif sleep_for < -1.0:
                    # We have fallen more than a second behind; stop trying
                    # to catch up or the reported rate becomes a lie.
                    next_due = time.perf_counter()

    finally:
        conn.commit()
        elapsed = max(time.perf_counter() - started, 1e-9)
        arr = np.asarray(latencies) if latencies else np.zeros(1)
        summary = {
            "processed": processed,
            "elapsed_seconds": round(elapsed, 2),
            "throughput_txn_per_sec": round(processed / elapsed, 1),
            "latency_ms": {
                "p50": round(float(np.percentile(arr, 50)), 4),
                "p95": round(float(np.percentile(arr, 95)), 4),
                "p99": round(float(np.percentile(arr, 99)), 4),
                "max": round(float(arr.max()), 4),
            },
            "decisions": counts,
            "money_blocked": round(money_blocked, 2),
            "money_at_risk_queued": round(money_at_risk, 2),
            "slo": {
                "budget_ms": budget_ms,
                "breaches": slo_breaches,
                "breach_rate": round(slo_breaches / max(processed, 1), 6),
            },
            "resilience": scorer.health(),
            "largest_mule_ring": largest_ring,
            "machine": platform.machine(),
        }
        db.set_meta(conn, "status", "finished")
        db.set_meta(conn, "summary", json.dumps(summary))
        db.set_meta(conn, "health", json.dumps(scorer.health()))
        db.set_meta(conn, "slo_breaches", str(slo_breaches))
        db.set_meta(conn, "largest_ring", str(largest_ring))
        db.set_meta(conn, "money_at_risk", f"{money_at_risk:.2f}")
        conn.commit()
        conn.close()

        print("\n" + "=" * 68)
        print(f"  processed        : {processed:,} transactions in {elapsed:.1f}s")
        print(f"  throughput       : {summary['throughput_txn_per_sec']:,} txn/s "
              f"(target {'max' if interval == 0 else f'{args.rate:g}'})")
        print(f"  scoring latency  : p50 {summary['latency_ms']['p50']:.3f} ms  "
              f"p95 {summary['latency_ms']['p95']:.3f} ms  "
              f"p99 {summary['latency_ms']['p99']:.3f} ms")
        print(f"  decisions        : ALLOW {counts[ALLOW]:,} | "
              f"REVIEW {counts[REVIEW]:,} | BLOCK {counts[BLOCK]:,}")
        print(f"  money protected  : {money_blocked:,.0f} (fraud correctly blocked)")
        print(f"  money at risk    : {money_at_risk:,.0f} (expected loss in the queue)")
        print(f"  latency SLO      : {slo_breaches:,} of {processed:,} over "
              f"{budget_ms:g} ms ({100 * slo_breaches / max(processed, 1):.3f}%)")
        h = summary["resilience"]
        print(f"  resilience       : breaker {h['breaker_state']}, "
              f"{h['breaker_trips']} trip(s), "
              f"{h['scored_degraded']:,} decisions in degraded mode")
        print(f"  largest mule ring: {largest_ring} accounts")
        print(f"  running on       : {platform.machine()} / Python {platform.python_version()}")
        print(f"  database         : {args.db}")
        print("=" * 68)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
