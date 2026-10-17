"""
ScamShield - throughput and latency benchmark.

Run the identical code on your laptop and on IBM LinuxONE, then compare
artifacts/benchmark.json.  platform.machine() is recorded in the file, so the
s390x result is self-evidencing.

    python benchmark.py                 # 50,000 transactions
    python benchmark.py --rows 200000
    python benchmark.py --out artifacts/benchmark_linuxone.json

Three numbers are reported
--------------------------
1. end_to_end   - features + score + decision, one transaction at a time.
                  This is the real figure: it is exactly what stream.py does
                  per payment, so the p99 is a genuine tail latency.
2. score_only   - just the model call, one row at a time.  Isolates the
                  inference cost from the feature engineering cost.
3. batch        - the whole matrix scored in one call.  Shows the ceiling when
                  transactions can be micro-batched instead of handled singly.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import config
import features as F
from scorer import Scorer


def percentiles(samples_ms: np.ndarray) -> dict[str, float]:
    return {
        "p50": round(float(np.percentile(samples_ms, 50)), 4),
        "p95": round(float(np.percentile(samples_ms, 95)), 4),
        "p99": round(float(np.percentile(samples_ms, 99)), 4),
        "max": round(float(samples_ms.max()), 4),
        "mean": round(float(samples_ms.mean()), 4),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Benchmark ScamShield scoring")
    ap.add_argument("--rows", type=int, default=config.BENCHMARK_ROWS)
    ap.add_argument("--csv", type=str, default="", help="defaults to the stream feed, "
                                                        "falling back to the sample")
    ap.add_argument("--out", type=str, default=str(config.BENCHMARK_PATH))
    ap.add_argument("--warmup-rows", type=int, default=2000,
                    help="rows scored before timing starts (lets caches settle)")
    args = ap.parse_args()

    # ---- pick a data file ---------------------------------------------
    if args.csv:
        csv_path = Path(args.csv)
    elif config.STREAM_CSV.exists():
        csv_path = config.STREAM_CSV
    else:
        csv_path = config.SAMPLE_CSV
    if not csv_path.exists():
        print(f"[benchmark] ERROR: {csv_path} not found - run train.py first.", file=sys.stderr)
        return 1

    df = F.sort_time_order(pd.read_csv(csv_path))
    # Repeat the feed if the caller asked for more rows than we have, so the
    # laptop and LinuxONE runs always score the same N.
    if len(df) < args.rows:
        reps = (args.rows // len(df)) + 1
        df = pd.concat([df] * reps, ignore_index=True)
    df = df.head(args.rows).reset_index(drop=True)
    n = len(df)

    print(f"[benchmark] {n:,} transactions from {csv_path.name}")
    print(f"[benchmark] machine={platform.machine()}  python={platform.python_version()}  "
          f"cpus={os.cpu_count()}")

    scorer = Scorer()
    rows = list(F.iter_txn_dicts(df))

    # ---- warm-up (JIT-less, but caches / page faults / BLAS threads) ----
    warm = min(args.warmup_rows, n)
    warm_scorer = Scorer(state=F.AccountState())
    for k in range(warm):
        warm_scorer.process(rows[k])

    # ---- 1. end-to-end, one transaction at a time -----------------------
    scorer.state = F.AccountState()
    lat = np.empty(n, dtype=np.float64)
    t_start = time.perf_counter()
    for k in range(n):
        t0 = time.perf_counter()
        scorer.process(rows[k])
        lat[k] = (time.perf_counter() - t0) * 1000.0
    e2e_elapsed = time.perf_counter() - t_start
    e2e = {
        "throughput_txn_per_sec": round(n / e2e_elapsed, 1),
        "elapsed_seconds": round(e2e_elapsed, 3),
        "latency_ms": percentiles(lat),
    }

    # ---- 2. score only --------------------------------------------------
    X, _ = F.build_feature_matrix(df)
    lat2 = np.empty(n, dtype=np.float64)
    t_start = time.perf_counter()
    for k in range(n):
        t0 = time.perf_counter()
        scorer.score_fast(X[k])
        lat2[k] = (time.perf_counter() - t0) * 1000.0
    so_elapsed = time.perf_counter() - t_start
    score_only = {
        "throughput_txn_per_sec": round(n / so_elapsed, 1),
        "elapsed_seconds": round(so_elapsed, 3),
        "latency_ms": percentiles(lat2),
    }

    # ---- 3. batch -------------------------------------------------------
    t0 = time.perf_counter()
    scorer.booster.inplace_predict(X)
    batch_elapsed = time.perf_counter() - t0
    batch = {
        "throughput_txn_per_sec": round(n / batch_elapsed, 1),
        "elapsed_seconds": round(batch_elapsed, 4),
        "per_txn_ms": round(batch_elapsed / n * 1000.0, 6),
    }

    result = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "rows": n,
        "data_file": csv_path.name,
        "platform": {
            "machine": platform.machine(),          # 's390x' on LinuxONE
            "processor": platform.processor() or platform.machine(),
            "system": platform.system(),
            "release": platform.release(),
            "python": platform.python_version(),
            "cpu_count": os.cpu_count(),
        },
        "model": {
            "n_estimators": config.XGB_PARAMS["n_estimators"],
            "max_depth": config.XGB_PARAMS["max_depth"],
            "n_features": len(F.FEATURE_NAMES),
        },
        "end_to_end": e2e,
        "score_only": score_only,
        "batch": batch,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))

    print("\n" + "=" * 70)
    print(f"  ScamShield benchmark - {platform.machine()} - {n:,} transactions")
    print("=" * 70)
    for name, block in (("end-to-end (features+score)", e2e),
                        ("score only", score_only)):
        lm = block["latency_ms"]
        print(f"  {name:<28} {block['throughput_txn_per_sec']:>10,.0f} txn/s   "
              f"p50 {lm['p50']:.3f}  p95 {lm['p95']:.3f}  p99 {lm['p99']:.3f} ms")
    print(f"  {'batch (whole matrix)':<28} {batch['throughput_txn_per_sec']:>10,.0f} txn/s   "
          f"{batch['per_txn_ms']:.5f} ms/txn")
    print("=" * 70)
    print(f"  saved -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
