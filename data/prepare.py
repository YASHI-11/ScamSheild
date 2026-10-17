"""
Turn the full Kaggle PaySim1 CSV (~6.3M rows, 470 MB) into a sample that
trains in a couple of minutes.

    python data/prepare.py /path/to/PS_20174392719_1491204439457_log.csv

Sampling rule
-------------
  * keep EVERY fraud row (there are only ~8,200 of them - we cannot spare any)
  * take a random sample of the non-fraud rows, stratified by (type, step bucket)
    so the time profile and the transaction-type mix of the original data survive
  * target size: config.SAMPLE_ROWS (300,000 by default)
  * the result is sorted by step, because every downstream stage is time-ordered
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

EXPECTED_COLUMNS = [
    "step", "type", "amount", "nameOrig", "oldbalanceOrg", "newbalanceOrig",
    "nameDest", "oldbalanceDest", "newbalanceDest", "isFraud", "isFlaggedFraud",
]

# Read with explicit dtypes: saves ~40% memory and avoids pandas guessing.
DTYPES = {
    "step": "int32",
    "type": "category",
    "amount": "float64",
    "nameOrig": "string",
    "oldbalanceOrg": "float64",
    "newbalanceOrig": "float64",
    "nameDest": "string",
    "oldbalanceDest": "float64",
    "newbalanceDest": "float64",
    "isFraud": "int8",
    "isFlaggedFraud": "int8",
}

STEP_BUCKET_HOURS = 24  # stratify time by day


def prepare(src: Path, n_rows: int, seed: int) -> pd.DataFrame:
    print(f"[prepare] reading {src} ...")
    df = pd.read_csv(src, dtype=DTYPES)
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise SystemExit(f"[prepare] ERROR: CSV is missing columns: {missing}")
    print(f"[prepare] loaded {len(df):,} rows, {int(df['isFraud'].sum()):,} fraud")

    fraud = df[df["isFraud"] == 1]
    legit = df[df["isFraud"] == 0]

    n_legit_target = max(0, n_rows - len(fraud))
    if n_legit_target >= len(legit):
        print("[prepare] sample target >= dataset size, keeping everything")
        sampled_legit = legit
    else:
        frac = n_legit_target / len(legit)
        print(f"[prepare] sampling {frac:.3%} of the {len(legit):,} legit rows, "
              f"stratified by (type, day)")
        day = (legit["step"] // STEP_BUCKET_HOURS).astype("int32")
        # Use GroupBy.sample, NOT groupby(...).apply(lambda g: g.sample(...)):
        # from pandas 3.0 onwards apply() drops the grouping columns from the
        # frame it hands back, which silently turned 'type' into NaN here.
        sampled_legit = legit.groupby(
            [legit["type"], day], observed=True
        ).sample(frac=frac, random_state=seed)

    out = pd.concat([fraud, sampled_legit], ignore_index=True)
    out = out.sort_values("step", kind="mergesort").reset_index(drop=True)

    # Shrink back to plain dtypes so the CSV round-trips cleanly.
    out["type"] = out["type"].astype(str)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Sample the PaySim CSV for ScamShield")
    ap.add_argument("csv", type=str, help="path to PS_20174392719_1491204439457_log.csv")
    ap.add_argument("--rows", type=int, default=config.SAMPLE_ROWS)
    ap.add_argument("--out", type=str, default=str(config.SAMPLE_CSV))
    ap.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    args = ap.parse_args()

    src = Path(args.csv)
    if not src.exists():
        raise SystemExit(
            f"[prepare] ERROR: {src} not found.\n"
            "            Download 'PaySim1' from Kaggle, or run\n"
            "            python data/make_synthetic.py\n"
            "            to generate a PaySim-shaped file instead."
        )

    out_df = prepare(src, args.rows, args.seed)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)

    n_fraud = int(out_df["isFraud"].sum())
    print(f"[prepare] wrote {len(out_df):,} rows -> {out_path}")
    print(f"[prepare] fraud rows kept: {n_fraud:,} ({100 * n_fraud / len(out_df):.3f}%)")
    print(f"[prepare] steps: {out_df['step'].min()} .. {out_df['step'].max()}")
    print(f"[prepare] type mix:\n{out_df['type'].value_counts(normalize=True).round(4)}")


if __name__ == "__main__":
    main()
