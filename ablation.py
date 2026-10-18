"""
ScamShield - does each feature group actually earn its place?

    python ablation.py
    python ablation.py --csv data/paysim_sample.csv

Adding features is easy; showing that they help is the part that counts.  This
script retrains the model with one group of features removed at a time, on
exactly the same time-based split, and reports what each group is worth.

It also breaks recall down BY SCAM TYPOLOGY, because an average hides the
thing a fraud team cares about.  A model can hold its overall recall steady
while quietly going blind to mule rings, and only a per-pattern table shows
it.  (The typology labels come from data/make_synthetic.py; the real Kaggle
file has no such column, in which case that section is skipped.)

Nothing here is on the serving path - it is evidence, produced offline.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from xgboost import XGBClassifier

import config
import features as F

# The original feature set, before the mule-ring, velocity and victim work.
BASELINE_FEATURES = [
    "amount", "log_amount", "amount_to_balance_ratio", "hour_of_day",
    "type_CASH_IN", "type_CASH_OUT", "type_DEBIT", "type_PAYMENT",
    "type_TRANSFER", "balance_error_orig", "balance_error_dest",
    "orig_drained_to_zero", "dest_was_empty", "dest_fanin_senders",
    "dest_new_sender", "dest_rapid_passthrough", "orig_rapid_passthrough",
    "orig_prior_txn_count", "dest_prior_inflow_count",
]

GROUPS: dict[str, list[str]] = {
    "mule rings": ["dest_ring_size", "orig_ring_size", "dest_ring_volume"],
    "velocity/structuring": ["orig_txns_in_window", "orig_amount_in_window",
                             "amount_just_below_round"],
    "victim side": ["orig_amount_vs_mean", "orig_amount_vs_max",
                    "orig_first_time_night"],
}


def _recall_at(scores: np.ndarray, y: np.ndarray, alert_rate: float) -> tuple[float, np.ndarray]:
    """Recall when exactly `alert_rate` of traffic is queued."""
    threshold = float(np.quantile(scores, 1.0 - alert_rate))
    flagged = scores >= threshold
    n_fraud = int(np.count_nonzero(y == 1))
    caught = int(np.count_nonzero(flagged & (y == 1)))
    return (caught / n_fraud if n_fraud else 0.0), flagged


def _evaluate(scores: np.ndarray, y: np.ndarray) -> dict:
    """
    Score quality at the operating point the system actually runs at.

    PR-AUC on its own is a threshold-free abstraction.  What the bank sees is
    "how much fraud do we catch inside the alert budget we can staff", so the
    threshold is pinned to the capacity cap, exactly as train.py does.

    A second, looser budget is reported alongside it.  The cap is a HARD
    constraint: at 2% the number of alerts is fixed, so a better model mostly
    changes WHICH fraud fills the queue rather than how much fits in it.
    Measuring at 5% as well separates "the model got better" from "the budget
    was the binding constraint".
    """
    recall_cap, flagged = _recall_at(scores, y, config.MAX_ALERT_RATE)
    recall_wide, _ = _recall_at(scores, y, min(config.MAX_ALERT_RATE * 2.5, 0.5))
    return {
        "pr_auc": round(float(average_precision_score(y, scores)), 4),
        "recall_at_capacity": round(recall_cap, 4),
        "recall_at_wide_budget": round(recall_wide, 4),
        "precision_at_capacity": (
            round(int(np.count_nonzero(flagged & (y == 1))) / int(flagged.sum()), 4)
            if flagged.sum() else 0.0
        ),
        "flagged": int(flagged.sum()),
        "_flagged_mask": flagged,
    }


def _recall_by_typology(flagged: np.ndarray, y: np.ndarray, scores: np.ndarray,
                        typology: pd.Series) -> dict[str, dict]:
    """
    Per-pattern detection, two ways.

    `recall` is what the shipped system catches, and it is coupled across
    patterns: they compete for one fixed alert budget, so a gain on one shows
    up as a loss on another even when the model improved.

    `pr_auc_isolated` removes that coupling.  For each pattern we score it
    against the legitimate traffic ALONE, ignoring the other fraud types, and
    so measure "can the model see this pattern at all" independently of what
    else is in the queue.  That is the number that tells you whether a feature
    group did its job.
    """
    out: dict[str, dict] = {}
    fraud = y == 1
    types = typology.to_numpy()
    legit = ~fraud
    for name in sorted(set(types[fraud])):
        rows = fraud & (types == name)
        n = int(np.count_nonzero(rows))
        if n == 0:
            continue
        subset = rows | legit
        try:
            isolated = float(average_precision_score(rows[subset], scores[subset]))
        except ValueError:
            isolated = float("nan")
        hit = int(np.count_nonzero(rows & flagged))
        out[name] = {
            "fraud_rows": n,
            "caught": hit,
            "recall": round(hit / n, 4),
            "pr_auc_isolated": round(isolated, 4),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Feature-group ablation study")
    ap.add_argument("--csv", type=str, default=str(config.SAMPLE_CSV))
    ap.add_argument("--train-fraction", type=float, default=config.TRAIN_FRACTION)
    ap.add_argument("--out", type=str,
                    default=str(config.ARTIFACT_DIR / "ablation.json"))
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"[ablation] ERROR: {csv_path} not found - run "
              f"python data/make_synthetic.py first", file=sys.stderr)
        return 1

    t_start = time.perf_counter()
    df = F.sort_time_order(pd.read_csv(csv_path))
    X, _state = F.build_feature_matrix(df)
    y = df["isFraud"].to_numpy().astype(int)

    # Same time-based split as train.py, so the numbers are comparable.
    steps = df["step"].to_numpy()
    split_step = int(np.quantile(steps, args.train_fraction))
    train_mask = steps <= split_step
    test_mask = ~train_mask
    if test_mask.sum() == 0 or y[test_mask].sum() == 0:
        cut = int(len(df) * args.train_fraction)
        train_mask = np.zeros(len(df), bool)
        train_mask[:cut] = True
        test_mask = ~train_mask

    y_train, y_test = y[train_mask], y[test_mask]
    n_pos = max(int(y_train.sum()), 1)
    spw = max(int(len(y_train) - y_train.sum()) / n_pos, 1.0)

    typology = (df.loc[test_mask, "fraud_type"].reset_index(drop=True)
                if "fraud_type" in df.columns else None)

    print(f"[ablation] {len(df):,} rows, {int(y.sum()):,} fraud, "
          f"split at step {split_step}")
    print(f"[ablation] test window: {int(test_mask.sum()):,} rows, "
          f"{int(y_test.sum()):,} fraud")
    print(f"[ablation] operating point: top {100 * config.MAX_ALERT_RATE:g}% "
          f"of scores (the analyst capacity cap)\n")

    # Which configurations to try: everything, then everything-minus-one-group,
    # then the original baseline on its own.
    all_new = [f for g in GROUPS.values() for f in g]
    configs: dict[str, list[str]] = {"all 28 features": F.FEATURE_NAMES}
    for name, group in GROUPS.items():
        configs[f"without {name}"] = [f for f in F.FEATURE_NAMES if f not in group]
    configs["baseline 19 only"] = [f for f in F.FEATURE_NAMES if f not in all_new]

    results: dict[str, dict] = {}
    for label, names in configs.items():
        cols = [F.FEATURE_NAMES.index(n) for n in names]
        model = XGBClassifier(**config.XGB_PARAMS, scale_pos_weight=spw)
        model.fit(X[train_mask][:, cols], y_train, verbose=False)
        scores = model.predict_proba(X[test_mask][:, cols])[:, 1]

        res = _evaluate(scores, y_test)
        flagged = res.pop("_flagged_mask")
        res["n_features"] = len(names)
        if typology is not None:
            res["by_typology"] = _recall_by_typology(
                flagged, y_test, scores, typology)
        results[label] = res
        print(f"[ablation] {label:<26} PR-AUC {res['pr_auc']:.4f}  "
              f"recall@cap {res['recall_at_capacity']:.3f}")

    # ---- report ------------------------------------------------------
    full = results["all 28 features"]
    base = results["baseline 19 only"]

    print("\n" + "=" * 78)
    print("  WHAT EACH FEATURE GROUP IS WORTH")
    print("=" * 78)
    wide_pct = 100 * min(config.MAX_ALERT_RATE * 2.5, 0.5)
    print(f"  {'configuration':<26} {'feats':>5} {'PR-AUC':>8} "
          f"{'r@2%':>7} {f'r@{wide_pct:g}%':>7} {'vs full':>9}")
    print("-" * 78)
    for label, res in results.items():
        delta = res["recall_at_capacity"] - full["recall_at_capacity"]
        marker = "" if label == "all 28 features" else f"{delta:+.3f}"
        print(f"  {label:<26} {res['n_features']:>5} {res['pr_auc']:>8.4f} "
              f"{res['recall_at_capacity']:>7.3f} "
              f"{res['recall_at_wide_budget']:>7.3f} {marker:>9}")
    print("-" * 78)
    print(f"  Overall, the new features are worth "
          f"{full['recall_at_capacity'] - base['recall_at_capacity']:+.3f} recall "
          f"at the 2% cap")
    print(f"  and {full['pr_auc'] - base['pr_auc']:+.4f} PR-AUC - which is "
          f"close to nothing, and that is the")
    print("  honest headline for the aggregate. The reason is below: at a FIXED")
    print("  alert budget the number of alerts cannot grow, so a better model")
    print("  changes which fraud fills the queue rather than how much fits in it.")

    if typology is not None:
        print("\n" + "=" * 78)
        print("  BY SCAM TYPOLOGY  (the table a fraud team actually reads)")
        print("=" * 78)
        patterns = sorted(full["by_typology"])
        print(f"  {'pattern':<20} {'cases':>5} | {'recall: base':>12} "
              f"{'all':>6} | {'isolated PR-AUC: base':>22} {'all':>6}")
        print("-" * 78)
        for name in patterns:
            f_row = full["by_typology"][name]
            b_row = base["by_typology"].get(
                name, {"recall": 0.0, "pr_auc_isolated": 0.0})
            print(f"  {name:<20} {f_row['fraud_rows']:>5} | "
                  f"{b_row['recall']:>12.3f} {f_row['recall']:>6.3f} | "
                  f"{b_row['pr_auc_isolated']:>22.3f} "
                  f"{f_row['pr_auc_isolated']:>6.3f}")
        print("-" * 78)
        print("  'recall' competes for one shared budget, so it can fall on one")
        print("  pattern purely because another improved. 'isolated PR-AUC' scores")
        print("  each pattern against legitimate traffic only, which is the number")
        print("  that says whether a feature group can see its target at all.")
        print("  'no_signal' is fraud generated with no detectable trace on purpose:")
        print("  ~0 recall there is the honest ceiling, not a bug.")
        print("  Patterns with few cases (under ~30) move a lot on one or two")
        print("  transactions - read those rows as direction, not magnitude.")
    print("=" * 78 + "\n")

    payload = {
        "dataset": csv_path.name,
        "rows": int(len(df)),
        "test_rows": int(test_mask.sum()),
        "test_fraud": int(y_test.sum()),
        "split_step": split_step,
        "operating_point": f"top {100 * config.MAX_ALERT_RATE:g}% of scores",
        "groups": GROUPS,
        "results": results,
        "platform": {"machine": platform.machine(),
                     "python": platform.python_version()},
        "total_seconds": round(time.perf_counter() - t_start, 2),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"[ablation] saved -> {out}")
    print(f"[ablation] done in {payload['total_seconds']}s on {platform.machine()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
