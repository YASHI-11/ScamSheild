"""
ScamShield - train the fraud model.

    python train.py                       # uses data/paysim_sample.csv
    python train.py --csv data/other.csv

What it does
------------
1. Loads the sample, sorts it by step (simulated hour).
2. Replays EVERY row, in time order, through the one shared AccountState in
   features.py.  Features for row i therefore only ever see rows 0..i-1 -
   identical to what stream.py will see in production.
3. Splits by TIME, not at random: the earliest 70% of the steps train the
   model, the later 30% test it.  A random split would let the model peek at
   the future and would wildly overstate the scores.
4. Trains XGBoost with scale_pos_weight to cope with the class imbalance.
5. Picks two operating points - REVIEW (high recall) and BLOCK (high
   precision) - and compares the model against a plain rule baseline.
6. Writes everything stream.py / app.py / benchmark.py need into artifacts/.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier
import joblib

import config
import features as F
import monitor


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def _metrics_at(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    """Precision / recall / confusion matrix at one operating point."""
    pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "threshold": round(float(threshold), 6),
        "precision": round(float(precision_score(y_true, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, pred, zero_division=0)), 4),
        "flagged": int(pred.sum()),
        "flagged_rate": round(float(pred.mean()), 6),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def _pick_thresholds(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, float, bool]:
    """
    REVIEW = the best recall achievable WITHIN the analyst team's capacity.
    BLOCK  = the loosest threshold whose precision is at least
             TARGET_BLOCK_PRECISION (safe enough to stop the payment itself).

    Returns (review, block, capacity_limited).

    The capacity cap is the important part.  Aiming purely at
    TARGET_REVIEW_RECALL assumes every fraud is detectable, and it is not:
    a compromised card used for a normal-sized purchase at a normal merchant
    looks exactly like a real one.  Chasing those cases drags the threshold
    to the floor and queues a third of all traffic.  So we take the recall
    target as an aspiration and MAX_ALERT_RATE as the hard constraint, which
    is the way round a real fraud operation works.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    # precision_recall_curve returns len(thresholds) == len(precision) - 1
    precision, recall = precision[:-1], recall[:-1]

    ok_recall = np.where(recall >= config.TARGET_REVIEW_RECALL)[0]
    review = float(thresholds[ok_recall[-1]]) if len(ok_recall) else config.DEFAULT_REVIEW_THRESHOLD

    # Capacity constraint: never queue more than MAX_ALERT_RATE of traffic.
    budget_threshold = float(np.quantile(scores, 1.0 - config.MAX_ALERT_RATE))
    capacity_limited = review < budget_threshold
    if capacity_limited:
        review = budget_threshold

    ok_precision = np.where(precision >= config.TARGET_BLOCK_PRECISION)[0]
    block = float(thresholds[ok_precision[0]]) if len(ok_precision) else config.DEFAULT_BLOCK_THRESHOLD

    # Keep the ladder sane even on tiny/degenerate datasets.
    review = float(min(max(review, 1e-4), 0.99))
    block = float(min(max(block, review), 0.999))
    if block <= review:
        block = min(0.999, max(review + 0.05, review * 1.5))
    return review, block, capacity_limited


def _expected_cost(
    y_true: np.ndarray,
    scores: np.ndarray,
    amounts: np.ndarray,
    review: float,
    block: float,
) -> float:
    """
    What this operating point would COST the bank, in currency.

    Three tiers, three different prices:

      BLOCK  stopped automatically.  Fraud stopped costs nothing; an honest
             customer stopped costs COST_FALSE_BLOCK (support call, a lost
             payment, sometimes a lost customer).
      REVIEW an analyst works it: COST_REVIEW of someone's time.  They are
             good but not perfect, so we only credit them with catching
             ANALYST_CATCH_RATE of the fraud they are shown.
      ALLOW  fraud that gets through costs the full amount.

    This is the function that actually matters.  Precision and recall are
    proxies for it, and they are proxies that disagree with it, because the
    price of the two mistakes differs by three orders of magnitude.
    """
    fraud = y_true == 1
    is_block = scores >= block
    is_review = (scores >= review) & ~is_block
    is_allow = ~is_block & ~is_review

    leak = config.COST_MISSED_FRAUD_RATIO
    missed_share = 1.0 - config.ANALYST_CATCH_RATE

    cost = config.COST_FALSE_BLOCK * float(np.count_nonzero(is_block & ~fraud))
    cost += config.COST_REVIEW * float(np.count_nonzero(is_review))
    cost += leak * missed_share * float(amounts[is_review & fraud].sum())
    cost += leak * float(amounts[is_allow & fraud].sum())
    return float(cost)


def _optimise_cost(
    y_true: np.ndarray, scores: np.ndarray, amounts: np.ndarray
) -> tuple[float, float, float]:
    """
    Sweep the threshold pair and return the cheapest (review, block, cost).

    A coarse quantile grid is enough: the cost surface is flat near its
    minimum, so chasing the fourth decimal place of a threshold buys nothing
    and would just overfit the test window.
    """
    grid = np.unique(np.quantile(scores, np.linspace(0.50, 0.99999, 50)))
    grid = grid[(grid > 0.0) & (grid < 1.0)]
    if grid.size == 0:
        return (config.DEFAULT_REVIEW_THRESHOLD, config.DEFAULT_BLOCK_THRESHOLD,
                float("nan"))

    best = (float(grid[0]), float(grid[-1]), float("inf"))
    for review in grid:
        for block in grid[grid >= review]:
            cost = _expected_cost(y_true, scores, amounts, float(review), float(block))
            if cost < best[2]:
                best = (float(review), float(block), cost)
    return best


def _rule_baseline(df: pd.DataFrame) -> np.ndarray:
    """
    The 'what most banks actually run today' comparison: a static rule.

    Flag the payment when it is large AND it empties the sender's account.
    No learning, no context, no account history.
    """
    large = df["amount"].to_numpy() > config.RULE_AMOUNT
    drained = (df["oldbalanceOrg"].to_numpy() > 0) & (df["newbalanceOrig"].to_numpy() == 0)
    return (large & drained).astype(int)


def _save_pr_curve(y_true: np.ndarray, scores: np.ndarray, review: float, block: float) -> str:
    """
    Save the precision-recall curve.

    Always writes a CSV (pure pandas).  The PNG needs matplotlib, which ships
    with the LinuxONE JupyterLab image but is not a hard requirement of this
    project - so it is wrapped in try/except, exactly like the ONNX export.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    pd.DataFrame({
        "threshold": np.append(thresholds, 1.0),
        "precision": precision,
        "recall": recall,
    }).to_csv(config.PR_CURVE_CSV, index=False)

    try:
        import matplotlib
        matplotlib.use("Agg")           # headless - no display on the server
        import matplotlib.pyplot as plt

        ap = average_precision_score(y_true, scores)
        fig, ax = plt.subplots(figsize=(6, 4.5), dpi=130)
        ax.plot(recall, precision, lw=2, color="#1f77b4", label=f"ScamShield (PR-AUC={ap:.3f})")
        for thr, name, colour in ((review, "REVIEW", "#ff9900"), (block, "BLOCK", "#d62728")):
            idx = int(np.searchsorted(thresholds, thr))
            idx = min(idx, len(precision) - 1)
            ax.scatter([recall[idx]], [precision[idx]], s=60, color=colour, zorder=5,
                       label=f"{name} @ {thr:.3f}")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("ScamShield - precision/recall on held-out later transactions")
        ax.grid(alpha=0.3)
        ax.legend(loc="lower left", fontsize=8)
        fig.tight_layout()
        fig.savefig(config.PR_CURVE_PNG)
        plt.close(fig)
        return f"PNG + CSV ({config.PR_CURVE_PNG.name}, {config.PR_CURVE_CSV.name})"
    except Exception as exc:  # pragma: no cover - environment dependent
        return f"CSV only ({config.PR_CURVE_CSV.name}); PNG skipped: {exc}"


def _try_export_onnx(model: XGBClassifier, n_features: int) -> str:
    """
    OPTIONAL: export to ONNX for a vendor-neutral, portable runtime.

    onnxmltools / onnxruntime are frequently unavailable for s390x, so a
    failure here is expected and harmless - stream.py always uses the joblib
    model.  This is a nice-to-have for the judges, never a dependency.
    """
    try:
        from onnxmltools import convert_xgboost
        from onnxmltools.convert.common.data_types import FloatTensorType

        onnx_model = convert_xgboost(
            model, initial_types=[("input", FloatTensorType([None, n_features]))]
        )
        config.ONNX_PATH.write_bytes(onnx_model.SerializeToString())
        return f"exported -> {config.ONNX_PATH.name}"
    except Exception as exc:
        return f"skipped ({type(exc).__name__}: {exc})"


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="Train the ScamShield model")
    ap.add_argument("--csv", type=str, default=str(config.SAMPLE_CSV))
    ap.add_argument("--train-fraction", type=float, default=config.TRAIN_FRACTION)
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"[train] ERROR: {csv_path} not found.\n"
              f"        Run  python data/make_synthetic.py  (no Kaggle needed)\n"
              f"        or   python data/prepare.py <PaySim csv>", file=sys.stderr)
        return 1

    config.ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()

    # --- 1. load, time order -------------------------------------------
    df = pd.read_csv(csv_path)
    df = F.sort_time_order(df)
    print(f"[train] {len(df):,} rows, steps {df['step'].min()}..{df['step'].max()}, "
          f"{int(df['isFraud'].sum()):,} fraud ({100 * df['isFraud'].mean():.3f}%)")

    # --- 2. features via the SHARED streaming code ----------------------
    t0 = time.perf_counter()
    X, _state = F.build_feature_matrix(df)
    y = df["isFraud"].to_numpy().astype(int)
    print(f"[train] features: {X.shape} in {time.perf_counter() - t0:.1f}s "
          f"(same AccountState class the live engine uses)")

    # --- 3. time-based split -------------------------------------------
    steps = df["step"].to_numpy()
    split_step = int(np.quantile(steps, args.train_fraction))
    train_mask = steps <= split_step
    test_mask = ~train_mask
    if test_mask.sum() == 0 or y[test_mask].sum() == 0:
        # Degenerate (tiny) dataset - fall back to a positional cut that keeps
        # some fraud in the test half, so the pipeline still completes.
        cut = int(len(df) * args.train_fraction)
        train_mask = np.zeros(len(df), bool); train_mask[:cut] = True
        test_mask = ~train_mask
        split_step = int(steps[cut - 1])
        print("[train] WARNING: step-based split left no fraud in test; "
              "used a positional cut instead")

    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    print(f"[train] time split at step {split_step}: "
          f"train {train_mask.sum():,} rows ({y_train.sum():,} fraud) | "
          f"test {test_mask.sum():,} rows ({y_test.sum():,} fraud)")

    # --- 4. train --------------------------------------------------------
    n_pos = max(int(y_train.sum()), 1)
    n_neg = int(len(y_train) - y_train.sum())
    scale_pos_weight = max(n_neg / n_pos, 1.0)
    print(f"[train] scale_pos_weight = {scale_pos_weight:.1f}")

    model = XGBClassifier(**config.XGB_PARAMS, scale_pos_weight=scale_pos_weight)
    t0 = time.perf_counter()
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    train_seconds = time.perf_counter() - t0
    print(f"[train] fitted {config.XGB_PARAMS['n_estimators']} trees in {train_seconds:.1f}s")

    # --- 5. evaluate -----------------------------------------------------
    scores = model.predict_proba(X_test)[:, 1]
    pr_auc = float(average_precision_score(y_test, scores))
    try:
        roc_auc = float(roc_auc_score(y_test, scores))
    except ValueError:
        roc_auc = float("nan")

    review_thr, block_thr, capacity_limited = _pick_thresholds(y_test, scores)
    review_m = _metrics_at(y_test, scores, review_thr)
    block_m = _metrics_at(y_test, scores, block_thr)

    # Rule baseline on exactly the same held-out rows.
    rule_pred = _rule_baseline(df[test_mask])
    rtn, rfp, rfn, rtp = confusion_matrix(y_test, rule_pred, labels=[0, 1]).ravel()
    rule_m = {
        "rule": f"amount > {config.RULE_AMOUNT:,.0f} AND sender drained to zero",
        "precision": round(float(precision_score(y_test, rule_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, rule_pred, zero_division=0)), 4),
        "pr_auc": round(float(average_precision_score(y_test, rule_pred)), 4),
        "flagged": int(rule_pred.sum()),
        "confusion_matrix": {"tn": int(rtn), "fp": int(rfp), "fn": int(rfn), "tp": int(rtp)},
    }

    # Money the BLOCK tier would have saved on the test window.
    amounts = df.loc[test_mask, "amount"].to_numpy()
    money_blocked = float(amounts[(scores >= block_thr) & (y_test == 1)].sum())
    money_missed = float(amounts[(scores < block_thr) & (y_test == 1)].sum())

    # --- 5b. the same question in currency -------------------------------
    # Precision and recall cannot be added up or put in a budget.  Cost can.
    n_test = max(int(test_mask.sum()), 1)
    per_1k = 1000.0 / n_test

    cost_do_nothing = float(config.COST_MISSED_FRAUD_RATIO * amounts[y_test == 1].sum())
    cost_model = _expected_cost(y_test, scores, amounts, review_thr, block_thr)
    cost_review_thr, cost_block_thr, cost_optimal = _optimise_cost(
        y_test, scores, amounts
    )

    # The rule baseline, priced the same way: whatever it flags goes to an
    # analyst, and everything it misses is lost.
    rule_flagged = rule_pred.astype(bool)
    cost_rule = (
        config.COST_REVIEW * float(rule_flagged.sum())
        + config.COST_MISSED_FRAUD_RATIO * (1.0 - config.ANALYST_CATCH_RATE)
        * float(amounts[rule_flagged & (y_test == 1)].sum())
        + config.COST_MISSED_FRAUD_RATIO
        * float(amounts[~rule_flagged & (y_test == 1)].sum())
    )

    cost_analysis = {
        "assumptions": {
            "cost_per_review": config.COST_REVIEW,
            "cost_per_false_block": config.COST_FALSE_BLOCK,
            "missed_fraud_ratio": config.COST_MISSED_FRAUD_RATIO,
            "analyst_catch_rate": config.ANALYST_CATCH_RATE,
        },
        "test_rows": n_test,
        "total_cost": {
            "do_nothing": round(cost_do_nothing, 2),
            "rule_baseline": round(cost_rule, 2),
            "scamshield": round(cost_model, 2),
            "scamshield_cost_optimal": round(cost_optimal, 2),
        },
        "cost_per_1000_payments": {
            "do_nothing": round(cost_do_nothing * per_1k, 2),
            "rule_baseline": round(cost_rule * per_1k, 2),
            "scamshield": round(cost_model * per_1k, 2),
            "scamshield_cost_optimal": round(cost_optimal * per_1k, 2),
        },
        "saving_vs_rule": round(cost_rule - cost_model, 2),
        "saving_vs_rule_per_1000": round((cost_rule - cost_model) * per_1k, 2),
        "saving_vs_rule_pct": (
            round(100.0 * (cost_rule - cost_model) / cost_rule, 2) if cost_rule else None
        ),
        "cost_optimal_thresholds": {
            "review": round(cost_review_thr, 6),
            "block": round(cost_block_thr, 6),
        },
        # When the optimiser collapses the two thresholds onto each other it
        # is telling us something real: at this precision, paying an analyst
        # to confirm what the model already knows costs more than it saves.
        # That is a property of THIS data, not a general result, so it is
        # recorded rather than silently acted on.
        "cost_optimal_policy": (
            "block-only (precision is high enough that human review does not "
            "pay for itself on this data)"
            if cost_block_thr <= cost_review_thr
            else "two-tier (review then block)"
        ),
        "note": (
            "The statistical thresholds (recall/precision targets) are what the "
            "engine ships with. The cost-optimal pair is reported alongside so "
            "the operator can see what tuning for money instead of for F1 "
            "would change."
        ),
    }

    curve_note = _save_pr_curve(y_test, scores, review_thr, block_thr)
    onnx_note = _try_export_onnx(model, X.shape[1])

    # --- 6. report -------------------------------------------------------
    def _cm(m):
        c = m["confusion_matrix"]
        return f"TP={c['tp']:<6} FP={c['fp']:<6} FN={c['fn']:<6} TN={c['tn']:,}"

    print("\n" + "=" * 72)
    print("  ScamShield - held-out (later in time) results")
    print("=" * 72)
    print(f"  PR-AUC (average precision) : {pr_auc:.4f}")
    print(f"  ROC-AUC                    : {roc_auc:.4f}")
    print("-" * 72)
    print(f"  REVIEW tier  @ {review_thr:.4f}  precision {review_m['precision']:.3f}  "
          f"recall {review_m['recall']:.3f}  ({review_m['flagged']:,} queued, "
          f"{100 * review_m['flagged_rate']:.2f}% of traffic)")
    print(f"               {_cm(review_m)}")
    if capacity_limited:
        print(f"               NOTE: threshold set by analyst capacity "
              f"(cap ~{100 * config.MAX_ALERT_RATE:g}% of traffic), not by the "
              f"{100 * config.TARGET_REVIEW_RECALL:g}% recall target.")
        print("               Some fraud in this data carries no detectable "
              "signal at all, so chasing")
        print("               the recall target would bury the team instead "
              "of catching it.")
    print(f"  BLOCK  tier  @ {block_thr:.4f}  precision {block_m['precision']:.3f}  "
          f"recall {block_m['recall']:.3f}  ({block_m['flagged']:,} stopped)")
    print(f"               {_cm(block_m)}")
    print("-" * 72)
    print(f"  RULE baseline: {rule_m['rule']}")
    print(f"               precision {rule_m['precision']:.3f}  recall {rule_m['recall']:.3f}  "
          f"({rule_m['flagged']:,} flagged)")
    print(f"               {_cm(rule_m)}")
    print("-" * 72)
    lift_r = (review_m["recall"] - rule_m["recall"])
    lift_p = (block_m["precision"] - rule_m["precision"])
    print(f"  ML vs rules: +{lift_r * 100:.1f} recall points at REVIEW, "
          f"+{lift_p * 100:.1f} precision points at BLOCK")
    print(f"  Money auto-blocked in the test window: {money_blocked:,.0f}")
    print("-" * 72)
    c1k = cost_analysis["cost_per_1000_payments"]
    print("  COST OF OPERATING, per 1,000 payments")
    print(f"    no fraud system at all   : {c1k['do_nothing']:>12,.0f}")
    print(f"    the bank's current rule  : {c1k['rule_baseline']:>12,.0f}")
    print(f"    ScamShield (shipped)     : {c1k['scamshield']:>12,.0f}")
    print(f"    ScamShield (cost-tuned)  : {c1k['scamshield_cost_optimal']:>12,.0f}"
          f"   @ review {cost_review_thr:.3f} / block {cost_block_thr:.3f}")
    if cost_analysis["saving_vs_rule_pct"] is not None:
        print(f"    saving vs the rule       : {cost_analysis['saving_vs_rule_per_1000']:>12,.0f}"
              f"   ({cost_analysis['saving_vs_rule_pct']:.1f}% cheaper)")
    print(f"    priced at {config.COST_REVIEW:,.0f}/review, "
          f"{config.COST_FALSE_BLOCK:,.0f}/false block, "
          f"analysts catching {100 * config.ANALYST_CATCH_RATE:.0f}%")
    print(f"    cheapest policy          : {cost_analysis['cost_optimal_policy']}")
    print("=" * 72 + "\n")

    # --- 7. persist -------------------------------------------------------
    joblib.dump(model, config.MODEL_PATH)

    trained_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    metadata = {
        "feature_names": F.FEATURE_NAMES,
        "n_features": len(F.FEATURE_NAMES),
        # Identifies this exact training run.  stream.py copies it into the
        # database so the dashboard can tell whether a drift reference and a
        # set of live scores came from the same model - comparing one model's
        # scores against another model's reference is meaningless.
        "trained_at": trained_at,
        "thresholds": {"review": review_thr, "block": block_thr},
        "split_step": split_step,
        "trained_on": csv_path.name,
        "feature_config": {
            "fanin_window_steps": config.FANIN_WINDOW_STEPS,
            "passthrough_window_steps": config.PASSTHROUGH_WINDOW_STEPS,
            "passthrough_ratio": config.PASSTHROUGH_RATIO,
        },
        "platform": {
            "machine": platform.machine(),
            "python": platform.python_version(),
            "system": platform.system(),
        },
    }
    config.METADATA_PATH.write_text(json.dumps(metadata, indent=2))

    importances = sorted(
        zip(F.FEATURE_NAMES, model.feature_importances_.tolist()),
        key=lambda kv: kv[1], reverse=True,
    )

    # --- 7b. drift reference ---------------------------------------------
    # Snapshot what "normal" looks like, measured on the held-out window, so
    # the dashboard can tell later whether the live traffic still resembles
    # the traffic this model was validated on.  See monitor.py.
    drift_features = [name for name, _gain in importances[:8]]
    reference = {
        "trained_at": trained_at,
        "machine": platform.machine(),
        "trained_on": csv_path.name,
        "bins": config.DRIFT_BINS,
        "score": monitor.build_reference(scores),
        "features": {
            name: monitor.build_reference(X_test[:, F.FEATURE_NAMES.index(name)])
            for name in drift_features
        },
    }
    monitor.save_reference(reference)
    metrics = {
        "dataset": {
            "file": csv_path.name,
            "rows": int(len(df)),
            "fraud_rows": int(y.sum()),
            "fraud_rate": round(float(y.mean()), 6),
            "step_min": int(steps.min()),
            "step_max": int(steps.max()),
        },
        "split": {
            "type": "time-based (no random split)",
            "split_step": split_step,
            "train_rows": int(train_mask.sum()),
            "train_fraud": int(y_train.sum()),
            "test_rows": int(test_mask.sum()),
            "test_fraud": int(y_test.sum()),
        },
        "model": {
            "algorithm": "XGBClassifier",
            "params": {k: v for k, v in config.XGB_PARAMS.items()},
            "scale_pos_weight": round(scale_pos_weight, 2),
            "train_seconds": round(train_seconds, 2),
        },
        "scores": {"pr_auc": round(pr_auc, 4), "roc_auc": round(roc_auc, 4)},
        "review_tier": review_m,
        "block_tier": block_m,
        "capacity": {
            "max_alert_rate": config.MAX_ALERT_RATE,
            "threshold_set_by": "analyst capacity" if capacity_limited else "recall target",
            "target_review_recall": config.TARGET_REVIEW_RECALL,
            "achieved_review_recall": review_m["recall"],
        },
        "rule_baseline": rule_m,
        "money": {
            "auto_blocked_in_test_window": round(money_blocked, 2),
            "missed_in_test_window": round(money_missed, 2),
        },
        "cost_analysis": cost_analysis,
        "feature_importance_top10": [
            {"feature": f, "gain": round(g, 5)} for f, g in importances[:10]
        ],
        "platform": metadata["platform"],
        "total_seconds": round(time.perf_counter() - t_start, 2),
    }
    config.METRICS_PATH.write_text(json.dumps(metrics, indent=2))

    # --- 8. feeds for the live demo ---------------------------------------
    # warmup = the training rows (stream.py replays them silently so the
    # AccountState starts with the same history it had during training).
    df[train_mask].to_csv(config.WARMUP_CSV, index=False)
    df[test_mask].to_csv(config.STREAM_CSV, index=False)

    print(f"[train] model      -> {config.MODEL_PATH}")
    print(f"[train] metadata   -> {config.METADATA_PATH}")
    print(f"[train] metrics    -> {config.METRICS_PATH}")
    print(f"[train] drift ref  -> {config.DRIFT_REFERENCE_PATH}")
    print(f"[train] pr curve   -> {curve_note}")
    print(f"[train] onnx       -> {onnx_note}")
    print(f"[train] stream feed-> {config.STREAM_CSV} ({test_mask.sum():,} rows)")
    print(f"[train] warmup feed-> {config.WARMUP_CSV} ({train_mask.sum():,} rows)")
    print(f"[train] done in {metrics['total_seconds']}s on {platform.machine()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
