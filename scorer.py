"""
ScamShield - the real-time scoring core.

One class, Scorer, used identically by stream.py (live demo) and benchmark.py
(throughput measurement).  Keeping it in one place guarantees the number the
benchmark prints is the number the demo actually achieves.

The clever bit
--------------
XGBoost's pred_contribs=True returns one TreeSHAP contribution per feature
PLUS a trailing bias term, and those values sum to the raw margin.  So a
SINGLE booster call gives us both the score and the explanation:

    margin = contribs.sum()
    score  = sigmoid(margin)

No second prediction pass, and no `shap` package - which matters because shap
has no s390x wheel.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import joblib
import numpy as np
import xgboost as xgb

import config
import explain
from features import FEATURE_NAMES, N_FEATURES, AccountState

ALLOW, REVIEW, BLOCK = "ALLOW", "REVIEW", "BLOCK"


def sigmoid(x: float) -> float:
    # Written out rather than scipy.special.expit - one less dependency.
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


class Scorer:
    """Loads the trained artifacts and scores one transaction at a time."""

    def __init__(
        self,
        model_path: Path = config.MODEL_PATH,
        metadata_path: Path = config.METADATA_PATH,
        state: AccountState | None = None,
    ) -> None:
        if not Path(model_path).exists():
            raise SystemExit(
                f"[scorer] {model_path} not found - run  python train.py  first."
            )
        self.model = joblib.load(model_path)
        self.booster = self.model.get_booster()

        meta = json.loads(Path(metadata_path).read_text())
        self.feature_names: list[str] = meta["feature_names"]
        self.review_threshold: float = float(meta["thresholds"]["review"])
        self.block_threshold: float = float(meta["thresholds"]["block"])
        self.metadata = meta

        if self.feature_names != FEATURE_NAMES:
            raise SystemExit(
                "[scorer] features.py has changed since the model was trained.\n"
                "         Re-run  python train.py  to keep them in sync."
            )

        self.state = state if state is not None else AccountState()
        # Reused scratch buffer: avoids allocating a (1, n) array per txn.
        self._buf = np.empty((1, N_FEATURES), dtype=np.float32)

    # ------------------------------------------------------------------
    def decide(self, score: float) -> str:
        if score >= self.block_threshold:
            return BLOCK
        if score >= self.review_threshold:
            return REVIEW
        return ALLOW

    def score_fast(self, feats: np.ndarray) -> float:
        """
        Probability only - the hot path every transaction takes.

        inplace_predict skips building a DMatrix, which turns out to be ~60%
        of the cost of scoring a single row.  Verified bit-identical to
        model.predict_proba in tests/test_pipeline.py.
        """
        self._buf[0] = feats
        return float(self.booster.inplace_predict(self._buf)[0])

    def score_with_contribs(self, feats: np.ndarray) -> tuple[float, np.ndarray]:
        """Score AND explain.  Only paid for on transactions we flag."""
        self._buf[0] = feats
        dmat = xgb.DMatrix(self._buf)
        contribs = self.booster.predict(dmat, pred_contribs=True)[0]
        # contribs sums to the raw margin; the last element is the bias term.
        return sigmoid(float(contribs.sum())), contribs[:-1]

    def process(self, txn: dict) -> dict:
        """
        The full real-time decision for one transaction.

        Order matters: features FIRST (using only the past), then fold the
        transaction into the rolling state.  See features.py.

        Cost control: the ~99% of payments that are obviously fine get the
        cheap score-only path.  TreeSHAP runs only for the handful we are
        going to put in front of a human, so the average latency stays low
        without giving up a reason for any alert.
        """
        feats = self.state.features(txn)
        score = self.score_fast(feats)
        decision = self.decide(score)

        if decision == ALLOW:
            reasons, summary = [], ""
        else:
            _score, contribs = self.score_with_contribs(feats)
            reasons, summary = explain.explain(contribs, feats, score)

        self.state.update(txn)
        return {
            "score": score,
            "decision": decision,
            "reasons": reasons,
            "summary": summary,
            "features": feats,
        }

    # ------------------------------------------------------------------
    def reasons_json(self, reasons: list[dict]) -> str:
        return json.dumps([
            {
                "feature": r["feature"],
                "value": round(r["value"], 4),
                "contribution": round(r["contribution"], 4),
            }
            for r in reasons
        ])
