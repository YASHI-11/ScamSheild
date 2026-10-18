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

Staying up when the model goes down
-----------------------------------
A payment switch is not allowed to stop taking decisions because a model
misbehaves - every transaction must get an answer inside its deadline, even a
conservative one.  So scoring runs behind a circuit breaker: after a few
consecutive failures the Scorer stops calling the model, falls back to the
static rule, and labels those decisions DEGRADED so the audit trail never
pretends a human-grade model made them.  It probes the model again after a
cooldown and closes the breaker on the first success.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import joblib
import numpy as np
import xgboost as xgb

import config
import explain
from features import FEATURE_NAMES, N_FEATURES, AccountState

ALLOW, REVIEW, BLOCK = "ALLOW", "REVIEW", "BLOCK"

# How a decision was reached.  Written to the database for every transaction.
MODEL, DEGRADED = "MODEL", "DEGRADED"

# Index of the ring-size feature, looked up once so the hot path can report
# the mule-cluster size without a dictionary lookup per transaction.
_RING_SIZE_IDX = FEATURE_NAMES.index("dest_ring_size")


def sigmoid(x: float) -> float:
    # Written out rather than scipy.special.expit - one less dependency.
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


class CircuitBreaker:
    """
    Classic three-state breaker: closed -> open -> half-open -> closed.

    closed     model is healthy, every call goes to it
    open       too many consecutive failures; skip the model entirely
    half-open  cooldown has elapsed, let ONE call through to test the water

    Keeping it in its own class means stream.py and benchmark.py inherit the
    same behaviour, and the tests can drive it without a model at all.
    """

    def __init__(
        self,
        failure_threshold: int = config.BREAKER_FAILURE_THRESHOLD,
        cooldown_seconds: float = config.BREAKER_COOLDOWN_SECONDS,
        clock=time.monotonic,
    ) -> None:
        self.failure_threshold = failure_threshold
        self.cooldown_seconds = cooldown_seconds
        self._clock = clock
        self.consecutive_failures = 0
        self.opened_at: float | None = None
        self.trips = 0            # how many times it has opened (for the UI)
        self.total_failures = 0

    @property
    def is_open(self) -> bool:
        """True when the model should NOT be called right now."""
        if self.opened_at is None:
            return False
        if self._clock() - self.opened_at >= self.cooldown_seconds:
            return False          # half-open: allow one probe through
        return True

    def record_success(self) -> None:
        self.consecutive_failures = 0
        self.opened_at = None

    def record_failure(self) -> None:
        # Work out the state BEFORE recording, because re-arming the timer
        # below would otherwise change the answer.
        was_half_open = self.opened_at is not None and not self.is_open

        self.consecutive_failures += 1
        self.total_failures += 1

        if was_half_open:
            # The probe we let through failed too, so the model is still sick.
            # The timer has to be re-armed: leaving the original opened_at in
            # place would keep the breaker permanently half-open, and every
            # single transaction would go on hitting a model we already know
            # is broken.
            self.opened_at = self._clock()
            self.trips += 1
        elif self.opened_at is None and self.consecutive_failures >= self.failure_threshold:
            self.opened_at = self._clock()
            self.trips += 1

    @property
    def state(self) -> str:
        if self.opened_at is None:
            return "closed"
        return "open" if self.is_open else "half-open"


def rule_decision(txn: dict) -> tuple[float, str, str]:
    """
    The fallback used while the breaker is open: the bank's own static rule.

    Returns (pseudo_score, decision, summary).

    Note the deliberate choice of REVIEW rather than BLOCK.  When the model is
    unavailable we have no calibrated probability, so auto-blocking on a crude
    rule would punish honest customers at exactly the moment we are least able
    to justify it.  Sending it to a human is the conservative answer in both
    directions: nothing is silently allowed, and nothing is silently stopped.
    """
    amount = float(txn.get("amount", 0.0))
    old_orig = float(txn.get("oldbalanceOrg", 0.0))
    new_orig = float(txn.get("newbalanceOrig", 0.0))
    drained = old_orig > 0.0 and new_orig == 0.0
    if amount > config.RULE_AMOUNT and drained:
        return 0.50, REVIEW, (
            "DEGRADED MODE - the model was unavailable, so this was flagged by "
            f"the fallback rule (amount over {config.RULE_AMOUNT:,.0f} and the "
            "sender's account was emptied). Needs a human decision."
        )
    return 0.0, ALLOW, ""


class Scorer:
    """Loads the trained artifacts and scores one transaction at a time."""

    def __init__(
        self,
        model_path: Path | None = None,
        metadata_path: Path | None = None,
        state: AccountState | None = None,
    ) -> None:
        # Resolved here rather than as default arguments on purpose: default
        # arguments are evaluated once, when this module is first imported,
        # which would freeze the paths before SCAMSHIELD_HOME has had a
        # chance to redirect them.  Reading config at call time is what makes
        # the isolated test runs - and any redirected home - work.
        model_path = Path(model_path) if model_path else config.MODEL_PATH
        metadata_path = (Path(metadata_path) if metadata_path
                         else config.METADATA_PATH)

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

        # Resilience.  breaker tracks model health; counters feed the
        # dashboard's health panel.
        self.breaker = CircuitBreaker()
        self.degraded_count = 0
        self.model_count = 0
        # Test/demo hook: set to N to make the next N scoring attempts fail,
        # so the breaker can be shown tripping and recovering on stage.
        # Nothing in the normal path ever sets this.
        self.inject_failures = 0

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

        Availability: if the model raises - or the breaker is already open -
        the transaction still gets a decision, from the static rule, tagged
        DEGRADED.  No transaction ever leaves this method without an answer.
        """
        feats = self.state.features(txn)
        mode = MODEL
        reasons: list[dict] = []

        # Demo/test hook.  The countdown runs per TRANSACTION, not per call to
        # the model: once the breaker opens it stops calling the model at all,
        # so counting calls would stretch a brief outage across one cooldown
        # per remaining failure.  An outage lasts a number of transactions.
        injected = self.inject_failures > 0
        if injected:
            self.inject_failures -= 1

        if self.breaker.is_open:
            # Skip the model entirely; we already know it is unhealthy.
            score, decision, summary = rule_decision(txn)
            mode = DEGRADED
        else:
            try:
                if injected:
                    raise RuntimeError("injected scoring failure (degraded-mode demo)")
                score = self.score_fast(feats)
                self.breaker.record_success()
            except Exception:
                # Any failure at all - bad input, corrupt booster, OOM.
                self.breaker.record_failure()
                score, decision, summary = rule_decision(txn)
                mode = DEGRADED
            else:
                decision = self.decide(score)
                if decision == ALLOW:
                    summary = ""
                else:
                    # Explaining is only paid for on the alerts a human sees.
                    # It must never be the reason a decision fails, so a
                    # failure here degrades the explanation, not the decision.
                    try:
                        _score, contribs = self.score_with_contribs(feats)
                        reasons, summary = explain.explain(contribs, feats, score)
                    except Exception:
                        reasons, summary = [], (
                            "Flagged by the model; detailed reasons unavailable."
                        )

        if mode == MODEL:
            self.model_count += 1
        else:
            self.degraded_count += 1

        self.state.update(txn)
        return {
            "score": score,
            "decision": decision,
            "reasons": reasons,
            "summary": summary,
            "features": feats,
            "mode": mode,
            # Expected loss drives the review queue: an analyst with ten
            # minutes should see the payment with the most money at risk,
            # not merely the highest probability.
            "expected_loss": score * float(txn.get("amount", 0.0)),
            "ring_size": int(feats[_RING_SIZE_IDX]),
        }

    # ------------------------------------------------------------------
    def health(self) -> dict:
        """Snapshot of scoring health, written to the database periodically."""
        total = self.model_count + self.degraded_count
        return {
            "breaker_state": self.breaker.state,
            "breaker_trips": self.breaker.trips,
            "model_failures": self.breaker.total_failures,
            "scored_by_model": self.model_count,
            "scored_degraded": self.degraded_count,
            "degraded_share": (self.degraded_count / total) if total else 0.0,
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
