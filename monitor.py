"""
ScamShield - model health monitoring (drift detection).

The problem this solves
-----------------------
A fraud model is the only kind of model whose adversary actively rewrites the
input distribution.  Criminals change tactics weekly; the model does not.  So
a system that was excellent in January is quietly mediocre by June, and
nothing in the usual accuracy dashboard tells you, because in production you
do not have labels - you find out months later from chargebacks.

What you CAN measure without labels is whether the data still looks like the
data you trained on.  The industry standard for that is the Population
Stability Index:

    PSI = sum over bins of  (actual% - expected%) * ln(actual% / expected%)

with the conventional reading:

    PSI < 0.10   stable, no action
    0.10 - 0.25  moderate shift, keep watching
    PSI > 0.25   significant shift, retrain

train.py stores the reference distribution (the score histogram and the
histograms of the most important features, measured on the held-out window).
The dashboard compares the live stream against it on every refresh.

Everything here is pure numpy, so it runs unchanged on s390x.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

import config

# Floor applied to any empty bin.  Without it a single empty bin sends PSI to
# infinity, which is mathematically correct and operationally useless.
_EPS = 1e-6

STABLE, MODERATE, SIGNIFICANT = "STABLE", "MODERATE", "SIGNIFICANT"


def build_reference(values: Sequence[float], bins: int = config.DRIFT_BINS) -> dict:
    """
    Capture a distribution as quantile bin edges plus the share in each bin.

    Quantile edges (rather than equal-width ones) matter here because fraud
    scores are extremely skewed - almost everything sits near zero, and
    equal-width bins would put 99.9% of the mass in bin 0 and detect nothing.
    """
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"edges": [], "pct": [], "n": 0}

    qs = np.linspace(0.0, 1.0, bins + 1)
    edges = np.unique(np.quantile(arr, qs))
    if edges.size < 2:
        # Degenerate: every value identical.  One bin, everything in it.
        return {"edges": [float(edges[0]), float(edges[0]) + 1.0],
                "pct": [1.0], "n": int(arr.size)}

    # Open the outer edges so unseen extremes in live data still land in a bin.
    inner = edges[1:-1].tolist()
    full = [-np.inf, *inner, np.inf]
    counts, _ = np.histogram(arr, bins=full)
    pct = counts / max(counts.sum(), 1)
    return {
        "edges": [float(e) for e in full],
        "pct": [float(p) for p in pct],
        "n": int(arr.size),
    }


def psi(reference: dict, live_values: Iterable[float]) -> float:
    """
    Population Stability Index of live_values against a stored reference.

    Returns 0.0 when there is nothing meaningful to compare, so a quiet
    dashboard never shows a scary number it cannot justify.
    """
    edges = reference.get("edges") or []
    ref_pct = np.asarray(reference.get("pct") or [], dtype=np.float64)
    if len(edges) < 2 or ref_pct.size == 0:
        return 0.0

    arr = np.asarray(list(live_values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0

    counts, _ = np.histogram(arr, bins=np.asarray(edges, dtype=np.float64))
    live_pct = counts / max(counts.sum(), 1)

    # Clip both sides: an empty bin on either side would otherwise blow up.
    ref_c = np.clip(ref_pct, _EPS, None)
    live_c = np.clip(live_pct, _EPS, None)
    return float(np.sum((live_c - ref_c) * np.log(live_c / ref_c)))


def psi_band(value: float) -> str:
    """Translate a PSI number into the conventional traffic light."""
    if value >= config.PSI_ALERT:
        return SIGNIFICANT
    if value >= config.PSI_WARN:
        return MODERATE
    return STABLE


# Both helpers resolve the default path at call time, not as a default
# argument: default arguments are evaluated at import, which would freeze the
# location before SCAMSHIELD_HOME could redirect it.
def save_reference(payload: dict, path: Path | None = None) -> None:
    target = Path(path) if path else config.DRIFT_REFERENCE_PATH
    Path(target).write_text(json.dumps(payload, indent=2))


def load_reference(path: Path | None = None) -> dict | None:
    """Load the stored reference, or None if training has not written one."""
    p = Path(path) if path else config.DRIFT_REFERENCE_PATH
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def score_drift(reference: dict | None, live_scores: Iterable[float]) -> dict:
    """Convenience wrapper used by the dashboard: PSI + band for the scores."""
    if not reference or "score" not in reference:
        return {"available": False, "psi": 0.0, "band": STABLE}
    value = psi(reference["score"], live_scores)
    return {
        "available": True,
        "psi": round(value, 4),
        "band": psi_band(value),
        "reference_n": reference["score"].get("n", 0),
        "trained_at": reference.get("trained_at", ""),
        "trained_on": reference.get("machine", ""),
    }
