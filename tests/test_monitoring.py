"""
Tests for drift detection.

A drift monitor that cries wolf is worse than none at all - the team learns to
ignore it, and then misses the real thing.  So these tests pin both halves of
the behaviour: it must stay quiet when nothing has changed, and it must fire
when the distribution genuinely moves.
"""
from __future__ import annotations

import numpy as np
import pytest

import config
import monitor


@pytest.fixture
def baseline() -> np.ndarray:
    """A skewed, fraud-score-shaped distribution: almost everything near 0."""
    rng = np.random.default_rng(11)
    return rng.beta(0.4, 12.0, size=20_000)


def test_identical_distribution_has_almost_no_drift(baseline):
    reference = monitor.build_reference(baseline)
    value = monitor.psi(reference, baseline)
    assert value < 0.01
    assert monitor.psi_band(value) == monitor.STABLE


def test_resampling_the_same_distribution_stays_stable(baseline):
    """Sampling noise alone must not register as drift."""
    rng = np.random.default_rng(99)
    reference = monitor.build_reference(baseline)
    fresh = rng.beta(0.4, 12.0, size=5_000)
    value = monitor.psi(reference, fresh)
    assert value < config.PSI_WARN
    assert monitor.psi_band(value) == monitor.STABLE


def test_a_real_shift_is_detected(baseline):
    """
    Scores drifting upwards - what happens when fraud tactics change and the
    model starts seeing patterns it was not calibrated on - must be caught.
    """
    rng = np.random.default_rng(7)
    reference = monitor.build_reference(baseline)
    shifted = rng.beta(3.0, 3.0, size=5_000)      # a completely different shape
    value = monitor.psi(reference, shifted)
    assert value > config.PSI_ALERT
    assert monitor.psi_band(value) == monitor.SIGNIFICANT


def test_psi_grows_monotonically_with_the_size_of_the_shift(baseline):
    """A bigger change must produce a bigger number, or the scale is useless."""
    rng = np.random.default_rng(3)
    reference = monitor.build_reference(baseline)
    values = []
    for extra in (0.0, 0.05, 0.20, 0.50):
        live = np.clip(rng.beta(0.4, 12.0, size=8_000) + extra, 0.0, 1.0)
        values.append(monitor.psi(reference, live))
    assert values == sorted(values)
    assert values[0] < config.PSI_WARN < values[-1]


def test_bands_match_the_industry_convention():
    assert monitor.psi_band(0.0) == monitor.STABLE
    assert monitor.psi_band(config.PSI_WARN - 1e-9) == monitor.STABLE
    assert monitor.psi_band(config.PSI_WARN) == monitor.MODERATE
    assert monitor.psi_band(config.PSI_ALERT - 1e-9) == monitor.MODERATE
    assert monitor.psi_band(config.PSI_ALERT) == monitor.SIGNIFICANT
    assert monitor.psi_band(10.0) == monitor.SIGNIFICANT


def test_empty_and_degenerate_inputs_do_not_explode():
    """
    The dashboard calls this on every refresh, including before any data has
    arrived.  It must return a quiet zero rather than NaN, inf or a traceback.
    """
    assert monitor.psi({"edges": [], "pct": []}, [0.1, 0.2]) == 0.0
    assert monitor.psi(monitor.build_reference([0.1, 0.2, 0.3]), []) == 0.0
    assert monitor.build_reference([])["n"] == 0

    # Every value identical - the quantile edges collapse.
    constant = monitor.build_reference([0.5] * 100)
    value = monitor.psi(constant, [0.5] * 50)
    assert np.isfinite(value)

    # An unseen extreme must land in the outer bin, not fall off the end.
    reference = monitor.build_reference(np.linspace(0.0, 1.0, 1000))
    assert np.isfinite(monitor.psi(reference, [50.0, -50.0]))


def test_score_drift_wrapper_handles_a_missing_reference():
    out = monitor.score_drift(None, [0.1, 0.2])
    assert out["available"] is False
    assert out["psi"] == 0.0
