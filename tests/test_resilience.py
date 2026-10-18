"""
Tests for the part of the system that matters when something is broken.

A real-time payment decision engine is judged on availability as much as on
accuracy: a transaction that gets no answer inside its deadline is a failed
payment, and a model that crashes must not be able to take the switch down
with it.  These tests drive the circuit breaker and the fallback rule directly,
with no model loaded at all, so they stay fast and deterministic.
"""
from __future__ import annotations

import config
from scorer import ALLOW, BLOCK, REVIEW, CircuitBreaker, rule_decision


class FakeClock:
    """A clock we can move by hand, so cooldowns need no sleeping."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


# ----------------------------------------------------------------------
# The breaker
# ----------------------------------------------------------------------
def test_breaker_starts_closed_and_ignores_isolated_failures():
    clock = FakeClock()
    cb = CircuitBreaker(failure_threshold=3, cooldown_seconds=5.0, clock=clock)
    assert cb.state == "closed"
    assert not cb.is_open

    # Two failures with a success between them must not trip anything: a
    # transient blip is not an outage.
    cb.record_failure()
    cb.record_success()
    cb.record_failure()
    assert cb.state == "closed"
    assert cb.trips == 0


def test_breaker_opens_after_consecutive_failures():
    clock = FakeClock()
    cb = CircuitBreaker(failure_threshold=3, cooldown_seconds=5.0, clock=clock)
    for _ in range(3):
        cb.record_failure()
    assert cb.is_open
    assert cb.state == "open"
    assert cb.trips == 1


def test_breaker_half_opens_after_the_cooldown_and_closes_on_success():
    clock = FakeClock()
    cb = CircuitBreaker(failure_threshold=2, cooldown_seconds=5.0, clock=clock)
    cb.record_failure()
    cb.record_failure()
    assert cb.state == "open"

    clock.advance(4.9)
    assert cb.is_open                 # still cooling down

    clock.advance(0.2)
    assert not cb.is_open             # half-open: one probe may through
    assert cb.state == "half-open"

    cb.record_success()               # the probe worked
    assert cb.state == "closed"
    assert cb.consecutive_failures == 0
    assert cb.trips == 1              # the trip is still on the record


def test_breaker_reopens_if_the_probe_fails_again():
    clock = FakeClock()
    cb = CircuitBreaker(failure_threshold=2, cooldown_seconds=5.0, clock=clock)
    cb.record_failure()
    cb.record_failure()
    clock.advance(6.0)
    assert cb.state == "half-open"

    cb.record_failure()               # the model is still sick
    assert cb.is_open
    assert cb.trips == 2
    assert cb.total_failures == 3


# ----------------------------------------------------------------------
# The fallback rule
# ----------------------------------------------------------------------
def _txn(amount: float, old: float, new: float) -> dict:
    return {"amount": amount, "oldbalanceOrg": old, "newbalanceOrig": new}


def test_fallback_never_auto_blocks():
    """
    The crucial safety property.

    While the model is down we have no calibrated probability, so blocking a
    payment outright would punish customers on the strength of a crude rule
    at exactly the moment we are least able to justify it.  Degraded mode is
    allowed to ALLOW and to ask a human - never to BLOCK.
    """
    big_and_drained = _txn(config.RULE_AMOUNT * 5, config.RULE_AMOUNT * 5, 0.0)
    _score, decision, _summary = rule_decision(big_and_drained)
    assert decision == REVIEW
    assert decision != BLOCK


def test_fallback_flags_the_large_drained_payment_and_passes_the_rest():
    over = config.RULE_AMOUNT + 1.0
    under = config.RULE_AMOUNT - 1.0

    _s, decision, summary = rule_decision(_txn(over, over, 0.0))
    assert decision == REVIEW
    assert "DEGRADED" in summary          # the analyst must know why

    # Large but the account was not emptied.
    _s, decision, _ = rule_decision(_txn(over, over * 3, over * 2))
    assert decision == ALLOW

    # Emptied the account, but a small amount.
    _s, decision, _ = rule_decision(_txn(under, under, 0.0))
    assert decision == ALLOW


def test_fallback_always_returns_a_decision():
    """Whatever it is handed, it answers.  No transaction is left hanging."""
    for txn in ({}, {"amount": 0.0}, _txn(1e12, 1e12, 0.0), _txn(-5.0, 0.0, 0.0)):
        score, decision, _summary = rule_decision(txn)
        assert decision in (ALLOW, REVIEW)
        assert 0.0 <= score <= 1.0
