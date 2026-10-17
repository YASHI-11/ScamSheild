"""
The single most important test in the project.

ScamShield trains in batch and scores in a stream.  If those two paths ever
disagree, the model is being fed something at run time that it never saw in
training, and every metric becomes a lie.

So this file re-derives every account-level feature from first principles -
by scanning the raw history for each row, with no deques, counters or
incremental state - and asserts that the O(1) streaming AccountState produces
exactly the same numbers.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

import config
import features as F
from features import CASH_IN, FEATURE_NAMES, AccountState


# ----------------------------------------------------------------------
# A deliberately naive, O(n^2) reference implementation.
# Written from the feature DEFINITIONS, not from features.py.
# ----------------------------------------------------------------------
def _received_money(row: dict, account: str) -> bool:
    """Did `account` receive money in this row?"""
    if row["type"] == CASH_IN:
        return row["nameOrig"] == account        # a deposit credits the sender
    return row["nameDest"] == account            # everything else credits the dest


def _last_inflow_before(rows: list[dict], i: int, account: str):
    """(step, amount) of the most recent money `account` received before row i."""
    for k in range(i - 1, -1, -1):
        if _received_money(rows[k], account):
            return rows[k]["step"], rows[k]["amount"]
    return None


def _is_passthrough(rows: list[dict], j: int, account: str) -> bool:
    """Did `account` forward on most of a very recent inflow, in row j?"""
    row = rows[j]
    if row["type"] == CASH_IN or row["nameOrig"] != account:
        return False
    inflow = _last_inflow_before(rows, j, account)
    if inflow is None:
        return False
    in_step, in_amount = inflow
    return (
        0 <= row["step"] - in_step <= config.PASSTHROUGH_WINDOW_STEPS
        and in_amount > 0
        and row["amount"] >= config.PASSTHROUGH_RATIO * in_amount
    )


def reference_features(rows: list[dict], i: int) -> dict[str, float]:
    """Every feature for row i, computed by scanning rows[0:i] only."""
    row = rows[i]
    step, ttype, amount = row["step"], row["type"], row["amount"]
    orig, dest = row["nameOrig"], row["nameDest"]
    past = rows[:i]

    # --- transaction level ------------------------------------------
    out = {
        "amount": amount,
        "log_amount": math.log1p(max(amount, 0.0)),
        "amount_to_balance_ratio": amount / (row["oldbalanceOrg"] + 1.0),
        "hour_of_day": float(step % 24),
        "balance_error_orig": row["oldbalanceOrg"] - amount - row["newbalanceOrig"],
        "balance_error_dest": row["oldbalanceDest"] + amount - row["newbalanceDest"],
        "orig_drained_to_zero": float(row["oldbalanceOrg"] > 0 and row["newbalanceOrig"] == 0),
        "dest_was_empty": float(row["oldbalanceDest"] == 0),
    }
    for t in F.TXN_TYPES:
        out[f"type_{t}"] = float(ttype == t)

    # --- account level, by brute force -------------------------------
    senders_in_window = {
        r["nameOrig"] for r in past
        if r["nameDest"] == dest and r["step"] > step - config.FANIN_WINDOW_STEPS
    }
    out["dest_fanin_senders"] = float(len(senders_in_window))
    out["dest_new_sender"] = float(
        not any(r["nameOrig"] == orig and r["nameDest"] == dest for r in past)
    )
    out["dest_rapid_passthrough"] = float(
        any(_is_passthrough(rows, j, dest) for j in range(i))
    )

    orig_pt = False
    if ttype != CASH_IN:
        inflow = _last_inflow_before(rows, i, orig)
        if inflow is not None:
            in_step, in_amount = inflow
            orig_pt = (
                0 <= step - in_step <= config.PASSTHROUGH_WINDOW_STEPS
                and in_amount > 0
                and amount >= config.PASSTHROUGH_RATIO * in_amount
            )
    out["orig_rapid_passthrough"] = float(orig_pt)
    out["orig_prior_txn_count"] = float(sum(r["nameOrig"] == orig for r in past))
    out["dest_prior_inflow_count"] = float(sum(r["nameDest"] == dest for r in past))
    return out


# ----------------------------------------------------------------------
# Test data: small, but hitting every pattern the features care about.
# ----------------------------------------------------------------------
def _make_rows(n: int = 260, seed: int = 7) -> list[dict]:
    rng = np.random.default_rng(seed)
    accounts = [f"C{i}" for i in range(14)]
    mule = "C_MULE"
    merchants = [f"M{i}" for i in range(4)]
    types = ["PAYMENT", "CASH_IN", "CASH_OUT", "TRANSFER", "DEBIT"]

    rows: list[dict] = []
    step = 1
    for k in range(n):
        # Advance the clock irregularly so window eviction really gets tested.
        if k % 3 == 0:
            step += int(rng.integers(0, 3))
        ttype = str(rng.choice(types))
        orig = str(rng.choice(accounts))
        # Every few rows, aim at the mule: builds fan-in and pass-through.
        if k % 5 == 0:
            dest = mule
        elif ttype in ("PAYMENT", "DEBIT", "CASH_IN"):
            dest = str(rng.choice(merchants))
        else:
            dest = str(rng.choice(accounts))
        # ... and let the mule forward it straight back out.
        if k % 5 == 2:
            orig, ttype = mule, "CASH_OUT"

        old_orig = float(rng.integers(0, 200000))
        amount = float(rng.integers(0, 150000))
        rows.append({
            "step": step,
            "type": ttype,
            "amount": amount,
            "nameOrig": orig,
            "oldbalanceOrg": old_orig,
            "newbalanceOrig": max(old_orig - amount, 0.0) if rng.random() < 0.8 else 0.0,
            "nameDest": dest,
            "oldbalanceDest": float(rng.integers(0, 100000)),
            "newbalanceDest": float(rng.integers(0, 100000)),
        })
    return rows


@pytest.fixture(scope="module")
def rows() -> list[dict]:
    return _make_rows()


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------
def test_incremental_state_matches_batch_reference(rows):
    """The headline guarantee: streaming features == definitional features."""
    state = AccountState()
    for i, txn in enumerate(rows):
        got = state.features(txn)          # uses only rows 0..i-1
        state.update(txn)                  # only now does row i become history

        want = reference_features(rows, i)
        for j, name in enumerate(FEATURE_NAMES):
            assert got[j] == pytest.approx(want[name], rel=1e-5, abs=1e-4), (
                f"row {i}, feature '{name}': streaming={got[j]} reference={want[name]}"
            )


def test_fanin_window_actually_expires():
    """
    Senders outside the look-back window must stop counting.

    Note the fresh state per query: eviction is deliberately one-way, because
    a payment stream only ever moves forwards.  Querying an earlier step after
    a later one is not a supported operation, so each case replays from zero.
    """
    dest = "C_TARGET"
    base = dict(type="TRANSFER", amount=100.0, oldbalanceOrg=1000.0,
                newbalanceOrig=900.0, nameDest=dest, oldbalanceDest=0.0,
                newbalanceDest=100.0)
    idx = FEATURE_NAMES.index("dest_fanin_senders")

    def fanin_at(step: int) -> float:
        state = AccountState(fanin_window_steps=3)
        for s, sender in ((1, "A"), (2, "B"), (3, "C")):
            state.update({**base, "step": s, "nameOrig": sender})
        return float(state.features({**base, "step": step, "nameOrig": "D"})[idx])

    # Window is "step > query_step - 3".
    assert fanin_at(3) == 3.0     # steps 1,2,3 all count (> 0)
    assert fanin_at(4) == 2.0     # steps 2,3 count (> 1); step 1 has aged out
    assert fanin_at(5) == 1.0     # only step 3 counts (> 2)
    assert fanin_at(6) == 0.0     # everything has aged out


def test_fanin_counts_distinct_senders_not_transactions():
    """Ten payments from one account is fan-in of 1, not 10."""
    base = dict(type="TRANSFER", amount=50.0, oldbalanceOrg=1000.0,
                newbalanceOrig=950.0, nameDest="C_TARGET", oldbalanceDest=0.0,
                newbalanceDest=50.0, step=1)
    idx = FEATURE_NAMES.index("dest_fanin_senders")

    repeat = AccountState()
    for _ in range(10):
        repeat.update({**base, "nameOrig": "A"})
    assert repeat.features({**base, "nameOrig": "Z"})[idx] == 1.0

    spread = AccountState()
    for k in range(10):
        spread.update({**base, "nameOrig": f"A{k}"})
    assert spread.features({**base, "nameOrig": "Z"})[idx] == 10.0


def test_rapid_passthrough_fires_only_inside_the_window():
    """Money in, then most of it straight back out == mule behaviour."""
    idx = FEATURE_NAMES.index("orig_rapid_passthrough")
    mule = "C_MULE"
    inflow = dict(step=10, type="TRANSFER", amount=100_000.0, nameOrig="C_VICTIM",
                  oldbalanceOrg=100_000.0, newbalanceOrig=0.0, nameDest=mule,
                  oldbalanceDest=0.0, newbalanceDest=100_000.0)

    def forward(step: int, amount: float) -> float:
        state = AccountState()
        state.update(inflow)
        out = dict(step=step, type="CASH_OUT", amount=amount, nameOrig=mule,
                   oldbalanceOrg=100_000.0, newbalanceOrig=0.0, nameDest="C_OTHER",
                   oldbalanceDest=0.0, newbalanceDest=0.0)
        return float(state.features(out)[idx])

    assert forward(10, 95_000.0) == 1.0    # same hour, 95% of it -> mule
    assert forward(12, 95_000.0) == 1.0    # still inside the 2-step window
    assert forward(13, 95_000.0) == 0.0    # too slow to look like a mule
    assert forward(10, 20_000.0) == 0.0    # only a fifth forwarded on


def test_features_is_side_effect_free(rows):
    """Calling features() twice must give the same answer - it must not learn."""
    state = AccountState()
    for txn in rows[:120]:
        a = state.features(txn)
        b = state.features(txn)
        np.testing.assert_array_equal(a, b)
        state.update(txn)


def test_no_lookahead_future_rows_cannot_change_the_past(rows):
    """
    Feed the same prefix, then two completely different futures.  The feature
    vectors for the prefix must be bit-identical - if they are not, something
    is leaking backwards in time.
    """
    prefix = rows[:90]
    future_a = rows[90:150]
    future_b = list(reversed(rows[150:210]))

    def replay(seq):
        state = AccountState()
        return np.array([state.process(t) for t in seq])

    out_a = replay(prefix + future_a)[: len(prefix)]
    out_b = replay(prefix + future_b)[: len(prefix)]
    np.testing.assert_array_equal(out_a, out_b)


def test_build_feature_matrix_matches_manual_replay(rows):
    """The DataFrame helper used by train.py must agree with the row loop."""
    df = pd.DataFrame(rows)
    matrix, _ = F.build_feature_matrix(df)

    state = AccountState()
    manual = np.array([state.process(t) for t in rows], dtype=np.float32)
    np.testing.assert_array_equal(matrix, manual)
    assert matrix.shape == (len(rows), len(FEATURE_NAMES))


def test_warm_up_equals_full_replay(rows):
    """
    stream.py warms up on the training rows with warm_up() (no features) and
    then continues with process().  That must land in the same place as
    replaying everything through process() from the start.
    """
    split = 140
    warmed = F.warm_up(pd.DataFrame(rows[:split]))
    continued = np.array([warmed.process(t) for t in rows[split:]], dtype=np.float32)

    full = AccountState()
    whole = np.array([full.process(t) for t in rows], dtype=np.float32)
    np.testing.assert_array_equal(continued, whole[split:])


def test_feature_vector_is_finite_and_right_shape(rows):
    state = AccountState()
    for txn in rows:
        v = state.process(txn)
        assert v.shape == (len(FEATURE_NAMES),)
        assert v.dtype == np.float32
        assert np.isfinite(v).all()
