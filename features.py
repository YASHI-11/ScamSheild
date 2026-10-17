"""
ScamShield - feature engineering.

This module is shared by BOTH training (train.py) and the live scoring engine
(stream.py / benchmark.py).  That is deliberate: there is exactly one
implementation of every feature, so what the model learns is exactly what it
sees in production.

Leakage rules enforced here
---------------------------
1. AccountState.features(txn) only ever reads state built from *earlier*
   transactions.  The caller must always do:

       feats = state.features(txn)   # score using the past
       state.update(txn)             # only then fold this txn into the past

2. Training replays the rows in time order through the very same class, so a
   batch-trained model and the streaming engine agree row for row.
3. isFlaggedFraud is never a feature - it is the bank's own legacy rule and
   would leak the answer.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict, deque
from typing import Any, Iterable, Mapping

import numpy as np

import config

# PaySim transaction types, fixed order so the one-hot columns are stable.
TXN_TYPES = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]

# In PaySim, CASH_IN is the only type where money flows *into* nameOrig
# (a deposit).  For every other type the money leaves nameOrig and lands on
# nameDest.  This single rule drives all the "mule" plumbing below.
CASH_IN = "CASH_IN"

FEATURE_NAMES: list[str] = [
    # --- transaction level -------------------------------------------------
    "amount",
    "log_amount",
    "amount_to_balance_ratio",
    "hour_of_day",
    "type_CASH_IN",
    "type_CASH_OUT",
    "type_DEBIT",
    "type_PAYMENT",
    "type_TRANSFER",
    "balance_error_orig",
    "balance_error_dest",
    "orig_drained_to_zero",
    "dest_was_empty",
    # --- account level (rolling, incremental) ------------------------------
    "dest_fanin_senders",
    "dest_new_sender",
    "dest_rapid_passthrough",
    "orig_rapid_passthrough",
    "orig_prior_txn_count",
    "dest_prior_inflow_count",
]

N_FEATURES = len(FEATURE_NAMES)

# Columns a transaction mapping must provide.
TXN_COLUMNS = [
    "step", "type", "amount", "nameOrig", "oldbalanceOrg", "newbalanceOrig",
    "nameDest", "oldbalanceDest", "newbalanceDest",
]


class AccountState:
    """
    Rolling, per-account memory of everything seen *so far*.

    Designed for streaming: update() is O(1) amortised and features() never
    scans history, so the same object can run for millions of transactions.
    """

    def __init__(
        self,
        fanin_window_steps: int = config.FANIN_WINDOW_STEPS,
        passthrough_window_steps: int = config.PASSTHROUGH_WINDOW_STEPS,
        passthrough_ratio: float = config.PASSTHROUGH_RATIO,
    ) -> None:
        self.fanin_window_steps = fanin_window_steps
        self.passthrough_window_steps = passthrough_window_steps
        self.passthrough_ratio = passthrough_ratio

        # dest -> deque[(step, sender)] limited to the fan-in window
        self._dest_sender_log: dict[str, deque] = defaultdict(deque)
        # dest -> Counter[sender] kept in sync with the deque above
        self._dest_sender_counts: dict[str, Counter] = defaultdict(Counter)
        # (sender, dest) pairs ever observed -> powers "first time sender"
        self._seen_pairs: set[tuple[str, str]] = set()
        # account -> (step, amount) of the last money it RECEIVED
        self._last_inflow: dict[str, tuple[int, float]] = {}
        # account -> how many times it received money and forwarded most of it
        self._passthrough_events: dict[str, int] = defaultdict(int)
        # account -> number of transactions it originated
        self._orig_txn_count: dict[str, int] = defaultdict(int)
        # account -> number of times it was the destination
        self._dest_inflow_count: dict[str, int] = defaultdict(int)

        self.n_seen = 0

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------
    def _evict(self, dest: str, step: int) -> None:
        """Drop fan-in events older than the window.

        A sender event at step s counts for the current step iff
        s > step - fanin_window_steps (i.e. the last N steps, current step
        included).  Eviction is monotone in step, so calling it from
        features() is safe and does not change any answer.
        """
        log = self._dest_sender_log.get(dest)
        if not log:
            return
        counts = self._dest_sender_counts[dest]
        cutoff = step - self.fanin_window_steps
        while log and log[0][0] <= cutoff:
            _, old_sender = log.popleft()
            counts[old_sender] -= 1
            if counts[old_sender] <= 0:
                del counts[old_sender]

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def features(self, txn: Mapping[str, Any]) -> np.ndarray:
        """Feature vector for txn using ONLY transactions already update()d."""
        step = int(txn["step"])
        ttype = str(txn["type"])
        amount = float(txn["amount"])
        orig = txn["nameOrig"]
        dest = txn["nameDest"]
        old_orig = float(txn["oldbalanceOrg"])
        new_orig = float(txn["newbalanceOrig"])
        old_dest = float(txn["oldbalanceDest"])
        new_dest = float(txn["newbalanceDest"])

        # ---- transaction level -------------------------------------------
        log_amount = math.log1p(max(amount, 0.0))
        amount_ratio = amount / (old_orig + 1.0)
        hour = step % 24

        # Classic PaySim tell: the ledger does not balance.
        balance_error_orig = old_orig - amount - new_orig
        balance_error_dest = old_dest + amount - new_dest

        orig_drained = 1.0 if (old_orig > 0.0 and new_orig == 0.0) else 0.0
        dest_was_empty = 1.0 if old_dest == 0.0 else 0.0

        # ---- account level ------------------------------------------------
        self._evict(dest, step)
        fanin = float(len(self._dest_sender_counts.get(dest, ())))
        new_sender = 0.0 if (orig, dest) in self._seen_pairs else 1.0

        dest_passthrough = 1.0 if self._passthrough_events.get(dest, 0) > 0 else 0.0

        # Live mule signal: this account received money very recently and is
        # right now forwarding most of it onwards.
        orig_passthrough = 0.0
        if ttype != CASH_IN:
            inflow = self._last_inflow.get(orig)
            if inflow is not None:
                in_step, in_amount = inflow
                if (
                    0 <= step - in_step <= self.passthrough_window_steps
                    and in_amount > 0.0
                    and amount >= self.passthrough_ratio * in_amount
                ):
                    orig_passthrough = 1.0

        orig_prior = float(self._orig_txn_count.get(orig, 0))
        dest_prior_in = float(self._dest_inflow_count.get(dest, 0))

        return np.array(
            [
                amount,
                log_amount,
                amount_ratio,
                float(hour),
                1.0 if ttype == "CASH_IN" else 0.0,
                1.0 if ttype == "CASH_OUT" else 0.0,
                1.0 if ttype == "DEBIT" else 0.0,
                1.0 if ttype == "PAYMENT" else 0.0,
                1.0 if ttype == "TRANSFER" else 0.0,
                balance_error_orig,
                balance_error_dest,
                orig_drained,
                dest_was_empty,
                fanin,
                new_sender,
                dest_passthrough,
                orig_passthrough,
                orig_prior,
                dest_prior_in,
            ],
            dtype=np.float32,
        )

    def update(self, txn: Mapping[str, Any]) -> None:
        """Fold txn into the rolling state.  Call AFTER features(txn)."""
        step = int(txn["step"])
        ttype = str(txn["type"])
        amount = float(txn["amount"])
        orig = txn["nameOrig"]
        dest = txn["nameDest"]

        if ttype != CASH_IN:
            # Money leaves orig.  Did it arrive only moments ago?  Then orig
            # just acted as a pass-through (money mule) account.
            inflow = self._last_inflow.get(orig)
            if inflow is not None:
                in_step, in_amount = inflow
                if (
                    0 <= step - in_step <= self.passthrough_window_steps
                    and in_amount > 0.0
                    and amount >= self.passthrough_ratio * in_amount
                ):
                    self._passthrough_events[orig] += 1
            # ... and lands on dest.
            self._last_inflow[dest] = (step, amount)
        else:
            # CASH_IN: the customer (nameOrig) is the one receiving money.
            self._last_inflow[orig] = (step, amount)

        self._orig_txn_count[orig] += 1
        self._dest_inflow_count[dest] += 1

        self._evict(dest, step)
        self._dest_sender_log[dest].append((step, orig))
        self._dest_sender_counts[dest][orig] += 1
        self._seen_pairs.add((orig, dest))

        self.n_seen += 1

    # Convenience ------------------------------------------------------
    def process(self, txn: Mapping[str, Any]) -> np.ndarray:
        """features() then update(), in the only correct order."""
        feats = self.features(txn)
        self.update(txn)
        return feats


# ----------------------------------------------------------------------
# Helpers used by train.py / benchmark.py / tests
# ----------------------------------------------------------------------
def iter_txn_dicts(df) -> Iterable[dict]:
    """Yield lightweight dicts from a DataFrame, in row order (fast path)."""
    cols = [df[c].to_numpy() for c in TXN_COLUMNS]
    for i in range(len(df)):
        yield {name: col[i] for name, col in zip(TXN_COLUMNS, cols)}


def build_feature_matrix(df, state: AccountState | None = None):
    """
    Replay df (already sorted in time order) through an AccountState and
    return the resulting (n_rows, N_FEATURES) matrix plus the final state.

    Pass an existing state to continue a replay (e.g. warm up on the training
    rows, then keep going on the test rows).
    """
    if state is None:
        state = AccountState()
    out = np.empty((len(df), N_FEATURES), dtype=np.float32)
    for i, txn in enumerate(iter_txn_dicts(df)):
        out[i] = state.process(txn)
    return out, state


def warm_up(df, state: AccountState | None = None) -> AccountState:
    """Replay rows into the state WITHOUT computing features (much faster)."""
    if state is None:
        state = AccountState()
    for txn in iter_txn_dicts(df):
        state.update(txn)
    return state


def sort_time_order(df):
    """Stable sort by step - the only ordering ScamShield ever uses."""
    return df.sort_values("step", kind="mergesort").reset_index(drop=True)
