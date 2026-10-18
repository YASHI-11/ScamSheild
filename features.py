"""
ScamShield - feature engineering.

This module is shared by BOTH training (train.py) and the live scoring engine
(stream.py / benchmark.py).  That is deliberate: there is exactly one
implementation of every feature, so what the model learns is exactly what it
sees in production.

The four families of signal
---------------------------
1. Transaction level      - amount, type, does the ledger balance, drained?
2. Mule plumbing          - fan-in, rapid pass-through, how new the accounts are
3. Mule RINGS             - criminals do not use one mule, they use a chain.
                            A union-find structure joins accounts that forward
                            money to each other, so we can ask "how big is the
                            cluster this account belongs to, and how much has
                            flowed through it".
4. Victim side            - is this payment out of character for the SENDER?
                            Scam victims are talked into transfers unlike
                            anything they have ever made before, so comparing
                            against the customer's own history is what makes
                            this protect people rather than only hunt mules.
   plus velocity/structuring: several payments parked just under a reporting
                            threshold inside a few hours.

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
4. Every window eviction is monotone in step (we only ever drop things that
   are older than "now"), so time only moves forward.  Tests assert this.
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
    # --- mule ring / network level -----------------------------------------
    "dest_ring_size",
    "orig_ring_size",
    "dest_ring_volume",
    # --- velocity / structuring --------------------------------------------
    "orig_txns_in_window",
    "orig_amount_in_window",
    "amount_just_below_round",
    # --- victim side (out of character for this customer) ------------------
    "orig_amount_vs_mean",
    "orig_amount_vs_max",
    "orig_first_time_night",
]

N_FEATURES = len(FEATURE_NAMES)

# Columns a transaction mapping must provide.
TXN_COLUMNS = [
    "step", "type", "amount", "nameOrig", "oldbalanceOrg", "newbalanceOrig",
    "nameDest", "oldbalanceDest", "newbalanceDest",
]

# Precomputed structuring bands: (lower_bound, threshold) pairs.
_STRUCTURING_BANDS = tuple(
    (t * (1.0 - config.STRUCTURING_BAND), t) for t in config.STRUCTURING_THRESHOLDS
)

# Night hours as a 24-bit mask, so "has this account ever paid at night?" is
# a single bitwise AND instead of a set lookup.
_NIGHT_MASK = 0
for _h in config.NIGHT_HOURS:
    _NIGHT_MASK |= 1 << int(_h)


def _is_just_below_round(amount: float) -> float:
    """1.0 when the amount sits in the band just under a reporting threshold.

    This is the classic structuring tell: 195,000 rather than 210,000, so the
    payment slips under a 200,000 rule.  A static rule cannot see it; combined
    with the velocity features below, the model can.
    """
    for low, high in _STRUCTURING_BANDS:
        if low <= amount < high:
            return 1.0
    return 0.0


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
        velocity_window_steps: int = config.VELOCITY_WINDOW_STEPS,
    ) -> None:
        self.fanin_window_steps = fanin_window_steps
        self.passthrough_window_steps = passthrough_window_steps
        self.passthrough_ratio = passthrough_ratio
        self.velocity_window_steps = velocity_window_steps

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

        # ---- mule rings: union-find over pass-through edges ---------------
        # We deliberately DO NOT union on every transfer - that would merge
        # the whole population into one component within seconds.  An edge is
        # added only when money demonstrably flowed straight through an
        # account, which is the behaviour that defines a mule chain.
        self._parent: dict[str, str] = {}
        self._ring_size: dict[str, int] = {}      # root -> number of accounts
        self._ring_volume: dict[str, float] = {}  # root -> money passed through

        # ---- velocity: payments made by one sender in the last W steps ----
        self._orig_recent: dict[str, deque] = defaultdict(deque)  # (step, amount)
        self._orig_recent_sum: dict[str, float] = defaultdict(float)

        # ---- victim side: this sender's own spending history ---------------
        self._orig_amount_sum: dict[str, float] = defaultdict(float)
        self._orig_amount_max: dict[str, float] = defaultdict(float)
        self._orig_send_count: dict[str, int] = defaultdict(int)
        self._orig_hour_mask: dict[str, int] = defaultdict(int)

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

    def _evict_velocity(self, orig: str, step: int) -> None:
        """Drop this sender's payments that fell out of the velocity window."""
        log = self._orig_recent.get(orig)
        if not log:
            return
        cutoff = step - self.velocity_window_steps
        total = self._orig_recent_sum[orig]
        while log and log[0][0] <= cutoff:
            _, old_amount = log.popleft()
            total -= old_amount
        self._orig_recent_sum[orig] = max(total, 0.0)

    # ---- union-find (mule rings) --------------------------------------
    def _find(self, account: str) -> str:
        """Root of this account's cluster, with iterative path compression."""
        parent = self._parent
        root = parent.get(account)
        if root is None:
            return account
        # Walk to the root.
        while True:
            nxt = parent.get(root)
            if nxt is None or nxt == root:
                break
            root = nxt
        # Compress the path so the next lookup is O(1).
        cur = account
        while cur != root:
            nxt = parent.get(cur, root)
            parent[cur] = root
            cur = nxt
        return root

    def _ring_add_edge(self, a: str, b: str, amount: float) -> None:
        """Record that money passed straight through from a to b."""
        ra, rb = self._find(a), self._find(b)
        if ra == rb:
            # Already the same cluster - just book the extra volume.
            self._ring_volume[ra] = self._ring_volume.get(ra, 0.0) + amount
            return
        sa = self._ring_size.get(ra, 1)
        sb = self._ring_size.get(rb, 1)
        # Union by size, so the trees stay shallow.
        if sa < sb:
            ra, rb, sa, sb = rb, ra, sb, sa
        self._parent[rb] = ra
        self._parent.setdefault(ra, ra)
        self._ring_size[ra] = sa + sb
        self._ring_volume[ra] = (
            self._ring_volume.get(ra, 0.0) + self._ring_volume.get(rb, 0.0) + amount
        )
        self._ring_size.pop(rb, None)
        self._ring_volume.pop(rb, None)

    def ring_of(self, account: str) -> tuple[int, float]:
        """(cluster size, money passed through the cluster) for one account.

        Size 1 means "not part of any observed pass-through chain".
        """
        if account not in self._parent:
            return 1, 0.0
        root = self._find(account)
        return self._ring_size.get(root, 1), self._ring_volume.get(root, 0.0)

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

        # ---- mule rings ---------------------------------------------------
        dest_ring_size, dest_ring_volume = self.ring_of(dest)
        orig_ring_size, _orig_ring_volume = self.ring_of(orig)

        # ---- velocity / structuring ---------------------------------------
        self._evict_velocity(orig, step)
        orig_txns_window = float(len(self._orig_recent.get(orig, ())))
        orig_amount_window = float(self._orig_recent_sum.get(orig, 0.0))
        just_below = _is_just_below_round(amount)

        # ---- victim side: is this out of character for the sender? --------
        # With no history there is no evidence either way, so we emit the
        # neutral value 1.0 ("exactly typical") rather than 0.  The model can
        # tell the two apart via orig_prior_txn_count.
        n_sent = self._orig_send_count.get(orig, 0)
        if n_sent > 0:
            mean_amount = self._orig_amount_sum[orig] / n_sent
            vs_mean = amount / (mean_amount + 1.0)
            vs_max = amount / (self._orig_amount_max[orig] + 1.0)
        else:
            vs_mean = 1.0
            vs_max = 1.0

        # First time this customer has ever transacted in the small hours.
        first_night = 0.0
        if n_sent > 0 and ((1 << hour) & _NIGHT_MASK):
            if not (self._orig_hour_mask.get(orig, 0) & _NIGHT_MASK):
                first_night = 1.0

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
                float(dest_ring_size),
                float(orig_ring_size),
                dest_ring_volume,
                orig_txns_window,
                orig_amount_window,
                just_below,
                vs_mean,
                vs_max,
                first_night,
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
        hour = step % 24

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
                    # This is the edge that defines a mule ring: money went
                    # in one side of `orig` and straight out to `dest`.
                    self._ring_add_edge(orig, dest, amount)
            # ... and lands on dest.
            self._last_inflow[dest] = (step, amount)

            # Victim-side history is about what an account SENDS.
            self._orig_amount_sum[orig] += amount
            if amount > self._orig_amount_max[orig]:
                self._orig_amount_max[orig] = amount
            self._orig_send_count[orig] += 1

            # Velocity window (outgoing payments only).
            self._evict_velocity(orig, step)
            self._orig_recent[orig].append((step, amount))
            self._orig_recent_sum[orig] += amount
        else:
            # CASH_IN: the customer (nameOrig) is the one receiving money.
            self._last_inflow[orig] = (step, amount)

        self._orig_hour_mask[orig] |= 1 << hour
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

    # Diagnostics ------------------------------------------------------
    def largest_rings(self, k: int = 5) -> list[tuple[int, float]]:
        """The k biggest pass-through clusters as (size, volume) pairs.

        Used by the dashboard's network panel and by the tests; never on the
        scoring hot path.
        """
        rings = [
            (size, self._ring_volume.get(root, 0.0))
            for root, size in self._ring_size.items()
            if size > 1
        ]
        rings.sort(key=lambda sv: (sv[0], sv[1]), reverse=True)
        return rings[:k]


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
