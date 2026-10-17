"""
Generate a small PaySim-SHAPED dataset so the whole ScamShield pipeline can be
run end to end without downloading anything from Kaggle.

It is NOT a scientific simulator - it only needs to reproduce the structure and
the fraud patterns that ScamShield looks for:

  * legitimate PAYMENT / CASH_IN / CASH_OUT / DEBIT traffic
  * "account takeover": TRANSFER that drains the victim to zero, immediately
    followed by the mule CASH_OUT-ing the money (rapid pass-through)
  * "mule collection": one destination account receiving money from many
    different senders inside a short window (high fan-in)

Usage
-----
    python data/make_synthetic.py                       # 40k rows -> data/paysim_sample.csv
    python data/make_synthetic.py --rows 100000 --out data/paysim_sample.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

COLUMNS = [
    "step", "type", "amount", "nameOrig", "oldbalanceOrg", "newbalanceOrig",
    "nameDest", "oldbalanceDest", "newbalanceDest", "isFraud", "isFlaggedFraud",
]

LEGIT_TYPES = ["PAYMENT", "CASH_IN", "CASH_OUT", "DEBIT", "TRANSFER"]
LEGIT_WEIGHTS = [0.44, 0.22, 0.22, 0.02, 0.10]


def _round2(x: float) -> float:
    return float(np.round(x, 2))


def generate(n_rows: int, seed: int = config.RANDOM_SEED, fraud_rate: float = 0.005) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    n_steps = max(48, min(744, n_rows // 60))      # ~1 month of hourly steps
    n_customers = max(200, n_rows // 6)
    n_merchants = max(50, n_rows // 40)
    n_mules = max(10, n_rows // 400)

    customers = np.array([f"C{1000000000 + i}" for i in range(n_customers)])
    merchants = np.array([f"M{1000000000 + i}" for i in range(n_merchants)])
    mules = np.array([f"C{9000000000 + i}" for i in range(n_mules)])

    # Running ledger so balance columns are self-consistent for legit traffic
    # (fraudulent rows deliberately break it, exactly like real PaySim).
    balance: dict[str, float] = {c: _round2(rng.gamma(2.0, 30000.0)) for c in customers}
    for m in mules:
        balance[m] = 0.0

    rows: list[tuple] = []
    n_fraud_target = max(20, int(n_rows * fraud_rate))
    fraud_steps = rng.integers(1, n_steps, size=n_fraud_target)
    fraud_by_step: dict[int, int] = {}
    for s in fraud_steps:
        fraud_by_step[int(s)] = fraud_by_step.get(int(s), 0) + 1

    per_step = max(1, n_rows // n_steps)

    for step in range(1, n_steps + 1):
        # ---------------- legitimate traffic ----------------------------
        hour = step % 24
        # fewer transactions in the small hours - gives hour_of_day signal
        activity = 0.35 if hour < 6 else 1.0
        n_legit = max(1, int(rng.poisson(per_step * activity)))

        for _ in range(n_legit):
            ttype = str(rng.choice(LEGIT_TYPES, p=LEGIT_WEIGHTS))
            orig = str(rng.choice(customers))
            old_orig = balance.get(orig, 0.0)

            if ttype == "CASH_IN":
                amount = _round2(rng.gamma(2.0, 4000.0))
                dest = str(rng.choice(merchants))
                new_orig = _round2(old_orig + amount)
                old_dest = new_dest = 0.0
            elif ttype in ("PAYMENT", "DEBIT"):
                amount = _round2(min(old_orig, rng.gamma(1.5, 2500.0)))
                dest = str(rng.choice(merchants))
                new_orig = _round2(max(old_orig - amount, 0.0))
                old_dest = new_dest = 0.0
            else:  # CASH_OUT / TRANSFER between customers
                amount = _round2(min(old_orig, rng.gamma(1.8, 9000.0)))
                dest = str(rng.choice(customers))
                while dest == orig:
                    dest = str(rng.choice(customers))
                old_dest = balance.get(dest, 0.0)
                new_orig = _round2(max(old_orig - amount, 0.0))
                new_dest = _round2(old_dest + amount)
                balance[dest] = new_dest

            balance[orig] = new_orig
            rows.append((step, ttype, amount, orig, old_orig, new_orig,
                         dest, old_dest, new_dest, 0, 0))

        # -------- legitimate but scary-looking: the big honest purchase ----
        # A customer empties their account on one large, genuine transfer.
        # These are the rows that make a naive "large AND drained" rule fire
        # on innocent people - which is exactly the cost ScamShield removes.
        for _ in range(int(rng.poisson(0.6))):
            buyer = str(rng.choice(customers))
            old_orig = max(balance.get(buyer, 0.0), _round2(rng.gamma(3.0, 90000.0)))
            dest = str(rng.choice(customers))
            while dest == buyer:
                dest = str(rng.choice(customers))
            old_dest = balance.get(dest, 0.0)
            rows.append((step, "TRANSFER", _round2(old_orig), buyer, old_orig, 0.0,
                         dest, old_dest, _round2(old_dest + old_orig), 0, 0))
            balance[buyer] = 0.0
            balance[dest] = _round2(old_dest + old_orig)

        # ---------------- fraud: takeover + pass-through ----------------
        for _ in range(fraud_by_step.get(step, 0)):
            victim = str(rng.choice(customers))
            victim_balance = max(balance.get(victim, 0.0), _round2(rng.gamma(3.0, 60000.0)))
            mule = str(rng.choice(mules))

            # 1 in 3 scams is "low and slow": the criminal only takes part of
            # the balance and keeps the books looking tidy, so no single
            # transaction-level red flag fires.  These are the hard cases.
            stealthy = rng.random() < 0.33
            if stealthy:
                amount = _round2(victim_balance * float(rng.uniform(0.25, 0.65)))
                new_victim = _round2(victim_balance - amount)
                dest_old = _round2(rng.gamma(1.5, 5000.0))
                dest_new = _round2(dest_old + amount)
            else:
                amount = _round2(victim_balance)
                new_victim = 0.0
                # PaySim leaves destination balances at 0 on fraud -> ledger error
                dest_old = dest_new = 0.0

            rows.append((step, "TRANSFER", amount, victim, victim_balance, new_victim,
                         mule, dest_old, dest_new, 1, 0))
            balance[victim] = new_victim

            # 2) The mule cashes (almost) all of it out within a step or two.
            out_step = int(min(n_steps, step + rng.integers(0, 2)))
            out_amount = _round2(amount * float(rng.uniform(0.85, 1.0)))
            out_old = dest_new if stealthy else amount
            rows.append((out_step, "CASH_OUT", out_amount, mule, out_old,
                         _round2(max(out_old - out_amount, 0.0)),
                         str(rng.choice(customers)), 0.0, 0.0, 1, 0))

        # ---------------- fraud: mule collection (high fan-in) ----------
        if step % 20 == 0:
            collector = str(rng.choice(mules))
            n_senders = int(rng.integers(8, 20))
            running = balance.get(collector, 0.0)
            for _ in range(n_senders):
                sender = str(rng.choice(customers))
                sender_balance = max(balance.get(sender, 0.0), 500.0)
                amount = _round2(sender_balance * float(rng.uniform(0.6, 1.0)))
                rows.append((step, "TRANSFER", amount, sender, sender_balance,
                             _round2(sender_balance - amount),
                             collector, running, 0.0, 1, 0))
                balance[sender] = _round2(sender_balance - amount)
                running = _round2(running + amount)
            balance[collector] = running

    df = pd.DataFrame(rows, columns=COLUMNS)
    df = df.sort_values("step", kind="mergesort").reset_index(drop=True)

    # PaySim's own legacy rule: flag transfers above 200k (rarely correct).
    df["isFlaggedFraud"] = (
        (df["type"] == "TRANSFER") & (df["amount"] > 200000)
    ).astype(int)
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate PaySim-shaped synthetic data")
    ap.add_argument("--rows", type=int, default=40000, help="approximate number of rows")
    ap.add_argument("--out", type=str, default=str(config.SAMPLE_CSV))
    ap.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    args = ap.parse_args()

    df = generate(args.rows, seed=args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    n_fraud = int(df["isFraud"].sum())
    print(f"[make_synthetic] wrote {len(df):,} rows -> {out}")
    print(f"[make_synthetic] fraud rows: {n_fraud:,} ({100 * n_fraud / len(df):.2f}%)")
    print(f"[make_synthetic] steps: {df['step'].min()} .. {df['step'].max()}")


if __name__ == "__main__":
    main()
