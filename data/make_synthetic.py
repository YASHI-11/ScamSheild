"""
Generate a small PaySim-SHAPED dataset so the whole ScamShield pipeline can be
run end to end without downloading anything from Kaggle.

It is NOT a scientific simulator - it only needs to reproduce the structure and
the fraud patterns that ScamShield looks for:

  FRAUD
  * "account takeover": TRANSFER that drains the victim to zero, immediately
    followed by the mule CASH_OUT-ing the money (rapid pass-through)
  * "mule collection": one destination account receiving money from many
    different senders inside a short window (high fan-in)
  * "layering chain": the money hops through 3-5 mule accounts within minutes,
    each forwarding almost all of it on - this is what creates a mule RING
  * "structuring": one large transfer split into several payments parked just
    under a reporting threshold
  * "social engineering": a long-standing customer is talked into one huge
    night-time payment to a brand-new payee.  The ledger balances perfectly,
    so the ONLY tells are behavioural

  LEGITIMATE, BUT LOOKS BAD
  * the big honest purchase that empties an account
  * a small business settling eight suppliers in one hour (high velocity)
  * a house deposit: large, to a payee never seen before
  * a 195,000 car purchase, which sits in the structuring band all by itself

That last group matters as much as the fraud.  Without it the new features
would be free wins - "high velocity" would mean fraud and nothing else - and
the model would score perfectly while learning nothing a bank could use.

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

# An extra column this generator adds on top of the PaySim schema, naming the
# pattern that produced each row.  Nothing in the model pipeline reads it -
# features.py only ever looks at COLUMNS, and train.py only at isFraud - but
# it lets ablation.py report recall per scam typology, which is the question
# a fraud team actually asks ("are we catching the mule rings?").  The real
# Kaggle file has no such column, so every consumer treats it as optional.
TYPE_COLUMN = "fraud_type"

LEGIT_TYPES = ["PAYMENT", "CASH_IN", "CASH_OUT", "DEBIT", "TRANSFER"]
LEGIT_WEIGHTS = [0.44, 0.22, 0.22, 0.02, 0.10]

# The threshold criminals structure their payments under.
STRUCTURING_TARGET = config.RULE_AMOUNT


def _round2(x: float) -> float:
    return float(np.round(x, 2))


def _pick_established(rng, sent_count: dict[str, int], customers, min_n: int = 3) -> str:
    """
    Choose a customer who already has a transaction history.

    The victim-side features compare a payment against the sender's OWN past,
    so a scam aimed at an account with no history would be undetectable by
    construction - and unrealistic: scammers target real, active customers.
    """
    established = [acct for acct, n in sent_count.items() if n >= min_n]
    if established:
        return str(rng.choice(established))
    return str(rng.choice(customers))


def generate(n_rows: int, seed: int = config.RANDOM_SEED, fraud_rate: float = 0.004) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    n_steps = max(48, min(744, n_rows // 60))      # ~1 month of hourly steps
    n_customers = max(200, n_rows // 6)
    n_merchants = max(50, n_rows // 40)
    n_mules = max(10, n_rows // 400)

    customers = np.array([f"C{1000000000 + i}" for i in range(n_customers)])
    merchants = np.array([f"M{1000000000 + i}" for i in range(n_merchants)])
    mules = np.array([f"C{9000000000 + i}" for i in range(n_mules)])

    # "Holding" accounts: criminal destinations that simply SIT on the money
    # instead of forwarding it straight on.
    #
    # They exist to keep the experiment honest.  Every scam that ends at a
    # `mules` account inherits a rapid-pass-through history, and that single
    # feature then explains almost all the fraud - which made the first
    # ablation show the newer features adding nothing, not because they are
    # useless but because the data handed the answer to one old feature.
    # Social engineering and structuring therefore pay into holders, so the
    # ONLY evidence is behavioural: the amount against the customer's own
    # history, the hour, and the burst of payments.
    n_holders = max(8, n_rows // 600)
    holders = np.array([f"C{8000000000 + i}" for i in range(n_holders)])

    # A handful of accounts behave like small businesses: they pay a lot of
    # suppliers, often in bursts.  They are the legitimate counterweight to
    # the velocity/structuring features.
    n_businesses = max(5, n_customers // 20)
    businesses = customers[:n_businesses]

    # Running ledger so balance columns are self-consistent for legit traffic
    # (fraudulent rows deliberately break it, exactly like real PaySim).
    balance: dict[str, float] = {c: _round2(rng.gamma(2.0, 30000.0)) for c in customers}
    for m in mules:
        balance[m] = 0.0
    for h in holders:
        balance[h] = 0.0

    # Customers who genuinely live at night (shift workers, other time zones).
    # Without them "it happened at 3am" alone would separate the victim scams,
    # and the per-customer night feature would never have to do any work.
    n_nightowls = max(5, n_customers // 25)
    nightowls = customers[n_businesses:n_businesses + n_nightowls]

    # How many payments each account has SENT so far.  Used only to aim the
    # scams at customers who have a history worth deviating from.
    sent_count: dict[str, int] = {}

    rows: list[tuple] = []
    # One label per row in `rows`, naming the pattern that generated it.
    fraud_types: list[str] = []

    def mark(name: str, start: int) -> None:
        """Label every row appended since index `start`."""
        fraud_types.extend([name] * (len(rows) - start))

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

        _m = len(rows)
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
            if ttype != "CASH_IN":
                sent_count[orig] = sent_count.get(orig, 0) + 1
            rows.append((step, ttype, amount, orig, old_orig, new_orig,
                         dest, old_dest, new_dest, 0, 0))

        mark("legit", _m)

        # -------- legitimate but scary-looking: the big honest purchase ----
        # A customer empties their account on one large, genuine transfer.
        # These are the rows that make a naive "large AND drained" rule fire
        # on innocent people - which is exactly the cost ScamShield removes.
        _m = len(rows)
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
            sent_count[buyer] = sent_count.get(buyer, 0) + 1

        mark("legit_big_purchase", _m)

        # -------- legitimate burst: a business settling its suppliers ------
        # High velocity with nothing wrong about it.  Stops the model from
        # concluding that "many payments in an hour" means fraud.
        _m = len(rows)
        if rng.random() < 0.5:
            biz = str(rng.choice(businesses))
            n_pay = int(rng.integers(4, 10))
            for _ in range(n_pay):
                old_orig = max(balance.get(biz, 0.0), _round2(rng.gamma(3.0, 40000.0)))
                amount = _round2(min(old_orig, rng.gamma(2.0, 6000.0)))
                dest = str(rng.choice(merchants))
                new_orig = _round2(max(old_orig - amount, 0.0))
                rows.append((step, "PAYMENT", amount, biz, old_orig, new_orig,
                             dest, 0.0, 0.0, 0, 0))
                balance[biz] = new_orig
                sent_count[biz] = sent_count.get(biz, 0) + 1

        mark("legit_business_burst", _m)

        # -------- legitimate: a house deposit to a brand-new payee ---------
        # Large, out of character, and to someone this customer has never
        # paid before - but in office hours, and the recipient keeps it.
        _m = len(rows)
        if 9 <= hour <= 17 and rng.random() < 0.45:
            buyer = _pick_established(rng, sent_count, customers)
            old_orig = max(balance.get(buyer, 0.0), _round2(rng.gamma(3.0, 120000.0)))
            amount = _round2(old_orig * float(rng.uniform(0.5, 0.9)))
            dest = str(rng.choice(customers))
            while dest == buyer:
                dest = str(rng.choice(customers))
            old_dest = balance.get(dest, 0.0)
            new_orig = _round2(old_orig - amount)
            new_dest = _round2(old_dest + amount)
            rows.append((step, "TRANSFER", amount, buyer, old_orig, new_orig,
                         dest, old_dest, new_dest, 0, 0))
            balance[buyer] = new_orig
            balance[dest] = new_dest
            sent_count[buyer] = sent_count.get(buyer, 0) + 1

        mark("legit_new_payee_large", _m)

        # -------- legitimate: one payment inside the structuring band ------
        # A 195,000 car.  One of these on its own is not structuring; the
        # model has to combine the band with the velocity to tell them apart.
        _m = len(rows)
        if rng.random() < 0.35:
            buyer = str(rng.choice(customers))
            amount = _round2(STRUCTURING_TARGET * float(rng.uniform(0.91, 0.99)))
            old_orig = max(balance.get(buyer, 0.0), _round2(amount * rng.uniform(1.1, 2.0)))
            dest = str(rng.choice(merchants))
            new_orig = _round2(old_orig - amount)
            rows.append((step, "PAYMENT", amount, buyer, old_orig, new_orig,
                         dest, 0.0, 0.0, 0, 0))
            balance[buyer] = new_orig
            sent_count[buyer] = sent_count.get(buyer, 0) + 1

        mark("legit_threshold_band", _m)

        # -------- legitimate: a night owl, sometimes with a big payment ----
        # Same hour, same "unusual" look, entirely innocent.  This is what
        # forces the model to use "unusual FOR THIS CUSTOMER" rather than
        # simply "after midnight".
        _m = len(rows)
        if hour in (0, 1, 2, 3, 4, 5) and rng.random() < 0.8:
            owl = str(rng.choice(nightowls))
            old_orig = max(balance.get(owl, 0.0), _round2(rng.gamma(2.0, 50000.0)))
            big = rng.random() < 0.25
            amount = _round2(old_orig * float(rng.uniform(0.6, 0.95)) if big
                             else min(old_orig, rng.gamma(1.5, 3000.0)))
            dest = str(rng.choice(customers if big else merchants))
            old_dest = balance.get(dest, 0.0) if big else 0.0
            new_orig = _round2(max(old_orig - amount, 0.0))
            rows.append((step, "TRANSFER" if big else "PAYMENT", amount, owl,
                         old_orig, new_orig, dest, old_dest,
                         _round2(old_dest + amount) if big else 0.0, 0, 0))
            balance[owl] = new_orig
            if big:
                balance[dest] = _round2(old_dest + amount)
            sent_count[owl] = sent_count.get(owl, 0) + 1
        mark("legit_nightowl", _m)

        # ---------------- fraud: takeover + pass-through ----------------
        _m = len(rows)
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

        mark("takeover", _m)

        # ---------------- fraud with NO signal at all --------------------
        # Not every fraud leaves a trace.  A compromised card used for an
        # ordinary-sized payment to an ordinary merchant in the afternoon is
        # indistinguishable from a real purchase using only the data a
        # payment switch can see - the bank finds out weeks later from a
        # chargeback.
        #
        # These rows exist to stop the model being flattered.  Without them
        # the synthetic data is perfectly separable, recall comes out at
        # ~100%, and the result tells you nothing about how the system would
        # behave in production.  They put a realistic ceiling on recall and
        # are the reason the headline numbers below are believable.
        _m = len(rows)
        for _ in range(int(rng.poisson(0.35))):
            victim = str(rng.choice(customers))
            old_orig = max(balance.get(victim, 0.0), _round2(rng.gamma(2.0, 20000.0)))
            amount = _round2(min(old_orig, rng.gamma(1.5, 2500.0)))
            dest = str(rng.choice(merchants))
            new_orig = _round2(old_orig - amount)
            rows.append((step, "PAYMENT", amount, victim, old_orig, new_orig,
                         dest, 0.0, 0.0, 1, 0))
            balance[victim] = new_orig
            sent_count[victim] = sent_count.get(victim, 0) + 1

        mark("no_signal", _m)

        # ---------------- fraud: mule collection (high fan-in) ----------
        _m = len(rows)
        if step % 23 == 0:
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

        mark("mule_collection", _m)

        # ---------------- fraud: layering chain (the mule RING) ---------
        # The money does not stop at the first mule.  It hops through a
        # chain of them within a couple of hours, each forwarding almost
        # everything on, before the last one cashes out.  Every hop is a
        # rapid pass-through, which is exactly the edge features.py unions
        # into a cluster - so these rows are what give dest_ring_size and
        # dest_ring_volume their signal.
        _m = len(rows)
        if step % 17 == 0:
            victim = _pick_established(rng, sent_count, customers)
            start_balance = max(balance.get(victim, 0.0),
                                _round2(rng.gamma(3.0, 70000.0)))
            chain_len = int(rng.integers(3, 6))
            chain = [str(m) for m in rng.choice(mules, size=chain_len, replace=False)]

            amount = _round2(start_balance * float(rng.uniform(0.80, 1.0)))
            rows.append((step, "TRANSFER", amount, victim, start_balance,
                         _round2(start_balance - amount),
                         chain[0], 0.0, 0.0, 1, 0))
            balance[victim] = _round2(start_balance - amount)
            sent_count[victim] = sent_count.get(victim, 0) + 1

            # Each hop keeps a small cut and forwards the rest immediately.
            held = amount
            for hop in range(chain_len - 1):
                hop_step = int(min(n_steps, step + hop))
                forwarded = _round2(held * float(rng.uniform(0.85, 0.97)))
                rows.append((hop_step, "TRANSFER", forwarded, chain[hop], held,
                             _round2(held - forwarded),
                             chain[hop + 1], 0.0, 0.0, 1, 0))
                held = forwarded

            # The end of the chain turns it into cash.
            out_step = int(min(n_steps, step + chain_len - 1))
            out_amount = _round2(held * float(rng.uniform(0.88, 1.0)))
            rows.append((out_step, "CASH_OUT", out_amount, chain[-1], held,
                         _round2(max(held - out_amount, 0.0)),
                         str(rng.choice(customers)), 0.0, 0.0, 1, 0))

        mark("layering_chain", _m)

        # ---------------- fraud: structuring / smurfing -----------------
        # The criminal knows the bank flags transfers over the threshold, so
        # the money leaves in several payments each parked just underneath
        # it.  No single payment breaks the rule; the burst does.
        _m = len(rows)
        if step % 29 == 0:
            smurf = _pick_established(rng, sent_count, customers)
            # A holding account, NOT a mule: it keeps the money, so there is
            # no pass-through tell.  The burst itself is the only evidence.
            collector = str(rng.choice(holders))
            n_splits = int(rng.integers(4, 8))
            pot = max(balance.get(smurf, 0.0),
                      _round2(STRUCTURING_TARGET * n_splits * 1.05))
            held = balance.get(collector, 0.0)
            for j in range(n_splits):
                amount = _round2(STRUCTURING_TARGET * float(rng.uniform(0.90, 0.995)))
                if amount > pot:
                    break
                split_step = int(min(n_steps, step + j // 3))
                # Both ledgers balance: no arithmetic red flag either.
                rows.append((split_step, "TRANSFER", amount, smurf, pot,
                             _round2(pot - amount), collector, held,
                             _round2(held + amount), 1, 0))
                pot = _round2(pot - amount)
                held = _round2(held + amount)
                sent_count[smurf] = sent_count.get(smurf, 0) + 1
            balance[smurf] = pot
            balance[collector] = held

        mark("structuring", _m)

        # ---------------- fraud: social engineering (the victim scam) ---
        # A long-standing customer is on the phone to someone claiming to be
        # their bank, and makes one enormous payment at 3am to an account
        # they have never paid before.  Note that BOTH ledgers balance
        # perfectly: there is no arithmetic tell at all.  The only evidence
        # is behavioural - the amount against this customer's own history,
        # the hour, the brand-new payee - which is precisely what the
        # victim-side features measure.
        _m = len(rows)
        if hour in (0, 1, 2, 3, 4, 5) and rng.random() < 0.35:
            victim = _pick_established(rng, sent_count, customers, min_n=4)
            old_orig = max(balance.get(victim, 0.0), _round2(rng.gamma(3.0, 80000.0)))
            amount = _round2(old_orig * float(rng.uniform(0.70, 1.0)))
            # Again a holding account: the scammer's account receives the
            # money and sits on it, so nothing about the RECIPIENT gives it
            # away either.  Only the sender's own behaviour is out of place.
            scammer = str(rng.choice(holders))
            old_dest = balance.get(scammer, 0.0)
            rows.append((step, "TRANSFER", amount, victim, old_orig,
                         _round2(old_orig - amount),
                         scammer, old_dest, _round2(old_dest + amount), 1, 0))
            balance[victim] = _round2(old_orig - amount)
            balance[scammer] = _round2(old_dest + amount)
            sent_count[victim] = sent_count.get(victim, 0) + 1

        mark("social_engineering", _m)

    df = pd.DataFrame(rows, columns=COLUMNS)
    # Attach the typology labels before sorting so they stay with their rows.
    assert len(fraud_types) == len(df), "a generator block forgot to label its rows"
    df[TYPE_COLUMN] = fraud_types
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
    print("[make_synthetic] patterns:")
    counts = df[df["isFraud"] == 1][TYPE_COLUMN].value_counts()
    for name, n in counts.items():
        print(f"    fraud  {name:<22} {n:>6,}")
    legit = df[df["isFraud"] == 0][TYPE_COLUMN].value_counts()
    for name, n in legit.items():
        print(f"    legit  {name:<22} {n:>6,}")


if __name__ == "__main__":
    main()
