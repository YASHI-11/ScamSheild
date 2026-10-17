# ScamShield — what it is, in plain words

## The one-sentence version

**ScamShield watches payments as they happen, spots the ones that look like a
scam or a money-mule account, and hands the suspicious ones to a human with a
sentence explaining why.**

---

## 1. The problem

A digital payment (UPI, mobile money) settles in about **two seconds**. In that
window a bank must decide: is this a grocery bill, or a pensioner's savings
being drained by a fraudster?

The hard part is that **a single payment almost never looks criminal.** The
fraud is in the *pattern*:

> One account receives money from 14 different strangers in 2 hours,
> then forwards almost all of it onward within minutes.

That account is a **money mule** — a middleman account criminals use to move
stolen money. Every single transfer into it looks perfectly normal on its own.
Only the shape of the whole picture gives it away.

Most banks today use a fixed rule like *"flag payments over 200,000 that empty
the account."* On our test data that rule catches **7%** of fraud and is
**wrong 2 times out of 3** when it does fire. It annoys honest customers and
misses nearly all the real criminals.

---

## 2. What ScamShield actually does

For every payment, in under a millisecond:

1. **Look at the payment itself** — amount, type, does the balance arithmetic
   actually add up, did it empty the account, what hour is it?
2. **Look at the two accounts' recent history** — how many different people
   have paid this recipient lately? Did this account receive money and
   immediately forward it? Is the sender brand new?
3. **Score it** 0 to 1 with a trained model (XGBoost).
4. **Decide** one of three things:

| Decision | Meaning | Who acts |
|---|---|---|
| 🟢 **ALLOW** | Looks fine | Nobody. It just goes through. |
| 🟡 **REVIEW** | Suspicious | **A human analyst decides.** |
| 🔴 **BLOCK** | Almost certainly fraud | Stopped automatically. |

5. **Explain it in English.** Not "risk score 0.97", but:

   > *"Recipient's ledger does not add up (off by 737.5K); recipient has a
   > history of forwarding money straight back out; unusually large amount
   > of 51.0K."*

6. **Record the human's decision** (Approve / Block / Escalate) in an audit
   table — which is both the regulator's paper trail and the training data
   for the next model.

**The human makes the final call.** ScamShield's job is to turn 30,000
payments into a queue of about a dozen, and make each one readable in five
seconds.

---

## 3. What runs where

This is the part worth being precise about, because the project spans two
machines.

### 🖥️ Your laptop — the *development* machine

| What | Why |
|---|---|
| Writing and editing the code | Normal development |
| Running the tests (`pytest`) | Fast feedback |
| Running the **dashboard** (the web page you look at) | See below — it *has* to be here |
| A local copy of the engine, for a live demo | So transactions visibly scroll in |

### 🏛️ The IBM LinuxONE server (`148.100.108.119`) — the *real* machine

This is an IBM Z mainframe guest. `uname -m` says **`s390x`** — a completely
different processor architecture from your laptop's Intel chip.

| What | Why |
|---|---|
| **Training the model** (`train.py`) | This is the real workload |
| **Scoring transactions** (`stream.py`) | The actual real-time decision engine |
| **The benchmark** (`benchmark.py`) | Proves how fast Z is |
| **The test suite** | Proves the code genuinely works on s390x |

The same Python files run on both. Nothing was rewritten for the mainframe.
The only architecture-specific line in the entire project is the one that
*measures* the architecture (`platform.machine()`).

### Why the dashboard can't run on the server

The dashboard is built with **Streamlit**, which requires a library called
**pyarrow**. pyarrow has never been published for s390x, and isn't in Ubuntu's
package archive either. Installing it would mean compiling Apache Arrow (a
large C++ project) from source on the mainframe — roughly an hour, with a good
chance of failing.

So we split it:

```
   LinuxONE (s390x)                      Your laptop
   ┌────────────────────┐                ┌────────────────────┐
   │  stream.py         │                │  app.py            │
   │  scores payments   │ ──────────────▶│  shows them to you │
   │  writes results    │   scamshield   │  (just a viewer)   │
   │  to a database     │      .db       │                    │
   └────────────────────┘                └────────────────────┘
      does the thinking                     draws the picture
```

**This is not a fudge.** The engine records which machine did the scoring into
the database, so the dashboard's *"Running on"* panel correctly reports
**s390x, Python 3.12.3, 3 CPU cores** — it is describing the mainframe that
did the work, not the laptop drawing the page.

---

## 4. The two things you can show on screen

You have two different dashboards, and **they cannot be the same screen** —
because one shows a mainframe and the other shows your laptop.

| | **Live demo** | **LinuxONE evidence** |
|---|---|---|
| Command | `streamlit run app.py` | `SCAMSHIELD_DB=./linuxone_results/scamshield.db streamlit run app.py` |
| Badge says | `AMD64` (your laptop) | ✅ **`s390x`** |
| Transactions | scrolling in live, ~50/sec | frozen snapshot of 30,537 |
| Good for | showing it *works* | proving it ran *on Z* |

Run them on two ports (8501 and 8502) and you can show both.

---

## 5. The results (measured, not estimated)

### Does the AI beat the old rule?

| | Old fixed rule | **ScamShield** |
|---|---|---|
| Fraud caught | 7% | **97%** |
| False alarms | 80 | **3** |
| Accuracy when it fires | 33% | **99%** |

### Is the mainframe actually better than the laptop?

Same code, same 50,000 transactions, three runs each:

| | Your laptop (12 cores) | **LinuxONE (3 cores)** |
|---|---|---|
| Transactions per second | 1,434 – 2,762 | **11,524 – 12,021** |
| Typical decision time | 0.25 – 0.65 ms | **0.058 ms** |
| Model training | 3.8 sec | **1.8 sec** |

**4–8× faster on a quarter of the cores.**

But here is the number that actually matters, and it's the one to say out loud
to judges:

| | Laptop | **LinuxONE** |
|---|---|---|
| How much the speed varied between identical runs | **1.9×** | **1.04×** |
| Worst case vs typical case (p95 ÷ p50) | 2.2 – 3.8× | **1.15×** |

Your laptop's speed nearly *halved* between two identical runs, and its slow
payments took up to 4× longer than its typical ones. On the mainframe, three
runs landed within 4% of each other, and even the slow payments were only 15%
slower than typical.

**Why that matters:** a payment has a hard two-second deadline. You don't care
about the best speed you can achieve — you care about the **worst speed you can
guarantee**. Mainframes are built for exactly that consistency, and it's the
one thing a laptop benchmark can't fake.

In one full run on LinuxONE, the system processed 30,537 payments in 5.1
seconds and blocked **62,473,305** in fraud.

---

## 6. How the pieces fit together

```
data/make_synthetic.py   →  invents realistic fake payment data
         │                   (so you don't need the Kaggle download)
         ▼
train.py                 →  learns what fraud looks like
         │                   saves: the model + two decision thresholds
         ▼
stream.py                →  replays payments one by one, scores each,
         │                   writes every decision into scamshield.db
         ▼
app.py                   →  the dashboard you look at; analysts click
                             Approve / Block / Escalate
```

Two supporting files do the clever bits:

- **`features.py`** — works out the "signals" from each payment. Critically,
  this *same file* is used both when training and when running live, so the
  model can never be fed something in production it didn't see in training.
  (That mismatch is the single most common way fraud systems quietly break.)
- **`explain.py`** — turns the model's maths into the English sentence the
  analyst reads.

---

## 7. Honest limitations

- The data is **synthetic**. The near-perfect scores come from generated data
  whose patterns are cleaner than real life. Running it on the real Kaggle
  PaySim file is the next step, and the numbers will be lower.
- The dashboard shows whether each payment was *really* fraud, because we're
  replaying a dataset where we know the answer. In the real world that column
  doesn't exist — which is exactly why the analyst feedback loop matters.
- The account memory grows forever; a production system would need to forget
  dormant accounts.

---

## 8. If someone asks you one question

> **"So what does it actually do?"**

*"It scores every payment in under a millisecond, catches 97% of fraud instead
of the 7% a normal bank rule catches, and instead of giving the analyst a risk
score it gives them a sentence — 'this account took money from 14 strangers and
forwarded it within minutes.' It runs on an IBM mainframe because a payment has
a hard deadline, and the mainframe's slowest response was only 15% slower than
its typical one, where my laptop's was four times slower."*
