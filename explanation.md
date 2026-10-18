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
the account."* On our test data that rule catches **1.7%** of fraud and is
wrong **93 times out of 100** when it does fire. It annoys honest customers
and misses nearly all the real criminals.

### Five different crimes, not one

ScamShield looks for five separate patterns, because they leave completely
different traces:

| Pattern | What happens |
|---|---|
| **Account takeover** | Someone drains the victim's balance to a mule, who cashes out within the hour |
| **Mule collection** | One account takes money from 8–20 unrelated strangers in a few hours |
| **Layering chain** | The money hops through 3–5 accounts in minutes — a *ring*, not one mule |
| **Structuring** | One big transfer split into several payments, each parked just under the 200,000 rule |
| **Social engineering** | A pensioner is on the phone to a fake "bank official" and sends their savings at 3am to an account they have never paid before |

That last one is the one that matters most for protecting vulnerable people,
and it is the hardest. **Both sides of the ledger balance perfectly.** There
is no arithmetic mistake, no mule history, nothing wrong with the transaction
at all. The only evidence is that it is wildly out of character *for that
particular customer* — 300× their usual payment, at an hour they have never
once transacted before. That is why ScamShield scores the sender's own
history, not just the recipient's.

### And some fraud leaves no trace whatsoever

A stolen card used to buy something ordinary, at an ordinary shop, on a
Tuesday afternoon. Nothing in the data tells you it was not the real
cardholder. The bank finds out weeks later from a chargeback.

Our data generator creates these deliberately. It makes the headline numbers
look worse — and it is the reason they are worth anything. An earlier version
of this project did not include them and reported 97% recall, which was a
pleasant number about a world that does not exist.

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

**The human makes the final call.** ScamShield's job is to turn 11,000
payments into a queue of about 20, and make each one readable in five
seconds.

### Three things it does that most fraud demos skip

**1. It only queues what a human can actually work.**
A fraud team has a fixed number of analysts. If you tune the model to catch
95% of fraud on our data, it flags **a third of all payments** — a queue
nobody could ever get through, so in practice the alerts get ignored and you
catch nothing. So ScamShield caps the queue at **2% of traffic** and asks
"what is the most fraud we can catch inside that?" The answer is 76%. That is
a smaller number than a 95% target, and it is the one that would actually
work on a Monday morning.

**2. It keeps deciding when the model breaks.**
A payment cannot wait for someone to restart a service. If the model starts
failing, a circuit breaker trips, ScamShield falls back to the simple rule,
and every one of those decisions is stamped `DEGRADED` so nobody later
mistakes it for a model decision. It then tests the model again after two
seconds and switches back automatically. Crucially, degraded mode is allowed
to *ask a human* but never to *block a payment* — with no real score, blocking
an honest customer is the one mistake you cannot justify. You can watch this
happen on stage:

```
python stream.py --rate 300 --inject-failure 400:20
```

**3. It notices when it has gone stale.**
Fraud is the one problem where the criminals actively change the inputs. A
model that was excellent in January is quietly mediocre by June, and in the
real world there are no labels to tell you — chargebacks arrive months later.
ScamShield records what "normal" looked like during training and compares the
live traffic against it on every refresh, showing a 🟢/🟡/🔴 light. Green
means today's payments still look like the ones the model was tested on.

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
| Fraud caught | 1.7% | **76%** |
| False alarms | 77 | **1** |
| Accuracy when it fires | 7% | **99.6%** |

76% rather than 97%, because roughly a fifth of the fraud in this data is the
untraceable kind described above. Nothing could catch it from the payment
alone.

### In money, which is what a bank actually asks

Precision and recall cannot go in a budget. So we priced every outcome — 150
for an analyst to review an alert, 2,500 for wrongly blocking an honest
customer, and the full amount for fraud that gets through:

| | Cost per 1,000 payments |
|---|---|
| No fraud system at all | 3,228,396 |
| The bank's current rule | 3,087,999 |
| **ScamShield** | **176,866** |

**About 94% cheaper than the rule banks run today.** (Those prices are our
assumptions, set in one config file — change them and the answer changes. The
point is to make the trade-off visible, not to claim one number.)

### The uncomfortable finding we are reporting anyway

I added nine new features for mule rings, payment bursts and victim-side
behaviour. Then I wrote `ablation.py` to retrain the model with each group
removed and measure whether they actually helped.

**They did not — about +0.000 recall.**

Two reasons, and both are worth knowing:

- The queue size is fixed at 2%, so a better model mostly changes *which*
  fraud gets caught, not how much. Per-pattern detection moved by up to 8
  points while the total stayed flat.
- Our synthetic data is too easy. Measured against legitimate traffic alone,
  the *original* features already identify mule collection, layering and
  structuring almost perfectly. There is nothing left for new features to
  find. Real data has ambiguity that a generator cannot fake.

The features stay, because they are correct, verified against an independent
reference implementation, cost no measurable speed, and tell the analyst
useful things ("this recipient sits in a 238-account network"). But the
project does not claim they make it more accurate. Proving that needs real
data, and that is the honest next step.

Most hackathon projects would not have measured this. Measuring it, and
reporting the answer when it was not the answer I wanted, is the part worth
pointing at.

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

> ⚠️ The LinuxONE speed figures above were measured with the earlier version
> of the model (19 features, easier data). The *hardware* conclusion — Z's
> flat latency tail — is unaffected, and the laptop benchmark shows the
> extra features cost nothing measurable. But the detection numbers from
> that Z run are out of date. Re-run `./run_all.sh` on the server to refresh
> them before presenting those specific rows.

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

- The data is **synthetic**. The patterns are cleaner than real life. Running
  it on the real Kaggle PaySim file is the single most valuable next step.
- **The nine new features do not measurably improve detection here** (see
  above). They are kept for correctness, analyst context and real-world
  typology coverage, not on a claim of better accuracy.
- **One shared queue starves the quieter scams.** Structuring and victim scams
  are detectable, but mule patterns outrank them for the same 2% of analyst
  capacity, so they rarely reach a human. The fix is a separate queue per scam
  type, which is designed but not built.
- The drift light watches the *risk scores*, not each feature individually. It
  catches the common case; two features shifting in opposite directions could
  hide from it.
- The mule-ring graph **never forgets**. Clusters only grow, so on a long run
  most mules end up in one big network. Realistic for a laundering ring, but a
  production version needs a time-windowed graph.
- The dashboard shows whether each payment was *really* fraud, because we're
  replaying a dataset where we know the answer. In the real world that column
  doesn't exist — which is exactly why the analyst feedback loop matters.
- The account memory grows forever; a production system would need to forget
  dormant accounts.

---

## 8. If someone asks you one question

> **"So what does it actually do?"**

*"It scores every payment in under a millisecond and catches 76% of the fraud
inside a queue of just 2% of traffic — the amount a real analyst team can
actually work — against 1.7% for the rule banks run today. Instead of a risk
score it hands the analyst a sentence: 'this account took money from 14
strangers and forwarded it within minutes.' It keeps deciding even when the
model fails, it tells you when it has gone stale, and it runs on an IBM
mainframe because a payment has a hard deadline — the mainframe's slowest
response was 15% slower than its typical one, where my laptop's was four times
slower."*

And if they ask what you are least happy with:

*"I added nine features for mule rings and victim behaviour, then measured
whether they helped — and they didn't, because my synthetic data was already
too easy for the original features. I left the measurement in the repo rather
than quietly dropping it."*
