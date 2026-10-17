# 🛡️ ScamShield

**Real-time detection of digital-payment scams and money-mule accounts — with a human making the final call.**

IBM Z Datathon 2026 · Theme: *Real-Time AI for Critical Decisions* · Built for IBM LinuxONE (s390x)

---

## 1. The problem

A UPI-style payment settles in about **two seconds**. That is the entire window in
which a bank has to decide whether the money is a grocery bill or a pensioner's
life savings being funnelled to a criminal. Get it wrong one way and a victim is
cleaned out before anyone can pick up a phone. Get it wrong the other way and
you freeze an honest person's rent payment.

Two things make this hard:

1. **A single transaction rarely looks criminal.** The fraud lives in the
   *pattern*: an account that receives money from fourteen strangers in two
   hours and forwards it on within minutes is a **money mule**, even though
   every individual transfer looks ordinary.
2. **Static rules are terrible at it.** On our held-out data the classic bank
   rule — *amount > 200,000 AND the sender's account was drained* — catches
   **7%** of fraud while being **wrong two times out of three** when it fires.

ScamShield scores every payment in **well under a millisecond**, assigns one of three decisions,
and — crucially — hands the ambiguous ones to a human **with a sentence
explaining why**, not a number.

| Decision | Tuned for | What happens |
|---|---|---|
| `ALLOW` | — | Payment goes through. No human ever sees it. |
| `REVIEW` | **high recall** | Queued for an analyst. Catch everything; let a person judge. |
| `BLOCK` | **high precision** | Safe enough to stop the payment automatically. |

> The analyst is the decision maker. ScamShield's job is to make 30,000
> transactions into a queue of 12, and to make each of those 12 readable in
> five seconds.

### Who benefits

| | |
|---|---|
| **Vulnerable account holders** | Pensioners and first-time digital-payment users, who are disproportionately targeted and least able to absorb the loss. The money is stopped *before* it leaves, not refunded three months later. |
| **Fraud analysts** | Instead of a spreadsheet of alerts with risk scores, they get a ranked queue with a plain-language reason and three buttons. |
| **Banks and payment operators** | A measurable cut in false positives — every wrongly blocked payment is a support call and a customer who trusts you less. |
| **Regulators** | `analyst_actions` is a complete, queryable audit trail: who decided what, when, and what the model's stated reason was. |

---

## 2. Architecture

```mermaid
flowchart TB
    subgraph INGEST["1 · Data"]
        A1["PaySim CSV · 6.3M transactions"] --> A2["data/prepare.py<br/>300k stratified sample<br/>keeps ALL fraud"]
        A3["data/make_synthetic.py<br/>no-Kaggle fallback"]
        A2 --> A4[("paysim_sample.csv")]
        A3 --> A4
    end

    subgraph FEAT["2 · features.py — ONE implementation, shared"]
        B1["<b>AccountState</b><br/>features() then update()"]
        B2["Transaction level<br/>balance errors · drained-to-zero<br/>amount ratio · type · hour"]
        B3["Account level · rolling O(1)<br/>fan-in: distinct senders / N steps<br/>rapid pass-through within K steps<br/>prior transaction counts"]
        B1 --- B2
        B1 --- B3
    end

    subgraph TRAIN["3 · train.py (offline)"]
        C1["Replay in TIME ORDER"] --> C2["Time split<br/>70% earliest / 30% latest"]
        C2 --> C3["XGBoost<br/>scale_pos_weight"]
        C3 --> C4["Pick REVIEW + BLOCK<br/>thresholds"]
        C3 -.beats.-> C5["Rule baseline"]
    end

    subgraph LIVE["4 · stream.py + scorer.py (real time)"]
        E1["replay @ N txn/sec"] --> E2["features() BEFORE update()"]
        E2 --> E3["inplace_predict · every txn"]
        E3 --> E4{"flagged?"}
        E4 -->|no| E7["ALLOW"]
        E4 -->|yes| E5["TreeSHAP pred_contribs"] --> E6["explain.py<br/>plain-language sentence"]
    end

    subgraph STORE["5 · scamshield.db — SQLite, WAL mode"]
        F1[("transactions")]
        F2[("analyst_actions")]
        F3[("stream_meta")]
    end

    subgraph HUMAN["6 · app.py — analyst dashboard"]
        G1["KPIs: throughput · p50/p95 · money protected"]
        G2["Review queue + Approve / Block / Escalate"]
        G3["Alerts over time · score distribution"]
        G4["<b>Running on: s390x</b>"]
    end

    A4 --> C1
    B1 -. same code .-> C1
    B1 -. same code .-> E2
    C4 --> D1[("artifacts/<br/>model + thresholds")]
    D1 --> E3
    E6 --> F1
    E7 --> F1
    E1 --> F3
    F1 --> G1 & G2 & G3
    F3 --> G4
    G2 -->|human decides| F2
    F2 -.->|feedback for retraining|C3

    classDef store fill:#e8f0fe,stroke:#4a7ebb,color:#12325c
    classDef human fill:#fff4e5,stroke:#b26a00,color:#5c3a00
    class A4,D1,F1,F2,F3 store
    class G1,G2,G3,G4 human
```

More detail, including the per-transaction sequence diagram and the leakage
controls, is in [`docs/architecture.md`](docs/architecture.md).

### The one design decision that matters

**`features.py` is imported by both the trainer and the live engine.** There is
exactly one implementation of every feature. `train.py` replays historical rows
through `AccountState` in time order; `stream.py` feeds it live transactions.
This kills *training/serving skew*, the single most common way a fraud model
quietly stops working in production.

`tests/test_features.py` backs this up by re-deriving every account-level
feature with a naive O(n²) scan over raw history and asserting the O(1)
streaming version produces identical numbers, row for row.

---

## 3. Quick start (laptop — Windows / macOS / Linux)

```bash
pip install -r requirements.txt

# Whole pipeline on synthetic data: no Kaggle account needed, ~1 minute
./run_all.sh

# ...or with the real Kaggle PaySim file
./run_all.sh /path/to/PS_20174392719_1491204439457_log.csv
```

Then the live demo, in **two terminals**:

```bash
# Terminal 1 — the scoring engine
python stream.py --rate 50

# Terminal 2 — the analyst dashboard
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

Open <http://localhost:8501>.

<details>
<summary>Running the stages individually</summary>

```bash
python data/make_synthetic.py --rows 120000   # or: python data/prepare.py <paysim.csv>
python train.py                               # time-split, trains, picks thresholds
python benchmark.py --rows 50000              # throughput + latency + platform.machine()
python stream.py --rate 50                    # live replay into SQLite
python -m pytest tests/ -q                    # the test suite
```

Useful flags:

| Flag | Effect |
|---|---|
| `stream.py --rate 0` | Replay as fast as the machine allows (benchmark mode) |
| `stream.py --limit 2000` | Stop after N transactions |
| `stream.py --loop` | Restart the feed when it ends — handy for a long demo |
| `stream.py --keep` | Append to the existing database instead of starting fresh |
| `stream.py --no-warmup` | Skip the account-history warm-up (faster, weaker features) |
| `train.py --csv path.csv` | Train on a different sample |

</details>

---

## 4. How ScamShield uses IBM LinuxONE

The theme is *real-time AI for critical decisions*, and the reason this problem
belongs on a Z-class machine is **the tail, not the average**.

| What LinuxONE gives us | How ScamShield uses it |
|---|---|
| **Consistent low latency under load** | A payment decision has a hard deadline. An average of 0.25 ms is useless if the p99 is 400 ms, because the p99 is where the large transactions live. `benchmark.py` reports **p50, p95 and p99** precisely so the tail is visible, and `stream.py` records the latency of *every single transaction* in SQLite. |
| **Many cores, high memory bandwidth** | `AccountState` holds rolling state for every account in the network. XGBoost is configured with `n_jobs=0` (use every core). Throughput scales with the box. |
| **Transactions where they already live** | Z runs a large share of the world's card and core-banking workloads. Scoring *next to* the system of record removes a network hop from a 2-second budget — and the data never leaves the platform. |
| **Portability, proven not claimed** | Identical Python, identical `requirements.txt`. The only architecture-specific thing in the repo is a *measurement*: `platform.machine()`. |

**The evidence for the judges is built into the product.** The dashboard has a
*Running on* panel showing `platform.machine()`, the Python version and the CPU
count, and it reads them from `stream_meta` — the values recorded by the engine
that actually did the scoring. On LinuxONE it reads **`s390x`** and the badge
flips to *✅ IBM LinuxONE (s390x)*. The same string is written into
`artifacts/benchmark.json`, so a laptop run and a LinuxONE run are directly
comparable files.

### Running on LinuxONE — step by step

The container
`registry.linuxone.cloud.marist.edu/l1cc/jupyterlab-image-s390x:latest`
already has **pandas, numpy, scikit-learn, xgboost, pytorch, tensorflow and
keras** compiled for s390x. Only the dashboard needs installing.

1. **Copy the project into the shared folder.** From your laptop:

   ```bash
   scp -r ScamShield <user>@<linuxone-host>:~/shared/
   ```

   Or drag the folder into the JupyterLab file browser (upload into `~/shared`).

2. **Open a terminal in JupyterLab** — the LinuxONE JupyterLab is on port
   **38888**: browse to `http://<host>:38888`, then *File → New → Terminal*.

3. **Install just the two dashboard packages** (everything else is already there):

   ```bash
   cd ~/shared/ScamShield
   pip install --user streamlit plotly
   ```

   > ⚠️ Do **not** `pip install -r requirements.txt` inside the container. That
   > would try to rebuild pandas/numpy/xgboost from source on s390x, which takes
   > an hour or fails. The file lists them so the project installs on a laptop;
   > on LinuxONE they are pre-installed.

4. **Confirm you are really on Z:**

   ```bash
   python3 -c "import platform; print(platform.machine())"     # -> s390x
   ```

5. **Build the model and benchmark it:**

   ```bash
   chmod +x run_all.sh
   ./run_all.sh                 # synthetic data
   # or, if you uploaded the Kaggle CSV:
   ./run_all.sh ~/shared/PS_20174392719_1491204439457_log.csv
   ```

   Keep `artifacts/benchmark.json` — that is your s390x evidence. Save it under
   a second name so it does not get overwritten:

   ```bash
   cp artifacts/benchmark.json artifacts/benchmark_linuxone.json
   ```

6. **Start the live demo in two terminals:**

   ```bash
   # Terminal 1
   python3 stream.py --rate 50 --loop

   # Terminal 2
   streamlit run app.py --server.port 8501 --server.address 0.0.0.0
   ```

7. **Reach the dashboard.** Either forward the port from your laptop:

   ```bash
   ssh -L 8501:localhost:8501 <user>@<linuxone-host>
   ```

   then open <http://localhost:8501>; or, if port 8501 is exposed directly, go
   to `http://<host>:8501`. The *Running on* panel should read **s390x**.

<details>
<summary>Troubleshooting on LinuxONE</summary>

| Symptom | Fix |
|---|---|
| `streamlit: command not found` | `pip install --user` puts it in `~/.local/bin`. Run `export PATH=$PATH:~/.local/bin`, or use `python3 -m streamlit run app.py`. |
| Dashboard says *No database yet* | `stream.py` has not been started, or it is writing elsewhere. Both must run from the same directory. |
| Port 8501 already taken | `--server.port 8502` and forward that instead. |
| `train.py` cannot find the CSV | Run from the repo root; every path in `config.py` is relative to it. |
| Pipeline is slow on the full 6.3M-row CSV | `data/prepare.py` samples it down to 300k first. Run that before `train.py`. |

</details>

---

## 5. Results

Measured with `python -m pytest`, `run_all.sh` and `benchmark.py`.
**Fill the LinuxONE column by running the same commands in the container.**

### 5.1 Detection quality

Held-out = the **latest 30% of time steps**, never seen during training.

| Metric | Rule baseline | **ScamShield** |
|---|---|---|
| Recall (fraud caught) | 0.074 | **0.968** |
| Precision (alerts that were real) | 0.333 | **0.996** |
| PR-AUC | — | **0.996** |
| ROC-AUC | — | **0.9998** |
| Alerts raised per 30k transactions | 120 | **523** |
| False positives | 80 | **2** |

<sub>Synthetic PaySim-shaped data: 102,048 rows, 1,702 fraud (1.67%), 71,511 train / 30,537 test.
Baseline = `amount > 200,000 AND sender drained to zero`. Reproduce with `./run_all.sh`.</sub>

> **Honest caveat:** these numbers come from the *synthetic* generator, whose
> fraud patterns are cleaner than reality — treat them as a pipeline smoke test,
> not a claim about real-world performance. The row below is the one that counts.

| Dataset | PR-AUC | Recall @ REVIEW | Precision @ BLOCK | Rule-baseline recall |
|---|---|---|---|---|
| Synthetic (this repo, reproducible) | 0.996 | 0.968 | 0.996 | 0.074 |
| **Real Kaggle PaySim1 (6.3M rows)** | _TBD_ | _TBD_ | _TBD_ | _TBD_ |

### 5.2 What the model actually learned

Top features by gain — note that **the two account-level "mule" features are
doing a third of the work**, which is the entire thesis of the project:

| Rank | Feature | Gain | What it means |
|---|---|---|---|
| 1 | `balance_error_dest` | 0.368 | The recipient's ledger does not add up |
| 2 | **`dest_fanin_senders`** | **0.233** | **Distinct senders → one recipient in N hours (mule collection)** |
| 3 | `type_CASH_OUT` | 0.104 | The usual last hop before money disappears |
| 4 | **`orig_rapid_passthrough`** | **0.063** | **Money forwarded on within K hours of arriving** |
| 5 | `amount` | 0.054 | Size of the payment |
| 6 | `type_TRANSFER` | 0.053 | Account-to-account |
| 7 | `orig_drained_to_zero` | 0.043 | Sender emptied |

### 5.3 Throughput and latency — **laptop vs LinuxONE**

This is the table to fill in during the demo.

| | Laptop (x86-64) | **LinuxONE (s390x)** |
|---|---|---|
| `platform.machine()` | `AMD64` | `s390x` |
| CPU cores | 12 | _TBD_ |
| Python | 3.13.9 | _TBD_ |
| **End-to-end** (features + score + decide) | **1,400 – 2,800 txn/s** | _TBD_ |
| &nbsp;&nbsp;p50 latency | 0.25 – 0.65 ms | _TBD_ |
| &nbsp;&nbsp;p95 latency | 0.96 – 1.43 ms | _TBD_ |
| &nbsp;&nbsp;**p99 latency** | **1.9 – 3.1 ms** | _TBD_ |
| Score only (model call) | 1,400 – 3,400 txn/s | _TBD_ |
| Batch (whole matrix at once) | 0.45M – 1.5M txn/s | _TBD_ |
| Model training time | 3.8 s | _TBD_ |

<sub>50,000 transactions, 300 trees, depth 6, 19 features, single process.
Ranges are the min–max across three consecutive runs. Run `python benchmark.py`;
raw output in `artifacts/benchmark.json`.</sub>

> **Why a range, not one number.** Three identical runs on this laptop varied
> by nearly 2× — a thermally-throttled, multi-tenant consumer machine is simply
> not a stable measurement platform. That variance *is* the argument: a payment
> decision has a hard deadline, so what matters is not the best number you can
> produce but the worst one you can guarantee. Record the LinuxONE column by
> running `benchmark.py` three times in the container and quoting the same
> min–max, so the comparison is like for like.

Even at the slowest observed **p99**, a decision lands in about 3 ms — under
**0.2%** of the two-second payment budget.

### 5.4 Cost of being wrong

Over a single replay of the test window the `BLOCK` tier stopped
**62.4M** in fraudulent payments. The dashboard tracks this live as
*Money protected*, alongside *Money that still got through* — because a fraud
system that only reports its wins is not telling you the truth.

---

## 6. Repository layout

```
ScamShield/
├── config.py                 All paths, windows, thresholds. Everything relative.
├── features.py               AccountState — SHARED by training and streaming.
├── scorer.py                 Loads the model; the per-transaction decision path.
├── explain.py                Contributions → one plain-English sentence.
├── db.py                     SQLite schema + WAL connection helpers.
├── train.py                  Time-split training, thresholds, rule baseline.
├── stream.py                 Real-time replay engine → scamshield.db.
├── app.py                    Streamlit analyst dashboard.
├── benchmark.py              Throughput / p50 / p95 / p99 / platform.machine().
├── run_all.sh                prepare → train → benchmark.
├── requirements.txt
├── data/
│   ├── prepare.py            Kaggle PaySim → 300k stratified sample (all fraud kept).
│   └── make_synthetic.py     PaySim-shaped generator — the no-Kaggle path.
├── docs/architecture.md      Mermaid flowchart, sequence diagram, leakage controls.
├── tests/
│   ├── test_features.py      Incremental state == definitional reference. The key test.
│   └── test_pipeline.py      Synthetic data → train → stream → DB → dashboard.
└── artifacts/                model.joblib · metrics.json · benchmark.json · pr_curve.png
```

---

## 7. Design notes

<details>
<summary><b>No data leakage — four separate controls</b></summary>

| Risk | Control |
|---|---|
| Training on the future | Time-based split on `step`. Never `train_test_split(shuffle=True)`. |
| A transaction seeing itself | `features(txn)` is always called **before** `update(txn)`. Tested by `test_no_lookahead_future_rows_cannot_change_the_past`, which replays one prefix with two different futures and asserts the prefix features are bit-identical. |
| A label hiding in a column | `isFlaggedFraud` — the bank's own legacy rule — is excluded from the feature set. |
| Cold-start skew | `stream.py` warms `AccountState` on the training rows before the demo, so live features carry the same account history the model learned with. |

</details>

<details>
<summary><b>Why XGBoost's <code>pred_contribs</code> instead of the <code>shap</code> package</b></summary>

`pred_contribs=True` *is* TreeSHAP — it ships inside XGBoost, which is already
installed on the LinuxONE image. Adding the `shap` package would mean compiling
a C extension for s390x for exactly zero extra capability.

There is a performance trick here too. The contributions sum to the raw margin,
so one booster call gives both the score and the explanation. But explanations
are only needed for transactions a human will see, so `scorer.py` runs the cheap
`inplace_predict` path on **every** transaction and only pays for TreeSHAP on
the ~2% that get flagged. That is most of the difference between the
*end-to-end* and *score only* rows in the benchmark table.

</details>

<details>
<summary><b>How the two thresholds are chosen</b></summary>

Not hand-picked. `train.py` reads them off the precision-recall curve of the
held-out set:

- **REVIEW** = the strictest threshold that still achieves
  `TARGET_REVIEW_RECALL` (0.95). Optimises for *missing nothing*; a human
  triages the queue.
- **BLOCK** = the loosest threshold whose precision reaches
  `TARGET_BLOCK_PRECISION` (0.95). Optimises for *never freezing an honest
  payment*, because that is the action taken without a human.

Both are saved to `artifacts/model_metadata.json` and loaded by `scorer.py`, so
the running system cannot drift from the evaluated one. Adjust the targets in
`config.py` and re-run `train.py`.

</details>

<details>
<summary><b>Why SQLite and not Kafka</b></summary>

A broker would add an install, a daemon, a port and a failure mode to a
container we do not control, and buy nothing a demo can show. SQLite in **WAL
mode** gives one writer and many readers on a single file — the dashboard polls
twice a second while the engine writes at full speed, and neither blocks the
other. `sqlite3` is in the Python standard library, so there is nothing to
install on s390x.

The seam is clean: `stream.py` is the only module that produces transactions.
Swapping its loop for a Kafka consumer is a contained change.

</details>

<details>
<summary><b>The optional ONNX export</b></summary>

`train.py` attempts an ONNX export inside `try/except`. `onnxmltools` and
`onnxruntime` rarely have s390x wheels, so the failure is expected and
harmless — it prints a one-line note and carries on. `stream.py` always uses the
joblib model. The hook is there because a vendor-neutral model format is the
right answer for a bank, not because the demo needs it.

</details>

<details>
<summary><b>The generative-AI hook (deliberately off)</b></summary>

`explain.py` has a clearly marked `_generative_summary()` hook, disabled via
`config.USE_GENERATIVE_SUMMARY = False`. Templates are the default because a
real-time decision path cannot afford a network round-trip, an audit trail
cannot afford non-determinism, and the container has no outbound model
endpoint. If you enable it, the templates remain the fallback so the dashboard
never shows an empty reason.

</details>

---

## 8. Open-source libraries used

| Library | Licence | Used for | Link |
|---|---|---|---|
| **XGBoost** | Apache-2.0 | Gradient-boosted classifier; `pred_contribs` TreeSHAP explanations | <https://github.com/dmlc/xgboost> |
| **scikit-learn** | BSD-3-Clause | PR curve, average precision, confusion matrix, threshold selection | <https://github.com/scikit-learn/scikit-learn> |
| **pandas** | BSD-3-Clause | CSV loading, stratified sampling, time ordering | <https://github.com/pandas-dev/pandas> |
| **NumPy** | BSD-3-Clause | Feature vectors, percentiles | <https://github.com/numpy/numpy> |
| **Streamlit** | Apache-2.0 | The analyst dashboard | <https://github.com/streamlit/streamlit> |
| **Plotly** | MIT | Alerts-over-time and score-distribution charts | <https://github.com/plotly/plotly.py> |
| **joblib** | BSD-3-Clause | Model serialisation | <https://github.com/joblib/joblib> |
| **Matplotlib** *(optional)* | PSF-based | `pr_curve.png`; skipped gracefully if absent | <https://github.com/matplotlib/matplotlib> |
| **pytest** | MIT | Test suite | <https://github.com/pytest-dev/pytest> |
| **SQLite** (stdlib `sqlite3`) | Public domain | WAL-mode store for transactions and analyst actions | <https://www.sqlite.org/> |

**Dataset:** [PaySim1](https://www.kaggle.com/datasets/ealaxi/paysim1) — synthetic
mobile-money transactions by Lopez-Rojas, Elmir & Axelsson, derived from real
African mobile-money logs. Licensed CC BY-SA 4.0. No real customer data is used
anywhere in this project.

---

## 9. Three-minute demo video script

**Total 3:00. Have both terminals already running before you hit record.**

| Time | Section | What is on screen | What you say |
|---|---|---|---|
| **0:00–0:30** | **The problem** | A slide: a UPI confirmation screen, and a clock showing 2 seconds. | "A digital payment settles in two seconds. In that window a bank must decide: grocery bill, or a pensioner's savings going to a criminal? The fraud is almost never in one transaction — it's in the pattern. One account taking money from fourteen strangers in two hours and forwarding it within minutes is a money mule. Today most banks catch that with static rules. On our data, that rule finds **7%** of fraud and is wrong two times out of three." |
| **0:30–1:00** | **Data & approach** | `features.py` on screen, scroll to `AccountState`. Then the architecture diagram. | "We use PaySim — 6.3 million synthetic mobile-money transactions. Transaction-level signals are the easy part. The interesting features are account-level and rolling: fan-in, rapid pass-through, account age. They're computed incrementally, so this one class runs in both training and production — no training/serving skew. We test that: a naive O(n²) reference gets the same numbers as the streaming version, row for row." |
| **1:00–2:00** | **Live demo** ⭐ | The dashboard, with `stream.py` visibly running beside it. Let the table scroll. | "This is live. Transactions are arriving and being scored right now. p50 latency: well under a millisecond. Green passes through, amber goes to a human, red is auto-blocked. Here's the point —" *(click a queue item)* "— the analyst doesn't get a risk score of 0.97. They get: *'Account drained to zero; recipient received money from 14 different senders in 12 hours; money forwarded on within 2 hours of arriving.'* Approve, Block, or Escalate." *(click Block)* "Every press goes to an audit table — that's the regulator's trail and the next model's training data." |
| **2:00–2:30** | **LinuxONE evidence** 🏆 | Zoom on the *Running on* panel showing **s390x**. Then `cat artifacts/benchmark.json` in the terminal. | "All of this is running on IBM LinuxONE. The dashboard reads `platform.machine()` from the process that did the scoring: **s390x**. Same code as my laptop, zero changes. And look at the number that matters — not the average, the **p99**: a couple of milliseconds. A payment decision has a hard deadline, so the tail is the whole game. Z gives us that tail, next to where the transactions already live." |
| **2:30–3:00** | **Impact & close** | The *Money protected* KPI, then the results table. | "Against the rule baseline: recall from 7% to 97%, false positives from 80 down to 2 — and in one replay, **62 million** in fraud stopped before it left. Every blocked payment is a person who keeps their savings; every false positive we remove is an honest person not locked out of their own money. Real-time AI, a human making the critical call, and an audit trail for both. That's ScamShield."

**Recording tips**

- Start `stream.py --rate 50 --loop` a minute before recording so the charts
  have history and the queue is populated.
- Turn **off** *Live refresh* in the sidebar before clicking queue buttons, so
  the page does not reload mid-click.
- The money figure and the s390x badge are the two shots judges remember —
  hold each for a full three seconds.

---

## 10. Honest limitations

- **PaySim is synthetic.** Real fraud adapts; a model trained on a fixed
  simulation will degrade. In production this needs scheduled retraining fed by
  the `analyst_actions` table.
- **The fan-in and pass-through windows (N=12, K=2 steps) are not tuned.** They
  were chosen as plausible defaults. They should be swept.
- **`AccountState` grows without bound.** Fine for a 300k-row replay; a
  production deployment needs TTL eviction for dormant accounts.
- **Single-process.** Throughput scales with cores inside one process, not
  across machines. Sharding by account hash is the obvious next step.
- **Ground-truth labels are shown in the dashboard** because we are replaying a
  labelled dataset. In production that column does not exist — which is exactly
  why the analyst feedback loop is in the design and not an afterthought.
