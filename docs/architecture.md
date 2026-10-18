# ScamShield - architecture

## 1. End-to-end flow

```mermaid
flowchart TB
    subgraph INGEST["1 · Data"]
        A1["PaySim CSV<br/>6.3M mobile-money transactions"]
        A2["data/prepare.py<br/>stratified 300k sample, keeps ALL fraud"]
        A3["data/make_synthetic.py<br/>PaySim-shaped generator (no Kaggle needed)"]
        A1 --> A2
        A2 --> A4[("data/paysim_sample.csv")]
        A3 --> A4
    end

    subgraph FEAT["2 · Shared feature engine — features.py"]
        B1["AccountState<br/>update() + features()"]
        B2["Transaction level<br/>amount · log amount · type one-hot<br/>balance errors · drained-to-zero<br/>amount/balance ratio · hour of day"]
        B3["Account level (rolling, O(1))<br/>fan-in: distinct senders in last N steps<br/>rapid pass-through within K steps<br/>prior transaction counts"]
        B4["Mule rings (union-find)<br/>cluster size + money through it<br/>edges ONLY on pass-through hops"]
        B5["Velocity / structuring<br/>payments + value in last W steps<br/>amount parked under a round threshold"]
        B6["Victim side (out of character)<br/>amount vs this sender's own mean/max<br/>first ever night-time payment"]
        B1 --- B2
        B1 --- B3
        B1 --- B4
        B1 --- B5
        B1 --- B6
    end

    subgraph TRAIN["3 · Offline — train.py"]
        C1["Replay rows in TIME ORDER<br/>through AccountState"]
        C2["Time-based split<br/>earliest 70% train / latest 30% test"]
        C3["XGBoost + scale_pos_weight"]
        C4["Pick REVIEW (best recall within<br/>analyst capacity) and BLOCK<br/>(high precision)"]
        C5["Rule baseline + cost model<br/>amount > 200k AND drained<br/>priced in currency"]
        C6["Record drift reference<br/>score + feature histograms"]
        C1 --> C2 --> C3 --> C4 --> C6
        C3 -.compare.-> C5
    end

    subgraph ART["4 · artifacts/"]
        D1[("model.joblib")]
        D2[("model_metadata.json<br/>features + thresholds")]
        D3[("metrics.json · pr_curve.png<br/>cost analysis + capacity")]
        D4[("benchmark.json<br/>platform.machine()")]
        D5[("drift_reference.json")]
        D6[("ablation.json<br/>per feature group<br/>per scam typology")]
    end

    subgraph LIVE["5 · Real time — stream.py + scorer.py"]
        E1["Replay held-out rows<br/>at N txn/sec"]
        E2["features() BEFORE update()<br/>→ no leakage"]
        E3["inplace_predict<br/>fast path, every txn"]
        E4{"score ≥ threshold?"}
        E5["TreeSHAP pred_contribs<br/>only for flagged txns"]
        E6["explain.py<br/>plain-language sentence"]
        E1 --> E2 --> E3 --> E4
        E4 -->|ALLOW| E7["pass straight through"]
        E4 -->|REVIEW / BLOCK| E5 --> E6
    end

    subgraph STORE["6 · scamshield.db (SQLite, WAL)"]
        F1[("transactions<br/>score · decision · reasons · latency")]
        F2[("analyst_actions<br/>APPROVE / BLOCK / ESCALATE")]
        F3[("stream_meta<br/>platform · thresholds · status")]
    end

    subgraph HUMAN["7 · Analyst — app.py (Streamlit)"]
        G1["KPI cards<br/>throughput · p50/p95 · money protected"]
        G2["Review queue<br/>amount · risk · WHY · 3 buttons"]
        G3["Charts<br/>alerts over time · score distribution"]
        G4["Running on panel<br/>platform.machine() = s390x"]
    end

    A4 --> C1
    B1 -. same code .-> C1
    B1 -. same code .-> E2
    C4 --> D1 & D2 & D3
    D1 & D2 --> E3
    E6 --> F1
    E7 --> F1
    E1 --> F3
    F1 --> G1 & G2 & G3
    F3 --> G4
    G2 -->|analyst decides| F2
    F2 -.->|feedback for the next model| C3

    classDef store fill:#e8f0fe,stroke:#4a7ebb,color:#12325c
    classDef human fill:#fff4e5,stroke:#b26a00,color:#5c3a00
    class D1,D2,D3,D4,F1,F2,F3,A4 store
    class G1,G2,G3,G4 human
```

## 2. The critical decision path (per transaction)

```mermaid
sequenceDiagram
    autonumber
    participant S as Payment stream
    participant A as AccountState
    participant M as XGBoost
    participant X as explain.py
    participant D as SQLite (WAL)
    participant H as Analyst

    S->>A: transaction arrives
    A->>A: features() — reads ONLY past events
    A-->>M: 28-dim vector
    alt circuit breaker CLOSED
        M-->>M: inplace_predict → risk score
        Note over M: ~0.2 ms, every transaction
        alt score < REVIEW threshold
            M-->>D: ALLOW (no explanation computed)
        else flagged
            M->>X: TreeSHAP pred_contribs
            Note over X: paid for on ~2% of traffic only
            X-->>D: top-3 reasons + one-sentence summary
        end
    else breaker OPEN (model unhealthy)
        M-->>D: fallback rule → ALLOW or REVIEW, tagged DEGRADED
        Note over M: never BLOCK without a calibrated score
    end
    A->>A: update() — now the txn becomes history
    D-->>H: dashboard refreshes (~2 s)
    H->>D: APPROVE / BLOCK / ESCALATE → analyst_actions
```

## 3. Why the feature code is shared

The most common way a fraud model dies in production is **training/serving
skew**: features computed one way in a notebook and another way in the live
system. ScamShield makes that structurally impossible.

```mermaid
flowchart LR
    F["features.py<br/><b>AccountState</b>"]
    T["train.py<br/>batch replay"]
    S["stream.py<br/>live scoring"]
    B["benchmark.py<br/>throughput"]
    F --> T
    F --> S
    F --> B
    TEST["tests/test_features.py<br/>incremental == definitional reference"]
    TEST -.verifies.-> F
```

There is exactly one implementation. `train.py` replays historical rows
through it in time order; `stream.py` feeds it live transactions. The test
suite re-derives every account-level feature with a naive O(n²) scan over raw
history and asserts the O(1) streaming version produces identical numbers.

## 4. Leakage controls

| Risk | Control |
|---|---|
| Future data in training | Time-based split; train on earliest 70% of steps, test on the rest. Never `train_test_split(shuffle=True)`. |
| A transaction seeing itself | `features(txn)` is called **before** `update(txn)`, always. Enforced by `AccountState.process()` and tested by `test_no_lookahead_future_rows_cannot_change_the_past`. |
| Label leaking through a column | `isFlaggedFraud` (the bank's own legacy rule) is excluded from the feature set. |
| Cold-start mismatch | `stream.py` warms `AccountState` on the training rows first, so live features have the same account history the model trained with. |

## 5. Runtime shape on IBM LinuxONE

```mermaid
flowchart TB
    subgraph Z["IBM LinuxONE · s390x · one container"]
        direction TB
        P1["Terminal 1<br/>python stream.py --rate 50"]
        P2["Terminal 2<br/>streamlit run app.py"]
        DB[("scamshield.db<br/>SQLite in WAL mode")]
        P1 -->|writes| DB
        DB -->|reads, never blocked| P2
    end
    J["Browser<br/>:8501"] --- P2
```

Two processes, one file, no message broker and no database server. WAL mode
is what makes the single file safe for a concurrent writer and reader — which
is why the dashboard stays live while the engine is at full throughput.

## 6. Resilience: the state machine behind every decision

A payment switch must answer every transaction inside its deadline, including
while the model is misbehaving. `scorer.py` therefore wraps scoring in a
three-state circuit breaker, and no transaction can leave `process()` without
a decision.

```mermaid
stateDiagram-v2
    [*] --> Closed
    Closed --> Open: N consecutive failures<br/>(BREAKER_FAILURE_THRESHOLD)
    Open --> HalfOpen: cooldown elapsed<br/>(BREAKER_COOLDOWN_SECONDS)
    HalfOpen --> Closed: probe succeeds
    HalfOpen --> Open: probe fails<br/>(timer re-armed)

    note right of Closed
        Model scores every txn.
        decision_mode = MODEL
    end note
    note right of Open
        Model not called at all.
        Fallback rule decides:
        ALLOW or REVIEW, never BLOCK.
        decision_mode = DEGRADED
    end note
```

Two details that are easy to get wrong, and both are pinned by tests in
`tests/test_resilience.py`:

- **A failing probe must re-arm the timer.** Leaving the original `opened_at`
  in place keeps the breaker permanently half-open, so every single
  transaction goes on hitting a model already known to be broken. This was a
  real bug, caught by `test_breaker_reopens_if_the_probe_fails_again`.
- **Degraded mode never blocks.** Without a calibrated probability, stopping a
  payment on a crude rule punishes honest customers exactly when the decision
  is least defensible. `test_fallback_never_auto_blocks` fails the build if
  that property is lost.

## 7. Knowing the model has gone stale

Fraud detection is adversarial: the criminals rewrite the input distribution,
and production has no labels to tell you about it — chargebacks arrive months
later. The only label-free signal available is whether the data still looks
like the training data.

```mermaid
flowchart LR
    T["train.py<br/>held-out window"] -->|quantile histograms| R[("drift_reference.json<br/>score + top features")]
    L["live stream<br/>decision_mode = MODEL only"] --> P["monitor.psi()"]
    R --> P
    P --> B{"PSI band"}
    B -->|"< 0.10"| G["STABLE"]
    B -->|"0.10 – 0.25"| Y["MODERATE<br/>keep watching"]
    B -->|"> 0.25"| X["SIGNIFICANT<br/>retrain"]
```

Quantile bin edges, not equal-width ones: fraud scores are extremely skewed,
and equal-width bins would put 99.9% of the mass in the first bin and detect
nothing.

Degraded-mode rows are excluded from the comparison. The fallback rule emits
fixed pseudo-scores, so including a two-second outage registered as
"significant drift" when the model had simply been offline.
