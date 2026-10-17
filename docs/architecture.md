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
        B1 --- B2
        B1 --- B3
    end

    subgraph TRAIN["3 · Offline — train.py"]
        C1["Replay rows in TIME ORDER<br/>through AccountState"]
        C2["Time-based split<br/>earliest 70% train / latest 30% test"]
        C3["XGBoost + scale_pos_weight"]
        C4["Pick REVIEW (high recall)<br/>and BLOCK (high precision)"]
        C5["Rule baseline<br/>amount > 200k AND drained"]
        C1 --> C2 --> C3 --> C4
        C3 -.compare.-> C5
    end

    subgraph ART["4 · artifacts/"]
        D1[("model.joblib")]
        D2[("model_metadata.json<br/>features + thresholds")]
        D3[("metrics.json · pr_curve.png")]
        D4[("benchmark.json<br/>platform.machine()")]
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
    A-->>M: 19-dim vector
    M-->>M: inplace_predict → risk score
    Note over M: ~0.2 ms, every transaction
    alt score < REVIEW threshold
        M-->>D: ALLOW (no explanation computed)
    else flagged
        M->>X: TreeSHAP pred_contribs
        Note over X: paid for on ~2% of traffic only
        X-->>D: top-3 reasons + one-sentence summary
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
