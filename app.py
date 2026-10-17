"""
ScamShield - analyst dashboard.

    streamlit run app.py --server.port 8501 --server.address 0.0.0.0

Reads the SQLite database that stream.py is writing to (WAL mode, so reading
never blocks the writer) and refreshes itself every couple of seconds.

The human is the decision maker here.  ScamShield narrows millions of payments
down to a handful, explains each one in a sentence, and the analyst presses
Approve / Block / Escalate.  Every press is written to analyst_actions, which
is both the audit trail and the training data for the next model.
"""
from __future__ import annotations

import json
import os
import platform
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
import db

# ----------------------------------------------------------------------
# Page setup
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="ScamShield - real-time scam & mule detection",
    page_icon="🛡️",
    layout="wide",
)

DECISION_COLOURS = {
    "ALLOW": "#1b7f4d",    # green
    "REVIEW": "#b26a00",   # amber
    "BLOCK": "#b3261e",    # red
}
DECISION_BG = {
    "ALLOW": "rgba(27,127,77,0.12)",
    "REVIEW": "rgba(178,106,0,0.16)",
    "BLOCK": "rgba(179,38,30,0.16)",
}
ACTION_LABELS = {"APPROVE": "Approve", "BLOCK": "Block", "ESCALATE": "Escalate"}

# Set SCAMSHIELD_NO_AUTOREFRESH=1 to render the page exactly once.  The test
# suite needs this (an endless sleep/rerun loop never returns), and it is handy
# if you want to take a screenshot without the page moving under you.
AUTO_REFRESH = os.environ.get("SCAMSHIELD_NO_AUTOREFRESH", "") != "1"

st.markdown(
    """
    <style>
      .block-container {padding-top: 2rem; padding-bottom: 2rem;}
      div[data-testid="stMetricValue"] {font-size: 1.6rem;}
      .ss-alert {border-left: 5px solid #b3261e; padding: 0.6rem 0.9rem;
                 border-radius: 6px; margin-bottom: 0.4rem;}
      .ss-review {border-left-color: #b26a00;}
      .ss-reason {font-size: 0.92rem; opacity: 0.92;}
      .ss-meta {font-size: 0.8rem; opacity: 0.7;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------
# Data access  (cached for a fraction of the refresh interval so the
# several widgets on the page share a single read)
# ----------------------------------------------------------------------
@st.cache_resource
def get_conn(db_path: str) -> sqlite3.Connection:
    return db.connect(db_path)


def read_sql(conn, sql: str, params: tuple = ()) -> pd.DataFrame:
    try:
        return pd.read_sql_query(sql, conn, params=params)
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=1.0, show_spinner=False)
def load_overview(db_path: str) -> dict:
    conn = get_conn(db_path)
    row = read_sql(conn, """
        SELECT COUNT(*)                                            AS n,
               SUM(decision != 'ALLOW')                            AS alerts,
               SUM(decision = 'BLOCK')                             AS blocks,
               SUM(decision = 'REVIEW')                            AS reviews,
               SUM(CASE WHEN decision='BLOCK' AND is_fraud=1 THEN amount ELSE 0 END)
                                                                   AS money_protected,
               SUM(CASE WHEN decision='ALLOW' AND is_fraud=1 THEN amount ELSE 0 END)
                                                                   AS money_missed,
               SUM(is_fraud)                                       AS fraud_total,
               SUM(CASE WHEN decision != 'ALLOW' AND is_fraud = 1 THEN 1 ELSE 0 END)
                                                                   AS fraud_caught,
               MIN(created_at)                                     AS first_at,
               MAX(created_at)                                     AS last_at
        FROM transactions
    """)
    out = row.iloc[0].to_dict() if len(row) else {}
    lat = read_sql(conn, "SELECT latency_ms FROM transactions ORDER BY id DESC LIMIT 20000")
    if len(lat):
        out["p50"] = float(lat["latency_ms"].quantile(0.50))
        out["p95"] = float(lat["latency_ms"].quantile(0.95))
        out["p99"] = float(lat["latency_ms"].quantile(0.99))
    else:
        out["p50"] = out["p95"] = out["p99"] = 0.0
    out["meta"] = db.get_meta(conn)
    return out


@st.cache_data(ttl=1.0, show_spinner=False)
def load_recent(db_path: str, limit: int) -> pd.DataFrame:
    conn = get_conn(db_path)
    return read_sql(conn, """
        SELECT id, seq, step, type, amount, name_orig, name_dest, score,
               decision, summary, latency_ms, is_fraud, created_at
        FROM transactions ORDER BY id DESC LIMIT ?
    """, (limit,))


@st.cache_data(ttl=1.0, show_spinner=False)
def load_queue(db_path: str, limit: int, include_actioned: bool) -> pd.DataFrame:
    conn = get_conn(db_path)
    where = ["t.decision IN ('REVIEW','BLOCK')"]
    if not include_actioned:
        where.append("a.id IS NULL")       # only items nobody has decided yet
    return read_sql(conn, f"""
        SELECT t.id, t.step, t.type, t.amount, t.name_orig, t.name_dest,
               t.score, t.decision, t.summary, t.reasons, t.is_fraud, t.created_at,
               a.action AS analyst_action
        FROM transactions t
        LEFT JOIN analyst_actions a ON a.txn_id = t.id
        WHERE {' AND '.join(where)}
        ORDER BY t.score DESC, t.id DESC
        LIMIT ?
    """, (limit,))


@st.cache_data(ttl=1.0, show_spinner=False)
def load_timeline(db_path: str) -> pd.DataFrame:
    conn = get_conn(db_path)
    return read_sql(conn, """
        SELECT created_at, decision, score FROM transactions
        ORDER BY id DESC LIMIT 20000
    """)


@st.cache_data(ttl=2.0, show_spinner=False)
def load_actions(db_path: str) -> pd.DataFrame:
    conn = get_conn(db_path)
    return read_sql(conn, """
        SELECT a.id, a.txn_id, a.action, a.note, a.created_at,
               t.amount, t.score, t.decision
        FROM analyst_actions a LEFT JOIN transactions t ON t.id = a.txn_id
        ORDER BY a.id DESC LIMIT 200
    """)


def record_action(db_path: str, txn_id: int, action: str, note: str = "") -> None:
    conn = get_conn(db_path)
    conn.execute(
        "INSERT INTO analyst_actions (txn_id, action, note, created_at) VALUES (?,?,?,?)",
        (int(txn_id), action, note,
         datetime.now(timezone.utc).isoformat(timespec="seconds")),
    )
    conn.commit()
    load_queue.clear()
    load_actions.clear()


def full_width(fn, *args, **kwargs):
    """
    Render a Streamlit widget full width, whichever API version is installed.

    Streamlit renamed the old full-width keyword to width="stretch".  The
    LinuxONE container gets whatever "pip install streamlit" hands it, so we
    try the new spelling and fall back to the old one rather than pin a
    version.
    """
    try:
        return fn(*args, width="stretch", **kwargs)
    except TypeError:
        return fn(*args, use_container_width=True, **kwargs)


def money(v: float) -> str:
    v = float(v or 0)
    if abs(v) >= 1_000_000_000:
        return f"{v / 1_000_000_000:.2f}B"
    if abs(v) >= 1_000_000:
        return f"{v / 1_000_000:.2f}M"
    if abs(v) >= 1_000:
        return f"{v / 1_000:.1f}K"
    return f"{v:,.0f}"


# ----------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------
st.sidebar.title("🛡️ ScamShield")
st.sidebar.caption("Real-time AI for critical decisions — IBM Z Datathon 2026")

db_path = st.sidebar.text_input(
    "Database", value=os.environ.get("SCAMSHIELD_DB", str(config.DB_PATH)))
live = st.sidebar.toggle("Live refresh", value=AUTO_REFRESH,
                         help="Pause this while you work through the queue.")
refresh_s = st.sidebar.slider("Refresh every (s)", 1, 10, config.DASHBOARD_REFRESH_SECONDS)
table_rows = st.sidebar.slider("Rows in live table", 10, 200, 40, step=10)
queue_rows = st.sidebar.slider("Items in review queue", 5, 50, 12)
show_actioned = st.sidebar.checkbox("Show already-actioned alerts", value=False)

if not Path(db_path).exists():
    st.title("ScamShield")
    st.warning(
        f"No database at `{db_path}` yet.\n\n"
        "Start the engine in another terminal:\n\n"
        "```\npython stream.py --rate 50\n```"
    )
    st.stop()

# ----------------------------------------------------------------------
# Header + "Running on" evidence panel
# ----------------------------------------------------------------------
overview = load_overview(db_path)
meta = overview.get("meta", {})
machine = meta.get("machine", platform.machine())
is_z = machine.lower().startswith("s390")

left, right = st.columns([3, 2])
with left:
    st.title("ScamShield")
    st.caption("Real-time detection of digital-payment scams and money-mule accounts. "
               "The model narrows the stream down; the analyst makes the call.")
with right:
    badge = "✅ IBM LinuxONE (s390x)" if is_z else f"💻 Dev machine ({machine})"
    st.markdown(f"#### Running on &nbsp; {badge}")
    rc1, rc2, rc3 = st.columns(3)
    rc1.metric("Architecture", machine)
    rc2.metric("Python", meta.get("python", platform.python_version()))
    rc3.metric("CPU cores", meta.get("cpu_count", os.cpu_count()))
    st.caption(f"{meta.get('platform', platform.platform())} · "
               f"engine status: **{meta.get('status', 'unknown')}**")

st.divider()

# ----------------------------------------------------------------------
# KPI cards
# ----------------------------------------------------------------------
n_total = int(overview.get("n") or 0)
if n_total == 0:
    st.info("The database exists but is empty — start `python stream.py` to begin scoring.")
    st.stop()

alerts = int(overview.get("alerts") or 0)
blocks = int(overview.get("blocks") or 0)
reviews = int(overview.get("reviews") or 0)
protected = float(overview.get("money_protected") or 0.0)
missed = float(overview.get("money_missed") or 0.0)

# Throughput from the wall-clock span of what has been written.
rate = 0.0
try:
    t0 = datetime.fromisoformat(overview["first_at"])
    t1 = datetime.fromisoformat(overview["last_at"])
    span = (t1 - t0).total_seconds()
    rate = n_total / span if span > 0 else 0.0
except Exception:
    pass

k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Transactions scored", f"{n_total:,}")
k2.metric("Throughput", f"{rate:,.0f} /s", help="Measured over the whole run so far.")
k3.metric("Latency p50 / p95", f"{overview['p50']:.2f} / {overview['p95']:.2f} ms",
          help="Features + model + decision, per transaction.")
k4.metric("Alerts raised", f"{alerts:,}",
          delta=f"{blocks:,} blocked · {reviews:,} to review", delta_color="off")
k5.metric("Money protected", money(protected),
          help="Total value of fraudulent payments the BLOCK tier stopped.")
k6.metric("Alert rate", f"{100 * alerts / n_total:.2f}%",
          help="Share of all traffic that needed any attention.")

st.divider()

# ----------------------------------------------------------------------
# Review queue — where the human decision happens
# ----------------------------------------------------------------------
queue_col, chart_col = st.columns([3, 2], gap="large")

with queue_col:
    st.subheader("Review queue")
    st.caption("Highest-risk payments first. Every action is written to the "
               "`analyst_actions` audit table.")

    queue = load_queue(db_path, queue_rows, show_actioned)
    if queue.empty:
        st.success("Queue is clear — nothing is waiting for a decision.")
    for _, row in queue.iterrows():
        txn_id = int(row["id"])
        decision = row["decision"]
        css = "ss-alert" + ("" if decision == "BLOCK" else " ss-review")
        actioned = row.get("analyst_action")

        st.markdown(
            f"""<div class="{css}" style="background:{DECISION_BG[decision]}">
            <b>{decision}</b> &nbsp;·&nbsp; <b>{money(row['amount'])}</b>
            &nbsp;·&nbsp; risk <b>{row['score']:.3f}</b>
            &nbsp;·&nbsp; {row['type']}
            <div class="ss-reason">{row['summary'] or 'No summary available.'}</div>
            <div class="ss-meta">#{txn_id} &nbsp; {row['name_orig']} → {row['name_dest']}
            &nbsp; step {int(row['step'])}
            {' &nbsp;·&nbsp; already actioned: <b>' + str(actioned) + '</b>' if actioned else ''}
            </div></div>""",
            unsafe_allow_html=True,
        )

        b1, b2, b3, b4 = st.columns([1, 1, 1, 3])
        if full_width(b1.button, "Approve", key=f"ap{txn_id}"):
            record_action(db_path, txn_id, "APPROVE")
            st.rerun()
        if full_width(b2.button, "Block", key=f"bl{txn_id}", type="primary"):
            record_action(db_path, txn_id, "BLOCK")
            st.rerun()
        if full_width(b3.button, "Escalate", key=f"es{txn_id}"):
            record_action(db_path, txn_id, "ESCALATE")
            st.rerun()
        with b4.expander("Why?"):
            try:
                reasons = json.loads(row["reasons"] or "[]")
            except Exception:
                reasons = []
            if reasons:
                full_width(
                    st.dataframe,
                    pd.DataFrame(reasons).rename(columns={
                        "feature": "Signal", "value": "Value",
                        "contribution": "Pushed risk up by",
                    }),
                    hide_index=True,
                )
            else:
                st.caption("No contribution detail stored for this transaction.")

with chart_col:
    st.subheader("Alerts over time")
    timeline = load_timeline(db_path)
    if not timeline.empty:
        timeline["created_at"] = pd.to_datetime(timeline["created_at"], format="mixed",
                                                errors="coerce", utc=True)
        timeline = timeline.dropna(subset=["created_at"])
        binned = (
            timeline.set_index("created_at")
            .groupby([pd.Grouper(freq="5s"), "decision"])
            .size().reset_index(name="count")
        )
        alerts_only = binned[binned["decision"] != "ALLOW"]
        if alerts_only.empty:
            st.caption("No alerts raised yet in this window.")
        else:
            fig = px.bar(
                alerts_only, x="created_at", y="count", color="decision",
                color_discrete_map=DECISION_COLOURS,
                labels={"created_at": "", "count": "alerts", "decision": ""},
            )
            fig.update_layout(height=230, margin=dict(l=0, r=0, t=10, b=0),
                              barmode="stack", legend=dict(orientation="h", y=1.15))
            full_width(st.plotly_chart, fig)

        st.subheader("Risk score distribution")
        fig2 = go.Figure()
        fig2.add_trace(go.Histogram(
            x=timeline["score"], nbinsx=50, marker_color="#4a7ebb", name="all payments",
        ))
        for thr, name, colour in (
            (float(meta.get("review_threshold", config.DEFAULT_REVIEW_THRESHOLD)),
             "REVIEW", DECISION_COLOURS["REVIEW"]),
            (float(meta.get("block_threshold", config.DEFAULT_BLOCK_THRESHOLD)),
             "BLOCK", DECISION_COLOURS["BLOCK"]),
        ):
            fig2.add_vline(x=thr, line_dash="dash", line_color=colour,
                           annotation_text=name, annotation_position="top")
        fig2.update_yaxes(type="log", title="payments (log)")
        fig2.update_xaxes(title="risk score")
        fig2.update_layout(height=250, margin=dict(l=0, r=0, t=20, b=0), showlegend=False)
        full_width(st.plotly_chart, fig2)

st.divider()

# ----------------------------------------------------------------------
# Live transaction table
# ----------------------------------------------------------------------
st.subheader("Live transaction feed")
recent = load_recent(db_path, table_rows)
if not recent.empty:
    view = recent[["id", "step", "type", "amount", "name_orig", "name_dest",
                   "score", "decision", "latency_ms", "summary"]].copy()
    view.columns = ["#", "Step", "Type", "Amount", "From", "To",
                    "Risk", "Decision", "Latency ms", "Why"]

    def _colour(row: pd.Series):
        bg = DECISION_BG.get(row["Decision"], "")
        return [f"background-color: {bg}"] * len(row)

    full_width(
        st.dataframe,
        view.style.apply(_colour, axis=1).format({
            "Amount": "{:,.0f}", "Risk": "{:.4f}", "Latency ms": "{:.3f}",
        }),
        hide_index=True, height=420,
    )

# ----------------------------------------------------------------------
# Audit trail
# ----------------------------------------------------------------------
with st.expander(f"Analyst audit trail"):
    actions = load_actions(db_path)
    if actions.empty:
        st.caption("No analyst decisions recorded yet.")
    else:
        counts = actions["action"].value_counts()
        cols = st.columns(len(ACTION_LABELS))
        for col, (key, label) in zip(cols, ACTION_LABELS.items()):
            col.metric(label + "d" if key != "ESCALATE" else "Escalated",
                       int(counts.get(key, 0)))
        full_width(st.dataframe, actions, hide_index=True)

# Detection quality on this run (ground truth is available because we are
# replaying a labelled dataset - in production this column simply would not
# exist, and the analyst_actions table becomes the feedback signal instead).
with st.expander("Detection quality on this replay (uses the dataset labels)"):
    fraud_total = int(overview.get("fraud_total") or 0)
    caught = int(overview.get("fraud_caught") or 0)
    c1, c2, c3 = st.columns(3)
    c1.metric("Fraudulent payments in the stream", f"{fraud_total:,}")
    c2.metric("Caught (REVIEW or BLOCK)", f"{caught:,}",
              delta=f"{100 * caught / fraud_total:.1f}% recall" if fraud_total else None,
              delta_color="off")
    c3.metric("Money that still got through", money(missed))

st.caption(
    f"Last refreshed {datetime.now().strftime('%H:%M:%S')} · "
    f"reading `{Path(db_path).name}` in WAL mode while the engine writes to it."
)

# ----------------------------------------------------------------------
# Auto refresh
# ----------------------------------------------------------------------
if live and AUTO_REFRESH:
    time.sleep(refresh_s)
    st.rerun()
