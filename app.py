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
import monitor

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
ACTION_LABELS = {"APPROVE": "Approved", "BLOCK": "Blocked", "ESCALATE": "Escalated"}

# An alert older than this is marked as breaching its service level.  A
# payment settles in seconds, so a queue item that ages is a queue item that
# arrived too late to act on.
QUEUE_SLA_SECONDS = 60

# Set SCAMSHIELD_NO_AUTOREFRESH=1 to render the page exactly once.  The test
# suite needs this (an endless sleep/rerun loop never returns), and it is handy
# if you want to take a screenshot without the page moving under you.
AUTO_REFRESH = os.environ.get("SCAMSHIELD_NO_AUTOREFRESH", "") != "1"

# Keep every line flush left: Markdown renders 4-space-indented lines as a
# code block, which would print this stylesheet onto the page as text.
st.markdown(
"""<style>
.block-container {padding-top: 2rem; padding-bottom: 2rem;}
div[data-testid="stMetricValue"] {font-size: 1.6rem;}
.ss-alert {border-left: 5px solid #b3261e; padding: 0.6rem 0.9rem;
border-radius: 6px; margin-bottom: 0.4rem;}
.ss-review {border-left-color: #b26a00;}
.ss-reason {font-size: 0.92rem; opacity: 0.92;}
.ss-meta {font-size: 0.8rem; opacity: 0.7;}
</style>""",
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


@st.cache_data(ttl=2.0, show_spinner=False)
def load_health(db_path: str) -> dict:
    """Degraded-mode and SLO counters, for the resilience strip."""
    conn = get_conn(db_path)
    row = read_sql(conn, """
        SELECT SUM(decision_mode = 'DEGRADED') AS degraded,
               SUM(decision_mode = 'MODEL')    AS by_model,
               MAX(ring_size)                  AS largest_ring,
               SUM(CASE WHEN decision != 'ALLOW' THEN expected_loss ELSE 0 END)
                                               AS queued_exposure
        FROM transactions
    """)
    return row.iloc[0].to_dict() if len(row) else {}


@st.cache_data(ttl=5.0, show_spinner=False)
def load_metrics() -> dict:
    """The training report (cost analysis, capacity, rule comparison)."""
    try:
        return json.loads(Path(config.METRICS_PATH).read_text())
    except (OSError, json.JSONDecodeError):
        return {}


@st.cache_data(ttl=5.0, show_spinner=False)
def load_drift(db_path: str) -> dict:
    """PSI of the live score distribution against the training reference."""
    reference = monitor.load_reference()
    if not reference:
        return {"available": False}
    conn = get_conn(db_path)

    # Drift only means something when the live scores and the reference came
    # from the SAME trained model.  Pointed at a database recorded by an
    # earlier model - the LinuxONE evidence file, say - the comparison
    # produces a large, authoritative-looking number that measures nothing but
    # the difference between two models.  Refuse rather than mislead.
    meta = db.get_meta(conn)
    recorded_by = meta.get("model_trained_at", "")
    if not recorded_by:
        return {"available": False, "reason": "run predates drift tracking"}
    if recorded_by != reference.get("trained_at", ""):
        return {"available": False, "reason": "recorded by a different model"}
    # Only scores the MODEL produced belong in a drift comparison.  The
    # fallback rule emits fixed pseudo-scores, so including a degraded period
    # would register as "significant drift" when all it means is that the
    # model was briefly offline.
    live = read_sql(conn, """
        SELECT score FROM transactions
        WHERE decision_mode = 'MODEL'
        ORDER BY id DESC LIMIT 20000
    """)
    if len(live) < 200:
        # Too little model-scored traffic for PSI to mean anything yet.
        return {"available": False, "reason": "warming up"}
    out = monitor.score_drift(reference, live["score"].to_numpy())
    # Per-feature PSI needs the feature values, which we deliberately do not
    # store per transaction (they would dominate the database).  The score is
    # a function of all of them, so score PSI is the summary signal; the
    # feature references are kept in the artifact for offline investigation.
    out["features_tracked"] = list((reference.get("features") or {}).keys())
    return out


@st.cache_data(ttl=3.0, show_spinner=False)
def load_rings(db_path: str, limit: int = 8) -> pd.DataFrame:
    """Recipients sitting inside the biggest pass-through clusters."""
    conn = get_conn(db_path)
    return read_sql(conn, """
        SELECT name_dest,
               MAX(ring_size)  AS ring_size,
               COUNT(*)        AS payments,
               SUM(amount)     AS total_amount,
               MAX(score)      AS worst_score,
               SUM(is_fraud)   AS known_fraud
        FROM transactions
        WHERE ring_size > 2
        GROUP BY name_dest
        ORDER BY ring_size DESC, total_amount DESC
        LIMIT ?
    """, (limit,))


@st.cache_data(ttl=2.0, show_spinner=False)
def load_agreement(db_path: str) -> pd.DataFrame:
    """
    Does the analyst agree with the model?

    This is the feedback loop that keeps the system honest.  Approving what
    the model wanted to block is a disagreement, and a rising disagreement
    rate is the earliest warning that the model has drifted - it arrives long
    before chargeback data does.
    """
    conn = get_conn(db_path)
    return read_sql(conn, """
        SELECT t.decision, a.action, COUNT(*) AS n
        FROM analyst_actions a JOIN transactions t ON t.id = a.txn_id
        GROUP BY t.decision, a.action
    """)


@st.cache_data(ttl=1.0, show_spinner=False)
def load_recent(db_path: str, limit: int) -> pd.DataFrame:
    conn = get_conn(db_path)
    return read_sql(conn, """
        SELECT id, seq, step, type, amount, name_orig, name_dest, score,
               decision, summary, latency_ms, is_fraud, created_at,
               decision_mode, ring_size
        FROM transactions ORDER BY id DESC LIMIT ?
    """, (limit,))


@st.cache_data(ttl=1.0, show_spinner=False)
def load_queue(
    db_path: str, limit: int, include_actioned: bool, order_by: str = "exposure"
) -> pd.DataFrame:
    """
    The work queue.

    Ordering matters more than it looks.  Sorting by probability alone sends
    an analyst to a near-certain 400 fraud before a probable 900,000 one.
    Ranking by expected loss (probability x amount) puts the money first,
    which is what the team is actually paid to protect.
    """
    conn = get_conn(db_path)
    where = ["t.decision IN ('REVIEW','BLOCK')"]
    if not include_actioned:
        where.append("a.id IS NULL")       # only items nobody has decided yet
    order = ("t.expected_loss DESC, t.id DESC" if order_by == "exposure"
             else "t.score DESC, t.id DESC")
    return read_sql(conn, f"""
        SELECT t.id, t.step, t.type, t.amount, t.name_orig, t.name_dest,
               t.score, t.decision, t.summary, t.reasons, t.is_fraud, t.created_at,
               t.decision_mode, t.expected_loss, t.ring_size,
               a.action AS analyst_action
        FROM transactions t
        LEFT JOIN analyst_actions a ON a.txn_id = t.id
        WHERE {' AND '.join(where)}
        ORDER BY {order}
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
queue_order = st.sidebar.radio(
    "Work the queue by",
    options=["exposure", "score"],
    format_func=lambda k: {"exposure": "Money at risk (recommended)",
                           "score": "Risk score"}[k],
    help="Expected loss is probability x amount. Sorting by score alone sends "
         "an analyst to a certain small fraud before a probable large one.",
)
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

# ----------------------------------------------------------------------
# System health — the operations view
#
# A real-time decision system is judged on whether it keeps deciding, not
# only on whether it decides well.  This strip answers the three questions
# an operator asks: is the model healthy, are we inside our latency budget,
# and does the live traffic still look like what we trained on?
# ----------------------------------------------------------------------
health_row = load_health(db_path)
drift = load_drift(db_path)
try:
    engine_health = json.loads(meta.get("health", "{}"))
except json.JSONDecodeError:
    engine_health = {}

degraded = int(health_row.get("degraded") or 0)
breaker_state = engine_health.get("breaker_state", "closed")
breaker_trips = int(engine_health.get("breaker_trips") or 0)
slo_breaches = int(meta.get("slo_breaches") or 0)
budget_ms = float(meta.get("latency_budget_ms", config.LATENCY_BUDGET_MS))
largest_ring = int(health_row.get("largest_ring") or 1)

h1, h2, h3, h4, h5 = st.columns(5)

if breaker_state == "closed" and degraded == 0:
    h1.metric("Scoring mode", "MODEL ✅", help="Every decision came from the model.")
elif breaker_state == "closed":
    h1.metric("Scoring mode", "MODEL ✅",
              delta=f"recovered after {breaker_trips} outage(s)", delta_color="off")
else:
    h1.metric("Scoring mode", "DEGRADED ⚠️",
              delta=f"breaker {breaker_state}", delta_color="inverse",
              help="The model is unavailable. Decisions are coming from the "
                   "fallback rule and going to humans, not being auto-blocked.")

h2.metric("Decisions served in degraded mode", f"{degraded:,}",
          delta=f"{100 * degraded / n_total:.1f}% of traffic" if n_total else None,
          delta_color="off",
          help="Transactions that still got an answer while the model was down. "
               "This number being non-zero is the system working, not failing.")

breach_pct = 100 * slo_breaches / n_total if n_total else 0.0
h3.metric(f"Latency SLO ({budget_ms:g} ms)",
          f"{breach_pct:.2f}% over",
          delta=f"p99 {overview['p99']:.2f} ms",
          delta_color="inverse" if overview["p99"] > budget_ms else "normal",
          help="A payment has a hard deadline. What matters is the worst case "
               "you can guarantee, not the average.")

if drift.get("available"):
    band = drift["band"]
    icon = {"STABLE": "🟢", "MODERATE": "🟡", "SIGNIFICANT": "🔴"}[band]
    h4.metric("Data drift (PSI)", f"{drift['psi']:.3f} {icon}",
              delta=band.title(), delta_color="off",
              help="Population Stability Index of the live risk scores against "
                   "the training window. Above 0.25 means the traffic no longer "
                   "looks like what this model was validated on — retrain.")
else:
    h4.metric("Data drift (PSI)", "—",
              delta=drift.get("reason", "no reference"), delta_color="off",
              help="Needs a drift reference from train.py and a few hundred "
                   "model-scored transactions.")

if largest_ring > 1:
    h5.metric("Largest mule network", f"{largest_ring} accts",
              help="Biggest cluster of accounts observed passing money between "
                   "themselves. Found by linking rapid pass-through hops.")
else:
    # Every row sitting at the schema default means this run predates ring
    # tracking, not that the network is one account wide.
    h5.metric("Largest mule network", "—", delta="not recorded in this run",
              delta_color="off",
              help="This database was written before mule-ring tracking "
                   "existed, so there is nothing to show.")

st.divider()

# ----------------------------------------------------------------------
# Review queue — where the human decision happens
# ----------------------------------------------------------------------
queue_col, chart_col = st.columns([3, 2], gap="large")

with queue_col:
    st.subheader("Review queue")
    queue = load_queue(db_path, queue_rows, show_actioned, queue_order)
    ordered_by = ("most money at risk first" if queue_order == "exposure"
                  else "highest risk score first")
    queued_exposure = float(health_row.get("queued_exposure") or 0.0)
    st.caption(f"Ordered by {ordered_by}. **{money(queued_exposure)}** of expected "
               "loss is waiting for a decision. Every action is written to the "
               "`analyst_actions` audit table.")

    if queue.empty:
        st.success("Queue is clear — nothing is waiting for a decision.")
    for _, row in queue.iterrows():
        txn_id = int(row["id"])
        decision = row["decision"]
        css = "ss-alert" + ("" if decision == "BLOCK" else " ss-review")
        actioned = row.get("analyst_action")

        # How long has this alert been sitting unattended?  In a real
        # operation an alert that ages past its SLA is a breach, because the
        # payment it refers to has usually already settled.
        waited = ""
        try:
            created = datetime.fromisoformat(row["created_at"])
            if created.tzinfo is None:
                created = created.replace(tzinfo=timezone.utc)
            seconds = (datetime.now(timezone.utc) - created).total_seconds()
            if seconds >= 0:
                age = (f"{seconds:.0f}s" if seconds < 90
                       else f"{seconds / 60:.0f}m" if seconds < 5400
                       else f"{seconds / 3600:.1f}h")
                flag = " ⏰" if seconds > QUEUE_SLA_SECONDS else ""
                waited = f' &nbsp;·&nbsp; waiting <b>{age}</b>{flag}'
        except (TypeError, ValueError):
            pass

        # NOTE: build this as ONE unindented line.  Markdown turns any line
        # indented by 4+ spaces into a code block, which previously made the
        # closing </div></div> render as literal text on the page.
        actioned_html = (
            f" &nbsp;·&nbsp; already actioned: <b>{actioned}</b>" if actioned else ""
        )
        ring = int(row.get("ring_size") or 1)
        ring_html = (f' &nbsp;·&nbsp; <b>{ring}</b>-account network'
                     if ring > 2 else "")
        degraded_html = (
            ' &nbsp;·&nbsp; <b>⚠️ DEGRADED</b> (fallback rule, model was down)'
            if row.get("decision_mode") == "DEGRADED" else ""
        )
        summary_html = row["summary"] or "No summary available."
        st.markdown(
            f'<div class="{css}" style="background:{DECISION_BG[decision]}">'
            f'<b>{decision}</b> &nbsp;·&nbsp; <b>{money(row["amount"])}</b>'
            f' &nbsp;·&nbsp; risk <b>{row["score"]:.3f}</b>'
            f' &nbsp;·&nbsp; exposure <b>{money(row.get("expected_loss") or 0)}</b>'
            f' &nbsp;·&nbsp; {row["type"]}{degraded_html}'
            f'<div class="ss-reason">{summary_html}</div>'
            f'<div class="ss-meta">#{txn_id} &nbsp; {row["name_orig"]} &rarr; '
            f'{row["name_dest"]} &nbsp; step {int(row["step"])}'
            f'{ring_html}{waited}{actioned_html}</div>'
            f'</div>',
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
                   "score", "decision", "decision_mode", "ring_size",
                   "latency_ms", "summary"]].copy()
    view.columns = ["#", "Step", "Type", "Amount", "From", "To",
                    "Risk", "Decision", "Mode", "Network", "Latency ms", "Why"]

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

st.divider()

# ----------------------------------------------------------------------
# Mule networks + what the system is worth in money
# ----------------------------------------------------------------------
net_col, cost_col = st.columns(2, gap="large")

with net_col:
    st.subheader("Mule networks detected")
    st.caption("Accounts linked because money passed straight through them. "
               "A single hop looks ordinary; the chain does not.")
    rings = load_rings(db_path)
    if rings.empty:
        st.caption("No pass-through clusters seen yet in this run.")
    else:
        ring_view = rings.copy()
        ring_view.columns = ["Recipient", "Network size", "Payments",
                             "Total moved", "Worst risk", "Known fraud"]
        full_width(
            st.dataframe,
            ring_view.style.format({
                "Total moved": "{:,.0f}", "Worst risk": "{:.3f}",
            }),
            hide_index=True, height=260,
        )

with cost_col:
    st.subheader("What it is worth")
    metrics = load_metrics()
    cost = (metrics.get("cost_analysis") or {})
    per_1k = cost.get("cost_per_1000_payments") or {}
    if not per_1k:
        st.caption("Run `python train.py` to produce the cost analysis.")
    else:
        st.caption("Cost of operating per 1,000 payments, priced at "
                   f"{cost['assumptions']['cost_per_review']:,.0f} per analyst "
                   f"review and {cost['assumptions']['cost_per_false_block']:,.0f} "
                   "per wrongly blocked customer. Lower is better.")
        bar = pd.DataFrame({
            "approach": ["No fraud system", "Bank's current rule", "ScamShield"],
            "cost": [per_1k.get("do_nothing", 0), per_1k.get("rule_baseline", 0),
                     per_1k.get("scamshield", 0)],
        })
        fig3 = px.bar(bar, x="cost", y="approach", orientation="h",
                      color="approach", text="cost",
                      color_discrete_sequence=["#b3261e", "#b26a00", "#1b7f4d"],
                      labels={"cost": "cost per 1,000 payments", "approach": ""})
        fig3.update_traces(texttemplate="%{text:,.0f}", textposition="outside")
        fig3.update_layout(height=220, margin=dict(l=0, r=40, t=10, b=0),
                           showlegend=False)
        full_width(st.plotly_chart, fig3)
        if cost.get("saving_vs_rule_pct") is not None:
            st.success(
                f"**{cost['saving_vs_rule_pct']:.1f}% cheaper** than the rule "
                f"banks run today — {money(cost['saving_vs_rule_per_1000'])} "
                "saved per 1,000 payments."
            )
        capacity = metrics.get("capacity") or {}
        if capacity.get("threshold_set_by") == "analyst capacity":
            st.info(
                "The review threshold here is set by **analyst capacity** "
                f"(max {100 * capacity['max_alert_rate']:g}% of traffic), not by "
                f"the {100 * capacity['target_review_recall']:g}% recall target. "
                "Some fraud carries no detectable signal, and a queue nobody "
                "can work is the same as no queue at all."
            )

st.divider()

# ----------------------------------------------------------------------
# Audit trail + the feedback loop
# ----------------------------------------------------------------------
with st.expander("Analyst audit trail and model agreement"):
    actions = load_actions(db_path)
    if actions.empty:
        st.caption("No analyst decisions recorded yet — work the queue above "
                   "and they will appear here.")
    else:
        counts = actions["action"].value_counts()
        cols = st.columns(len(ACTION_LABELS) + 1)
        for col, (key, label) in zip(cols, ACTION_LABELS.items()):
            col.metric(label, int(counts.get(key, 0)))

        # Agreement rate: the analyst confirming the model (Block/Escalate on
        # something the model flagged) versus overruling it (Approve).
        agreement = load_agreement(db_path)
        if not agreement.empty:
            total = int(agreement["n"].sum())
            confirmed = int(agreement.loc[
                agreement["action"].isin(["BLOCK", "ESCALATE"]), "n"].sum())
            rate = 100.0 * confirmed / total if total else 0.0
            cols[-1].metric("Analyst agreement", f"{rate:.0f}%",
                            help="Share of worked alerts the analyst confirmed "
                                 "rather than overruled. A falling agreement "
                                 "rate is the earliest sign the model has "
                                 "drifted — it arrives months before "
                                 "chargeback data does.")
        st.caption("These rows are the training labels for the next model — "
                   "the loop that keeps the system current.")
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
