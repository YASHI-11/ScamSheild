"""
ScamShield - central configuration.

Every path here is RELATIVE to the repository root so the project behaves the
same on a Windows/Mac laptop and on IBM LinuxONE (s390x) inside the container.
"""
import os
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent

# Where generated files go.  Normally the repo itself, but setting
# SCAMSHIELD_HOME redirects every OUTPUT (artifacts, generated feeds, the
# database) somewhere else.  The test suite relies on this: without it,
# running `pytest` would retrain into artifacts/ and overwrite the model and
# feeds you just built for the demo.
HOME = Path(os.environ.get("SCAMSHIELD_HOME", ROOT)).resolve()

DATA_DIR = HOME / "data"
ARTIFACT_DIR = HOME / "artifacts"

# PaySim sample produced by data/prepare.py (or data/make_synthetic.py)
SAMPLE_CSV = DATA_DIR / "paysim_sample.csv"

# Written by train.py: the held-out (later-in-time) rows that stream.py replays.
STREAM_CSV = DATA_DIR / "stream_feed.csv"
# Written by train.py: the earlier rows, replayed silently by stream.py so the
# AccountState starts the demo with the same history it had during training.
WARMUP_CSV = DATA_DIR / "warmup_feed.csv"

MODEL_PATH = ARTIFACT_DIR / "model.joblib"
METADATA_PATH = ARTIFACT_DIR / "model_metadata.json"   # feature list + thresholds
METRICS_PATH = ARTIFACT_DIR / "metrics.json"
# Score/feature distribution captured at training time.  monitor.py compares
# the live stream against this to detect drift (a model going stale).
DRIFT_REFERENCE_PATH = ARTIFACT_DIR / "drift_reference.json"
PR_CURVE_PNG = ARTIFACT_DIR / "pr_curve.png"
PR_CURVE_CSV = ARTIFACT_DIR / "pr_curve.csv"
BENCHMARK_PATH = ARTIFACT_DIR / "benchmark.json"
ONNX_PATH = ARTIFACT_DIR / "model.onnx"

DB_PATH = HOME / "scamshield.db"

# --------------------------------------------------------------------------
# Data preparation
# --------------------------------------------------------------------------
SAMPLE_ROWS = 300_000      # target size of the stratified sample (ALL fraud kept)
RANDOM_SEED = 42

# --------------------------------------------------------------------------
# Feature engineering (see features.py)
# --------------------------------------------------------------------------
# In PaySim one "step" == one hour of simulated time.
FANIN_WINDOW_STEPS = 12        # N: look-back window for "distinct senders to dest"
PASSTHROUGH_WINDOW_STEPS = 2   # K: received-then-forwarded within K steps
PASSTHROUGH_RATIO = 0.80       # "most of it" == >= 80% of the money just received

# --- velocity / structuring -------------------------------------------
# Criminals who know the bank's reporting threshold split one big transfer
# into several smaller ones ("structuring" / "smurfing").  A single payment
# looks harmless; six of them in six hours do not.
VELOCITY_WINDOW_STEPS = 6      # W: look-back window for "payments by this sender"
# Amounts landing in the band just UNDER a round reporting threshold.
STRUCTURING_THRESHOLDS = (50_000.0, 100_000.0, 200_000.0, 500_000.0, 1_000_000.0)
STRUCTURING_BAND = 0.10        # "just under" == within 10% below the threshold

# --- victim-side (out-of-character) signals ---------------------------
# Scam victims are talked into a payment unlike anything they normally make.
# These features compare the payment against the SENDER'S OWN history, which
# is what makes the system protect vulnerable customers rather than just
# hunting criminals.
NIGHT_HOURS = (0, 1, 2, 3, 4, 5)

# --------------------------------------------------------------------------
# Train / test split (time based - never random)
# --------------------------------------------------------------------------
TRAIN_FRACTION = 0.70          # earliest 70% of steps train, the rest is test

# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
XGB_PARAMS = dict(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.15,
    subsample=0.9,
    colsample_bytree=0.9,
    min_child_weight=1,
    reg_lambda=1.0,
    objective="binary:logistic",
    eval_metric="aucpr",
    tree_method="hist",
    n_jobs=0,                  # 0 => use every core (shows off LinuxONE)
    random_state=RANDOM_SEED,
)

# --------------------------------------------------------------------------
# Decision thresholds
# --------------------------------------------------------------------------
# REVIEW is tuned for recall (catch nearly everything, humans triage).
# BLOCK is tuned for precision (safe to stop the payment automatically).
TARGET_REVIEW_RECALL = 0.95
TARGET_BLOCK_PRECISION = 0.95

# ANALYST CAPACITY - the constraint every real fraud team actually operates
# under, and the one most fraud demos quietly ignore.
#
# A recall target on its own is not implementable.  Some fraud genuinely
# leaves no trace in the data a payment switch can see, so pushing recall
# towards 100% drives the threshold towards zero and buries the team: on our
# own test data, insisting on 95% recall queues a third of ALL traffic.  A
# queue nobody can work is the same as no queue at all.
#
# So the REVIEW tier is capped at the share of traffic a team can genuinely
# review, and the threshold becomes "the best recall available WITHIN
# capacity".  train.py reports loudly when the cap, rather than the recall
# target, is what set the threshold.
MAX_ALERT_RATE = 0.02          # at most 2% of traffic may enter the queue
# Fallbacks used only if the targets cannot be met on the test set.
DEFAULT_REVIEW_THRESHOLD = 0.30
DEFAULT_BLOCK_THRESHOLD = 0.90

# --------------------------------------------------------------------------
# Cost model  (the thresholds above optimise statistics; this optimises MONEY)
# --------------------------------------------------------------------------
# A bank does not care about F1 - it cares about its loss account.  Each
# possible outcome has a different price, and they differ by three orders of
# magnitude, so the statistically "best" threshold is almost never the
# cheapest one.  train.py sweeps the thresholds to minimise expected cost and
# reports the saving against the rule banks run today.
COST_REVIEW = 150.0            # analyst time to work one queued alert
COST_FALSE_BLOCK = 2_500.0     # blocking an honest customer: support + churn
COST_MISSED_FRAUD_RATIO = 1.0  # fraud we allow through costs the full amount
# Analysts are good, not perfect.  Assuming they catch everything flatters
# the model, so we only credit them with this share of the fraud we queue.
ANALYST_CATCH_RATE = 0.90

# --------------------------------------------------------------------------
# Resilience  (what happens when the model itself fails)
# --------------------------------------------------------------------------
# A payment switch cannot stop taking decisions because a model misbehaves.
# scorer.py wraps scoring in a circuit breaker: after this many consecutive
# failures it stops calling the model and falls back to the static rule,
# marking those decisions DEGRADED so the audit trail stays honest.
LATENCY_BUDGET_MS = 10.0       # per-transaction SLO; breaches are counted
BREAKER_FAILURE_THRESHOLD = 5  # consecutive failures before the breaker opens
BREAKER_COOLDOWN_SECONDS = 2.0 # how long it stays open before probing again

# --------------------------------------------------------------------------
# Drift monitoring (see monitor.py)
# --------------------------------------------------------------------------
# Fraud changes every week; a model trained in January is wrong by June.
# We store the training score distribution and compare the live one against
# it using the Population Stability Index.  The bands below are the industry
# convention.
DRIFT_BINS = 10
PSI_WARN = 0.10                # 0.10 - 0.25: moderate shift, keep watching
PSI_ALERT = 0.25               # above 0.25: retrain

# --------------------------------------------------------------------------
# Rule baseline (the "what banks do today" comparison)
# --------------------------------------------------------------------------
RULE_AMOUNT = 200_000.0

# --------------------------------------------------------------------------
# Streaming / dashboard
# --------------------------------------------------------------------------
STREAM_RATE = 50               # transactions per second (0 == as fast as possible)
DASHBOARD_REFRESH_SECONDS = 2
TOP_REASONS = 3

# --------------------------------------------------------------------------
# Benchmark
# --------------------------------------------------------------------------
BENCHMARK_ROWS = 50_000

# --------------------------------------------------------------------------
# Explanations
# --------------------------------------------------------------------------
# Template-based summaries are the default and need no network / no extra model.
# Set to True only if you wire up the optional hook in explain.py.
USE_GENERATIVE_SUMMARY = False
