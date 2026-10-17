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
# Fallbacks used only if the targets cannot be met on the test set.
DEFAULT_REVIEW_THRESHOLD = 0.30
DEFAULT_BLOCK_THRESHOLD = 0.90

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
