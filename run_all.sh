#!/usr/bin/env bash
#
# ScamShield - build the whole thing: data -> model -> ablation -> benchmark.
#
#   ./run_all.sh                 # synthetic data, no Kaggle needed (~1 minute)
#   ./run_all.sh /path/to/PS_20174392719_1491204439457_log.csv
#
# Afterwards, start the live demo in TWO terminals (see the end of this file).
#
set -euo pipefail
cd "$(dirname "$0")"

PYTHON="${PYTHON:-python3}"
command -v "$PYTHON" >/dev/null 2>&1 || PYTHON=python

PAYSIM_CSV="${1:-}"

echo "============================================================"
echo " ScamShield pipeline"
echo " python : $($PYTHON -c 'import sys,platform; print(sys.version.split()[0], platform.machine())')"
echo "============================================================"

# ------------------------------------------------------------------
# 1. Data
# ------------------------------------------------------------------
if [ -n "$PAYSIM_CSV" ]; then
  echo
  echo ">>> [1/4] Sampling the real PaySim dataset"
  "$PYTHON" data/prepare.py "$PAYSIM_CSV"
else
  echo
  echo ">>> [1/4] No PaySim CSV given - generating synthetic PaySim-shaped data"
  echo "          (pass the Kaggle CSV as an argument to use the real thing)"
  "$PYTHON" data/make_synthetic.py --rows 120000
fi

# ------------------------------------------------------------------
# 2. Train
# ------------------------------------------------------------------
echo
echo ">>> [2/4] Training the model (time-based split, no random shuffle)"
"$PYTHON" train.py

# ------------------------------------------------------------------
# 3. Benchmark
# ------------------------------------------------------------------
echo
echo ">>> [3/4] Ablation: what is each feature group actually worth?"
"$PYTHON" ablation.py

# ------------------------------------------------------------------
# 4. Benchmark
# ------------------------------------------------------------------
echo
echo ">>> [4/4] Benchmarking throughput and latency on this machine"
"$PYTHON" benchmark.py

echo
echo "============================================================"
echo " Done.  Artifacts are in ./artifacts :"
echo "   metrics.json       model quality + cost analysis + rule baseline"
echo "   ablation.json      per-feature-group and per-scam-typology results"
echo "   drift_reference.json  what 'normal' looks like, for drift detection"
echo "   benchmark.json     throughput / latency / platform.machine()"
echo "   pr_curve.png       precision-recall curve"
echo "   model.joblib       the trained model"
echo
echo " Now start the LIVE DEMO in TWO terminals:"
echo
echo "   Terminal 1 - the scoring engine:"
echo "       $PYTHON stream.py --rate 50"
echo
echo "   Terminal 2 - the analyst dashboard:"
echo "       streamlit run app.py --server.port 8501 --server.address 0.0.0.0"
echo
echo "   To demo the circuit breaker (model outage) instead:"
echo "       $PYTHON stream.py --rate 300 --inject-failure 400:20"
echo
echo "   Then open http://localhost:8501"
echo "   (On LinuxONE: forward port 8501, or use the Jupyter proxy.)"
echo "============================================================"
