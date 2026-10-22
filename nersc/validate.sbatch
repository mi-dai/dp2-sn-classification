#!/bin/bash
#SBATCH --job-name=validate
#SBATCH --constraint=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=nersc/logs/%x-%j.out
# Classify the DP2 validation simulation with ParSNIP and a SuperNNova model directory, without redshift,
# and score them against the truth.
# Submit from the repo root:
#   sbatch -A <account> -q <qos> nersc/validate.sbatch [CATALOG=data/dp2_sim_sne] [CLASSIFIER=examples/models/plasticc_classifier.pkl] [SNN_MODEL]
# Outputs in examples/results/validate/: predictions (*.parquet) and figures (eval_*/).
# ParSNIP is skipped if the classifier doesn't exist (run train_parsnip_classifier.sbatch first).
# SNN_MODEL is a SuperNNova model directory (snn_train.sbatch, or one written by SuperNNova), run as
# snn_<dir name>; there is no default, and without it only ParSNIP runs. The Fink models (elasticc_ia,
# elasticc_broad) are no longer validated here.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

CATALOG=${1:-data/dp2_sim_sne}
CLASSIFIER=${2:-examples/models/plasticc_classifier.pkl}
SNN_MODEL=${3:-}
OUT=examples/results/validate
if [ -n "$SNN_MODEL" ] && [ ! -d "$SNN_MODEL" ]; then
    echo "SNN_MODEL $SNN_MODEL is not a directory" >&2
    exit 1
fi
mkdir -p "$OUT"
run() { .venv/bin/python examples/classify_rubin_dia.py "$CATALOG" --results-dir "$OUT/hyrax" "$@"; }
evaluate() { .venv/bin/python examples/evaluate.py "$CATALOG" "$OUT/$1.parquet" --output-dir "$OUT/eval_$1"; }

if [ -f "$CLASSIFIER" ]; then
    run --method parsnip --classifier "$CLASSIFIER" --mwebv-column mwebv --output "$OUT/parsnip.parquet" && evaluate parsnip
else
    echo "No $CLASSIFIER: skipping ParSNIP"
fi
if [ -n "$SNN_MODEL" ]; then
    NAME=snn_$(basename "$SNN_MODEL")
    run --method snn --model "$SNN_MODEL" --mwebv-column mwebv --output "$OUT/$NAME.parquet" && evaluate "$NAME"
else
    echo "No SNN_MODEL given: skipping SuperNNova"
fi
