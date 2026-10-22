#!/bin/bash
#SBATCH --job-name=classify_edp2
#SBATCH --constraint=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=nersc/logs/%x-%j.out
# Classify the real EDP2 DIA catalog (no labels, no redshifts) with ParSNIP and a SuperNNova model directory,
# without redshift.
# Submit from the repo root:
#   sbatch -A <account> -q <qos> nersc/classify_edp2.sbatch [CATALOG=<EDP2 default>] [CLASSIFIER=examples/models/plasticc_classifier.pkl] [SNN_MODEL]
# Outputs in examples/results/edp2/. SNN_MODEL is a SuperNNova model directory (snn_train.sbatch, or one
# written by SuperNNova), run as snn_<dir name>; there is no default, and without it only ParSNIP runs.
# The Fink elasticc_ia model is no longer run here.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

CATALOG=${1:-/global/cfs/cdirs/lsst/groups/TD/SN/EDP2/for_fastdb/subsample_joined.hats}
CLASSIFIER=${2:-examples/models/plasticc_classifier.pkl}
SNN_MODEL=${3:-}
OUT=examples/results/edp2
if [ -n "$SNN_MODEL" ] && [ ! -d "$SNN_MODEL" ]; then
    echo "SNN_MODEL $SNN_MODEL is not a directory" >&2
    exit 1
fi
mkdir -p "$OUT"

if [ -f "$CLASSIFIER" ]; then
    .venv/bin/python examples/classify_rubin_dia.py "$CATALOG" --method parsnip --classifier "$CLASSIFIER" \
        --output "$OUT/parsnip.parquet" --results-dir "$OUT/hyrax"
else
    echo "No $CLASSIFIER: skipping ParSNIP"
fi
if [ -n "$SNN_MODEL" ]; then
    .venv/bin/python examples/classify_rubin_dia.py "$CATALOG" --method snn --model "$SNN_MODEL" \
        --output "$OUT/snn_$(basename "$SNN_MODEL").parquet" --results-dir "$OUT/hyrax"
else
    echo "No SNN_MODEL given: skipping SuperNNova"
fi
