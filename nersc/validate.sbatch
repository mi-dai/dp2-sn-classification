#!/bin/bash
#SBATCH --job-name=validate
#SBATCH --constraint=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=nersc/logs/%x-%j.out
# Classify the DP2 validation simulation with both models and score them against the truth.
# Submit from the repo root:
#   sbatch -A <account> -q <qos> nersc/validate.sbatch [CATALOG=data/dp2_sim_sne] [CLASSIFIER=plasticc_classifier.pkl]
# Outputs in results/validate/: predictions (*.parquet) and figures (eval_*/).
# ParSNIP is skipped if the classifier doesn't exist (run train_parsnip_classifier.sbatch first).
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

CATALOG=${1:-data/dp2_sim_sne}
CLASSIFIER=${2:-plasticc_classifier.pkl}
OUT=results/validate
mkdir -p "$OUT"
run() { .venv/bin/python examples/classify_rubin_dia.py "$CATALOG" --results-dir "$OUT/hyrax" "$@"; }
evaluate() { .venv/bin/python examples/evaluate.py "$CATALOG" "$OUT/$1.parquet" --output-dir "$OUT/eval_$1"; }

if [ -f "$CLASSIFIER" ]; then
    run --method parsnip --classifier "$CLASSIFIER" --mwebv-column mwebv --output "$OUT/parsnip.parquet" && evaluate parsnip
else
    echo "No $CLASSIFIER: skipping ParSNIP"
fi
# SuperNNova without redshift, and elasticc_broad given the true redshift (flagged by evaluate.py).
run --method snn --model elasticc_ia --mwebv-column mwebv --output "$OUT/snn_ia.parquet" && evaluate snn_ia
run --method snn --redshift-column redshift --mwebv-column mwebv --output "$OUT/snn_broad.parquet" && evaluate snn_broad
