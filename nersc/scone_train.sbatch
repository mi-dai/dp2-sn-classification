#!/bin/bash
#SBATCH --job-name=scone_train
#SBATCH --constraint=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=12:00:00
#SBATCH --output=nersc/logs/%x-%j.out
# Train SCONE on PLAsTiCC (training set + test-set DDF, 40,774 objects), SNIa vs non-SNIa, no redshift,
# 400 epochs. SCONE publishes no trained weights, so this is needed before running SCONE.
# Writes examples/models/scone_plasticc/ (scone_model.json, model.pt). Submit from the repo root:
#   sbatch -A <account> -q <qos> nersc/scone_train.sbatch [extra args for scone_train.py, e.g. --overwrite]
# The GP heatmaps are made first, in parallel on --cpus-per-task processes (one per CPU of the node ran out
# of memory; --mem=0 gives the job all of the node's memory), and cached in data/scone_heatmaps/
# (reused by later runs). The PLAsTiCC files are downloaded by nersc/prefetch.py (run by setup_env.sh).
# If the job hits its time limit, resubmit with --resume latest: it continues from the last epoch's
# checkpoint in examples/results/scone_train/ (same other arguments; --epochs is the total).
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

if [ ! -d data/plasticc_hats ]; then
    .venv/bin/python examples/plasticc_to_hats.py data/plasticc_hats --ddf
fi
.venv/bin/python examples/scone_train.py data/plasticc_hats \
    --output examples/models/scone_plasticc --results-dir examples/results/scone_train "$@"
