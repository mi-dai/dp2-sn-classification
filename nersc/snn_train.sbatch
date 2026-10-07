#!/bin/bash
#SBATCH --job-name=snn_train
#SBATCH --constraint=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=nersc/logs/%x-%j.out
# Optional: train SuperNNova on PLAsTiCC (training set + test-set DDF, 40,774 objects), SNIa vs non-SNIa,
# no redshift, 90 epochs (~2-3 h on CPU). The Fink pretrained models need no training.
# Writes examples/models/snn_plasticc/ (cli_args.json, data_norm.json, model.pt). Submit from the repo root:
#   sbatch -A <account> -q <qos> nersc/snn_train.sbatch [extra args for snn_train.py, e.g. --overwrite]
# The PLAsTiCC files are downloaded by nersc/prefetch.py (run by setup_env.sh) on the login node.
# If the job hits its time limit, resubmit with --resume latest: it continues from the last epoch's
# checkpoint in examples/results/snn_train/ (same other arguments; --epochs is the total).
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

if [ ! -d data/plasticc_hats ]; then
    .venv/bin/python examples/plasticc_to_hats.py data/plasticc_hats --ddf
fi
.venv/bin/python examples/snn_train.py data/plasticc_hats \
    --output examples/models/snn_plasticc --results-dir examples/results/snn_train "$@"
