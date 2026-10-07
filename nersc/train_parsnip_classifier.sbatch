#!/bin/bash
#SBATCH --job-name=train_parsnip_classifier
#SBATCH --constraint=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=nersc/logs/%x-%j.out
# PLAsTiCC (training set + test-set DDF, 40,774 objects) -> HATS -> ParSNIP features -> LightGBM classifier
# (SNIa vs non-SNIa; pass --classes dp2 for SNIa/SNII/SNIbc/other).
# Writes examples/models/plasticc_classifier.pkl / .json / _out_of_sample.parquet. Submit from the repo root:
#   sbatch -A <account> -q <qos> nersc/train_parsnip_classifier.sbatch [extra args for parsnip_train_classifier.py]
# The PLAsTiCC files are downloaded by nersc/prefetch.py (run by setup_env.sh) on the login node.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

if [ ! -d data/plasticc_hats ]; then
    .venv/bin/python examples/plasticc_to_hats.py data/plasticc_hats --ddf
fi
.venv/bin/python examples/parsnip_train_classifier.py data/plasticc_hats \
    --output examples/models/plasticc_classifier.pkl --results-dir results/train_parsnip_classifier "$@"
