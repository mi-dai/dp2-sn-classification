#!/bin/bash
#SBATCH --job-name=simulate_dp2
#SBATCH --constraint=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=nersc/logs/%x-%j.out
# DP2 validation simulation (SN Ia / II / Ib/c in the DP2 DIA schema). Submit from the repo root,
# passing your account and QOS:
#   sbatch -A <account> -q <qos> nersc/simulate_dp2.sbatch [OUTPUT=data/dp2_sim_sne] [N_PER_CLASS=1000] [extra args]
# Extra args go to examples/simulate_dp2.py (e.g. --ddf-only, --seed 1).
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

OUTPUT=${1:-data/dp2_sim_sne}
N_PER_CLASS=${2:-1000}
shift $(( $# > 2 ? 2 : $# ))

.venv/bin/python examples/simulate_dp2.py "$OUTPUT" --n-per-class "$N_PER_CLASS" \
    --num-jobs "${SLURM_CPUS_PER_TASK:-1}" "$@"
