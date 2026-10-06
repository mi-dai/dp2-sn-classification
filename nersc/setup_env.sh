#!/bin/bash
# Set up or update the environment on NERSC Perlmutter. Safe to rerun (e.g. after `git pull`).
# Run on a LOGIN node from anywhere in the repo:  bash nersc/setup_env.sh
#
# - .venv on top of desc-td-env (which provides lightcurvelynx, sncosmo, dustmaps for the simulation)
# - the package with the notebook and dev extras; supernnova (--no-deps) for the reference tests
# - the Jupyter kernel "dp2-sn-classification (td_env)"
# - downloads that compute nodes may not be able to do (nersc/prefetch.py)
set -euo pipefail

TD_PYTHON=${TD_PYTHON:-/global/common/software/lsst/install/td_env/2026-09-03-32-23/py/envs/td_env/bin/python}
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"

if [ ! -x .venv/bin/python ]; then
    echo "Creating .venv on top of desc-td-env ($TD_PYTHON)"
    "$TD_PYTHON" -m venv --system-site-packages .venv
fi

# The distribution used to be called hyrax-parsnip.
.venv/bin/pip uninstall -y -q hyrax-parsnip 2>/dev/null || true
.venv/bin/pip install -q -e ".[notebook,dev]"
# Only for tests/test_snn_reference.py: supernnova pins pandas<3, so never install its dependencies.
.venv/bin/pip install -q --no-deps supernnova==3.0.51 natsort seaborn

.venv/bin/python -m ipykernel install --user --name dp2-sn-classification-td \
    --display-name "dp2-sn-classification (td_env)"
if jupyter kernelspec list 2>/dev/null | grep -q "hyrax-parsnip-td"; then
    jupyter kernelspec remove -f hyrax-parsnip-td || true  # old kernel name
fi

mkdir -p nersc/logs
.venv/bin/python nersc/prefetch.py "$@"

echo
echo "Done. Check with: .venv/bin/pytest -q"
echo "Submit jobs from the repo root, e.g.: sbatch -A <account> -q <qos> nersc/train_parsnip_classifier.sbatch"
