# Notes for Claude Code

Supernova classification of Rubin DP2 transients with ParSNIP (`src/hyrax_parsnip`) and SuperNNova
(`src/hyrax_snn`) run through Hyrax on HATS catalogs. README.md has the full documentation.

## Where things happen
- **Code is developed on the maintainer's Mac**, not here. On NERSC, Claude Code is used to set up the
  environment, run jobs and look at results. Don't edit tracked files or commit/push on NERSC unless the
  user explicitly asks; update with `git pull --ff-only`.
- If a job fails because of a bug, report it (command, error, log path) rather than patching the code here.

## NERSC environment (Perlmutter)
- Set up or update everything with `bash nersc/setup_env.sh` on a **login node**, from the repo. It is safe to
  rerun after a `git pull`. It builds `.venv` on top of `desc-td-env` (lightcurvelynx, sncosmo, dustmaps),
  installs the package (distribution `dp2-sn-classification`, editable, `notebook` + `dev` extras) and
  supernnova `--no-deps` (only for the reference tests), registers the Jupyter kernel
  `dp2-sn-classification (td_env)`, and runs `nersc/prefetch.py`.
- `nersc/prefetch.py` downloads what compute nodes may not reach: SuperNNova's pretrained models
  (`~/.cache/hyrax_snn`), the PLAsTiCC files (`data/plasticc_raw`, ~330 MB), and the simulation's passbands and
  templates; it checks the SFD dust map. Each step reports and continues on failure.
- Use `.venv/bin/python` (no activation needed). Check the install with `.venv/bin/pytest -q` (30 tests).
- `desc-td-env` path is set in `nersc/setup_env.sh` (`TD_PYTHON`, overridable).

## Running jobs
Submit from the repo root, passing the user's account and QOS (not stored in the scripts):
`sbatch -A <account> -q <qos> nersc/<job>.sbatch`. Logs go to `nersc/logs/`. Order:

| job | does | outputs |
|---|---|---|
| `simulate_dp2.sbatch [OUT] [N_PER_CLASS]` | DP2 validation simulation (SN Ia/II/Ib/c, DP2 DIA schema + truth) | `data/dp2_sim_sne` |
| `train_parsnip_classifier.sbatch` | PLAsTiCC (train + test DDF, 40,774 objects) → ParSNIP features → LightGBM | `plasticc_classifier.{pkl,json}` |
| `validate.sbatch [CATALOG] [CLASSIFIER]` | both models on the simulation + `evaluate.py` | `results/validate/` |
| `classify_edp2.sbatch [CATALOG] [CLASSIFIER]` | both models on the real EDP2 catalog, no redshift | `results/edp2/` |

Short checks can run interactively on a login node; anything long goes through Slurm. The demo notebook
`examples/demo_workflow.ipynb` (kernel `dp2-sn-classification (td_env)`) uses the same `data/` paths and the
classifier in the repo root. If the classifier is missing it trains it in the notebook (training set + DDF,
~35 min; `DEMO_PLASTICC_DDF=0` for the training set only); running `train_parsnip_classifier.sbatch` first avoids that.
`data/plasticc_hats` is always training set + DDF; `data/plasticc_hats_train` is the training set only.

## Data
- Real target (unlabeled, no redshifts): `/global/cfs/cdirs/lsst/groups/TD/SN/EDP2/for_fastdb/subsample_joined.hats`.
- DP2 inputs for the simulation (`examples/simulate_dp2.py`): the DP2 visit-detector table and DIA catalog under
  `/global/cfs/cdirs/lsst/shared/rubin/DP2/HATS/`.
- Generated data (`data/`), results (`results/`), logs, classifiers and root-level `.parquet` files are git-ignored.

## Decisions to respect
- The DP2 simulation is for **validation only**: don't train on it (`evaluate.py --kfold-lightgbm` is a quick
  check, not a result).
- Real data are classified **without redshift**: ParSNIP `plasticc_photoz` with the PLAsTiCC classifier;
  SuperNNova `elasticc_ia`. Results from models given the true redshift are flagged by `evaluate.py`.
- A ParSNIP classifier only applies to features from the same ParSNIP model and redshift setting; its `.json`
  is checked by `classify_rubin_dia.py`.
- SuperNNova's input options (`--detection-snr`, `--no-time-window`, `--fink-exact`) are still being chosen by
  comparing them on the simulation; the defaults are a starting point.

## Reporting results
When summarizing runs, include the numbers `evaluate.py` prints (accuracy / per-type recall, AUC,
efficiency/purity), the job IDs and output paths, so they can be compared on the Mac.
