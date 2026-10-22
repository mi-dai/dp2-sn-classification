# Notes for Claude Code

Supernova classification of Rubin DP2 transients with ParSNIP (`src/hyrax_parsnip`) and SuperNNova
(`src/hyrax_snn`) run through Hyrax on HATS catalogs, both reading them with the shared dataset in
`src/hyrax_lightcurves`. README.md has the full documentation.

## Where things happen
- Code is developed here on NERSC (since 2026-10-22; previously on the maintainer's Mac). Fix bugs found in
  runs directly, with a test, and run `.venv/bin/python -m pytest -q` before committing.
- Commit only when the user asks. Pushing from Claude's shell has no credentials: commit locally, the user pushes.

## NERSC environment (Perlmutter)
- Set up or update everything with `bash nersc/setup_env.sh` on a **login node**, from the repo. It is safe to
  rerun after a `git pull`. It builds `.venv` on top of `desc-td-env` (lightcurvelynx, sncosmo, dustmaps),
  installs the package (distribution `dp2-sn-classification`, editable, `notebook` + `dev` extras) and
  supernnova `--no-deps` (only for the reference tests), registers the Jupyter kernel
  `dp2-sn-classification (td_env)`, and runs `nersc/prefetch.py`.
- `nersc/prefetch.py` downloads what compute nodes may not reach: SuperNNova's pretrained models
  (`~/.cache/hyrax_snn`), the PLAsTiCC files (`data/plasticc_raw`, ~330 MB), and the simulation's passbands and
  templates; it checks the SFD dust map. Each step reports and continues on failure.
- Use `.venv/bin/python` (no activation needed). Check the install with `.venv/bin/python -m pytest -q` (36 tests).
- `desc-td-env` path is set in `nersc/setup_env.sh` (`TD_PYTHON`, overridable).

## Running jobs
Submit from the repo root, passing the user's account and QOS (not stored in the scripts):
`sbatch -A <account> -q <qos> nersc/<job>.sbatch`. Logs go to `nersc/logs/`. Order:

| job | does | outputs |
|---|---|---|
| `simulate_dp2.sbatch [OUT] [N_PER_CLASS]` | DP2 validation simulation (SN Ia/II/Ib/c, DP2 DIA schema + truth) | `data/dp2_sim_sne` |
| `train_parsnip_classifier.sbatch` | PLAsTiCC (train + test DDF, 40,774 objects) → ParSNIP features → LightGBM (SNIa vs non-SNIa) | `examples/models/plasticc_classifier.{pkl,json}` |
| `snn_train.sbatch` (optional) | PLAsTiCC (same catalog) → SuperNNova trained with Hyrax (SNIa vs non-SNIa, no redshift, 90 epochs) | `examples/models/snn_plasticc/` (time limit hit: resubmit with `--resume latest`) |
| `validate.sbatch [CATALOG] [CLASSIFIER] [SNN_MODEL]` | both models on the simulation + `evaluate.py`; also the trained SuperNNova model if `SNN_MODEL` (default `examples/models/snn_plasticc`) exists | `examples/results/validate/` |
| `classify_edp2.sbatch [CATALOG] [CLASSIFIER] [SNN_MODEL]` | both models on the real EDP2 catalog, no redshift; same `SNN_MODEL` rule | `examples/results/edp2/` |

Short checks can run interactively on a login node; anything long goes through Slurm. The demo notebook
`examples/demo_workflow.ipynb` (kernel `dp2-sn-classification (td_env)`) uses the same `data/` paths and the
classifier in `examples/models/`. If the classifier is missing it trains it in the notebook (training set + DDF,
~35 min; `DEMO_PLASTICC_DDF=0` for the training set only); running `train_parsnip_classifier.sbatch` first avoids that.
`data/plasticc_hats` is always training set + DDF; `data/plasticc_hats_train` is the training set only.

## Data
- Real target (unlabeled, no redshifts): `/global/cfs/cdirs/lsst/groups/TD/SN/EDP2/for_fastdb/subsample_joined.hats`.
- DP2 inputs for the simulation (`examples/simulate_dp2.py`): the DP2 visit-detector table and DIA catalog under
  `/global/cfs/cdirs/lsst/shared/rubin/DP2/HATS/`.
- Outputs: data in `data/`; trained classifiers in `examples/models/`; predictions, figures and Hyrax runs in
  `examples/results/` (script defaults resolve there from any working directory). All git-ignored.

## Decisions to respect
- The DP2 simulation is for **validation only**: don't train on it (`evaluate.py --kfold-lightgbm` is a quick
  check, not a result).
- Real data are classified **without redshift**: ParSNIP `plasticc_photoz` with the PLAsTiCC classifier;
  SuperNNova `elasticc_ia`. Both are SNIa vs non-SNIa by default (`parsnip_train_classifier.py --classes dp2`
  gives SNIa/SNII/SNIbc/other). Results from models given the true redshift are flagged by `evaluate.py`.
- A ParSNIP classifier only applies to features from the same ParSNIP model and redshift setting; its `.json`
  is checked by `classify_rubin_dia.py`.
- SuperNNova: the Fink pretrained models are the default and need no training. Training on PLAsTiCC
  (`snn_train.py`) is optional, for comparing with ParSNIP on the same training data; a trained model is an
  ordinary SuperNNova model directory, used with `classify_rubin_dia.py --method snn --model DIR` (its
  `cli_args.json` records the flux zeropoint, so DP2 nJy fluxes are rescaled to match).
- SuperNNova's input options (`--detection-snr`, `--no-time-window`, `--fink-exact`) are still being chosen by
  comparing them on the simulation; the defaults are a starting point.

## Reporting results
When summarizing runs, include the numbers `evaluate.py` prints (accuracy / per-type recall, AUC,
efficiency/purity), the job IDs and output paths, so they can be compared across runs.
