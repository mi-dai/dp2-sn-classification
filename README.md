# dp2-sn-classification

Photometric classification of Rubin DP2 transients with pretrained models run through
[Hyrax](https://hyrax.readthedocs.io), on light curves stored in [HATS](https://hats.readthedocs.io)
catalogs. Two classifiers are available, each as its own package in the `dp2-sn-classification` distribution:

| | ParSNIP (`hyrax_parsnip`) | SuperNNova (`hyrax_snn`) |
|---|---|---|
| Model | [ParSNIP](https://parsnip.readthedocs.io) generative model + LightGBM classifier | [SuperNNova](https://supernnova.readthedocs.io) recurrent network |
| Pretrained | `plasticc`, `plasticc_photoz`, `ps1` (light-curve model only) | Fink's ELAsTiCC classifiers (complete) |
| Training needed | the LightGBM classifier, once (here: on PLAsTiCC) | none; optional training (e.g. on PLAsTiCC) |
| Without redshift | `plasticc_photoz` (predicts the redshift) | `elasticc_ia`, `SN_vs_other` |
| With redshift | `plasticc` | `elasticc_broad`, `*_vs_other` |
| Classes | what the classifier is trained on (here: SNIa vs non-SNIa by default) | fixed by the model (e.g. SNIa/other; SN/Fast/Long/Periodic/NonPeriodic), or what it is trained on |
| Also gives | latent features, predicted redshift; optional fine-tuning | |

Both read the same catalogs with the same dataset class and Rubin DIA settings, and the example
scripts run either one with `--method {parsnip,snn}`.

## Workflow

The real data (the EDP2 DIA catalog) are unlabeled and have no redshifts. A simulation of DP2 with the
same schema is used to **validate** the classifiers; nothing is trained on it.

```bash
# Validation catalog: SN Ia / II / Ib/c on the real DP2 visits (NERSC; or let the demo notebook make it)
python examples/simulate_dp2.py data/dp2_sim_sne

# ParSNIP: train its classifier on PLAsTiCC once, then classify and score
python examples/plasticc_to_hats.py data/plasticc_hats --ddf
python examples/parsnip_train_classifier.py data/plasticc_hats   # writes examples/models/plasticc_classifier.{pkl,json}
python examples/classify_rubin_dia.py data/dp2_sim_sne --method parsnip --classifier examples/models/plasticc_classifier.pkl
python examples/evaluate.py data/dp2_sim_sne examples/results/parsnip_predictions.parquet

# SuperNNova: pretrained, classify and score
python examples/classify_rubin_dia.py data/dp2_sim_sne --method snn
python examples/evaluate.py data/dp2_sim_sne examples/results/snn_predictions.parquet

# SuperNNova trained on PLAsTiCC (optional), to compare with ParSNIP on the same training data
python examples/snn_train.py data/plasticc_hats   # writes examples/models/snn_plasticc/
python examples/classify_rubin_dia.py data/dp2_sim_sne --method snn --model examples/models/snn_plasticc

# Real data: classify_rubin_dia.py defaults to the EDP2 catalog
python examples/classify_rubin_dia.py --method parsnip --classifier examples/models/plasticc_classifier.pkl
python examples/classify_rubin_dia.py --method snn
```

`examples/demo_workflow.ipynb` walks through the same steps interactively and compares both models.
`classify_rubin_dia.py` writes one table for either method: `diaObjectId`, `method`, `model`,
`redshift_input`, `p_<class>` and `predicted_class` (plus ParSNIP's features). `evaluate.py` scores it
against the simulation truth and warns when the model was given the redshift.

Outputs go under `examples/` whatever the working directory: trained classifiers in `examples/models/`,
predictions, figures and Hyrax runs in `examples/results/` (e.g. `examples/results/snn_predictions.parquet`,
`examples/results/eval_snn_predictions/`). Data (simulation, PLAsTiCC) stay in `data/`. All are git-ignored.

## Install

```bash
python -m venv .venv
.venv/bin/pip install -e ".[dev]"
```

On macOS, LightGBM (required by ParSNIP) needs the OpenMP runtime: `brew install libomp`.

At NERSC, run `bash nersc/setup_env.sh` on a login node (safe to rerun after `git pull`). It builds the venv on
top of `desc-td-env`, which provides lightcurvelynx for the simulations, installs the package, registers the
Jupyter kernel `dp2-sn-classification (td_env)`, and pre-downloads the models and data that compute nodes may
not reach (`nersc/prefetch.py`). By hand, the core steps are:

```bash
/global/common/software/lsst/install/td_env/2026-09-03-32-23/py/envs/td_env/bin/python \
    -m venv --system-site-packages .venv
.venv/bin/pip install -e ".[notebook,dev]"
.venv/bin/python -m ipykernel install --user --name dp2-sn-classification-td --display-name "dp2-sn-classification (td_env)"
```

Slurm jobs for the long steps are in `nersc/` (`simulate_dp2`, `train_parsnip_classifier`, `validate`,
`classify_edp2`); submit them from the repo root with your account and QOS:
`sbatch -A <account> -q <qos> nersc/validate.sbatch`. `CLAUDE.md` describes this setup for Claude Code.

Extras: `notebook` (Jupyter), `sim` (lightcurvelynx, sncosmo, dustmaps, mocpy; included in `desc-td-env`).
SuperNNova's pretrained models are downloaded on first use (see below); SuperNNova itself is not a
dependency.

## Data

### Catalog format

A nested HATS catalog with one row per object, e.g. as produced by LSDB / nested-pandas:

| column       | type                                                    |
|--------------|---------------------------------------------------------|
| `object_id`  | id                                                      |
| `redshift`   | float (only for models that take a redshift input)      |
| `lightcurve` | nested: `mjd`, `flux`, `fluxerr`, `band`                |

Both classifiers read it with `hyrax_parsnip.dataset.ParsnipHATSDataset`. All names are configurable under
`[data_set.ParsnipHATSDataset]`, and so are `mwebv_column`, `label_column`, `flux_scale` (multiplies flux and
error), `flag_columns` (drops flagged points) and `photoz_column` (per-object photo-z prior, see below). For training, `label_scheme` (`ia`, `dp2` or `all`) maps the
`label_column` types to classes, and the `label_index` field gives each object's class index. `band_map` maps catalog bands to the bands a model knows;
see [`default_config.toml`](src/hyrax_parsnip/default_config.toml).

### Rubin DIA catalogs

Rubin DIA catalogs have one row per `diaObject` with nested `diaObjectForcedSource` photometry
(`midpointMjdTai`, `band`, `psfDiffFlux`/`psfDiffFluxErr` in nJy, flags). `RUBIN_DIA_SETTINGS` in each
package maps these columns and drops flagged points; ParSNIP's version rescales nJy to the PLAsTiCC
zeropoint (27.5), SuperNNova's keeps nJy:

```python
for key, value in hyrax_parsnip.RUBIN_DIA_SETTINGS.items():
    h.set_config(f"data_set.ParsnipHATSDataset.{key}", value)
```

### DP2 simulation (validation)

`examples/simulate_dp2.py` forward-models SN Ia (SALT3) and SN II / Ib/c (Vincenzi et al. 2019 templates)
with lightcurvelynx through the DP2 visit-detector table, and writes them in the DP2 DIA schema plus truth
columns (`type`, `redshift`, `t0`, `mwebv`, `template`). It needs NERSC (the DP2 visit table) and the `sim`
extra. Use it for validation only.

### PLAsTiCC (training ParSNIP's classifier)

`examples/plasticc_to_hats.py` downloads [PLAsTiCC](https://zenodo.org/records/2539456) (Kessler et al.
2019, CC-BY-4.0) from Zenodo and writes a labeled HATS catalog: the official training set (7,848 objects)
and, with `--ddf`, all 32,926 Deep Drilling Field objects of the unblinded test set (~330 MB download).
Multimodal Universe's copy on Hugging Face (`hf://MultimodalUniverse/plasticc`, readable with Hyrax's
`MultimodalUniverseDataset`) has only the training set and no Y-band data, so the Zenodo files are used.

## ParSNIP (`hyrax_parsnip`)

### Inference with a pretrained model

```python
from hyrax import Hyrax
import hyrax_parsnip

h = Hyrax()
hyrax_parsnip.configure(h, "/path/to/hats_catalog", pretrained="plasticc")

results = h.infer()                                   # no training run needed
predictions = hyrax_parsnip.load_predictions(results) # astropy Table, one row per object
```

`configure` points `infer.model_weights_file` at the pretrained ParSNIP file. Objects that ParSNIP cannot
process (e.g. no observations in the model's time window) get NaN features rather than being dropped, so
object ids stay aligned. The output has ParSNIP's latent features (`s1..s3`, color, amplitude, luminosity, ...).

### Without redshifts

The `plasticc` and `ps1` models need a redshift for every object. `plasticc_photoz` predicts it from the
light curve instead:

```python
hyrax_parsnip.configure(h, catalog, pretrained="plasticc_photoz")  # also sets require_redshift = false
h.set_config("data_set.ParsnipHATSDataset.redshift_column", False)
predictions = hyrax_parsnip.load_predictions(h.infer())             # includes predicted_redshift
```

The model was trained with a host-galaxy photo-z as an encoder input. Every object gets
`model.HyraxParsnip.photoz` / `photoz_error` (default 0.5 ± 1.0), a weak prior. On simulated SN Ia/II-P/Ib/c
at z = 0.03–0.2, this gave Δz/(1+z) = +0.075 ± 0.068 (NMAD). Hyrax logs a warning about NaN inputs (the
missing redshifts); the photo-z model ignores them.

To test the photo-z estimate with an informative prior, set the dataset's `photoz_column` (e.g. the true
redshift of a simulation): objects with a value get it as host photo-z, with error
`model.HyraxParsnip.photoz_fractional_error` × (1+z) (default 0.05); the others keep the constant prior.
From the command line: `classify_rubin_dia.py CATALOG --method parsnip --photoz-column redshift
[--photoz-error 0.05]`; the output is then marked `redshift_input`, since the prior carries the redshift.
On the same toy simulation (300 objects), the prior centred on the true z gave Δz/(1+z) bias / NMAD of
+0.005 / 0.011 (σ = 0.01(1+z)), +0.017 / 0.017 (0.05) and +0.036 / 0.028 (0.2), against +0.075 / 0.068 with
the constant prior. A classifier trained on constant-prior features (`parsnip_train_classifier.py`) isn't
matched to these features (`classify_rubin_dia.py` warns).

### Classification

ParSNIP does not ship a classifier, so one is trained once on labeled data, saved, and reused:

```python
labels = hyrax_parsnip.catalog_metadata(catalog, ["type"])
predictions = hyrax_parsnip.load_predictions(h.infer(), metadata=labels)

classifier, out_of_sample = hyrax_parsnip.train_classifier(predictions, label_column="type")
classifier.write("classifier.pkl")

# Later, on new data:
classifier = hyrax_parsnip.load_classifier("classifier.pkl")
probabilities = hyrax_parsnip.classify(classifier, new_predictions)
```

The classifier is ParSNIP's LightGBM model on 11 features (latents, color, luminosity and their errors).
Its default `min_child_weight=1000` suits PLAsTiCC-sized training sets; with only a few hundred labeled
objects it prevents every split and all probabilities come out equal, so pass e.g. `min_child_weight=10.0`.

Here the classifier is trained on PLAsTiCC with `examples/parsnip_train_classifier.py` (see
[Workflow](#workflow)):
- By default it is binary, **SNIa vs non-SNIa** (every other PLAsTiCC type, including SNIa-91bg and SNIax), which
  compares directly with SuperNNova's `elasticc_ia`. `--classes dp2` trains SNIa, SNII, SNIbc and "other";
  `--classes all` every PLAsTiCC type.
- By default the features come from `plasticc_photoz` with the same weak photo-z prior used for DP2, so no
  external redshift is involved. `--redshift-column redshift` trains the `plasticc` (true-redshift) variant.
  The classifier has no redshift feature, but with `plasticc` the redshift enters through `luminosity` and
  the rest-frame latents.
- It writes a `.json` next to the classifier recording the ParSNIP model and whether it was given the
  redshift. `classify_rubin_dia.py` takes `--model` from it and refuses a different model or redshift
  setting, since the features would differ.
- Fluxes are used as distributed: ParSNIP was trained on these values, and DP2's nJy fluxes are converted to
  the same scale (zp 27.5).
- Trained on the PLAsTiCC training set alone, the binary classifier's K-fold accuracy is 0.93 (SNIa recall 0.925).
  Applied without retraining to an independent toy simulation (sncosmo SN Ia/II-P/Ib/c), SN Ia vs rest AUC was
  0.95 (efficiency 0.90, purity 0.70 at P(SNIa) ≥ 0.5); the contamination is mostly SN Ib/c (37% called SNIa).
  The 4-class version (`--classes dp2`) reached K-fold accuracy 0.85, and 0.71 on the toy simulation.

### Training and fine-tuning ParSNIP (optional)

```python
h.set_config("train.epochs", 50)
h.set_config("data_loader.batch_size", 128)
hyrax_parsnip.configure(h, catalog, pretrained=False,       # or "plasticc" to fine-tune
                        groups=("train", "infer"), model_weights_file=False)

model = h.train()
model.export_parsnip("my_model.pt")   # usable with parsnip.load_model / pretrained="my_model.pt"
results = h.infer()                    # uses the latest training run
```

- Training uses ParSNIP's own loss, optimizer, augmentation, and ReduceLROnPlateau schedule. Hyrax's
  `[optimizer]`/`[criterion]`/`[scheduler]` settings are ignored (Hyrax logs a warning about this).
- A Hyrax epoch is one pass over the catalog. ParSNIP's `fit` counts ~25,000 augmented light curves as an
  epoch, so small catalogs need more Hyrax epochs.
- For a new model, the architecture comes from `[model.HyraxParsnip.settings]`, which accepts any key from
  `parsnip/settings.py`. Reusing a Hyrax `.pth` checkpoint therefore needs the same settings (or the same
  `pretrained`) as the run that produced it.

## SuperNNova (`hyrax_snn`)

`hyrax_snn` runs SuperNNova classifiers through Hyrax, reading catalogs with the same dataset and
settings as ParSNIP. By default no training is needed: the pretrained models are complete classifiers trained by the
[Fink](https://fink-broker.org) broker on ELAsTiCC (Fraga et al. 2024,
[arXiv:2404.08798](https://arxiv.org/abs/2404.08798)), published in
[fink-science](https://github.com/astrolabsoftware/fink-science) (Apache-2.0). They are downloaded from a
pinned commit on first use (`~/.cache/hyrax_snn/`); run once on a machine with internet access (e.g. a NERSC
login node) before using compute nodes.

| model | redshift | classes |
|---|---|---|
| `elasticc_ia` | no | SNIa, other |
| `SN_vs_other` | no | SN, other |
| `elasticc_broad` | yes (+ MWEBV) | SN, Fast, Long, Periodic, NonPeriodic |
| `Fast_vs_other`, `Long_vs_other`, `Periodic_vs_other`, `NonPeriodic_vs_other` | yes (+ MWEBV) | X, other |

```python
from hyrax import Hyrax
import hyrax_snn

h = Hyrax()
settings = dict(hyrax_snn.RUBIN_DIA_SETTINGS)          # Rubin DIA columns, fluxes kept in nJy
# settings["redshift_column"] = "redshift"             # for models with redshift
hyrax_snn.configure(h, catalog, pretrained="elasticc_ia", dataset_settings=settings)
predictions = hyrax_snn.load_predictions(h.infer())    # object_id, one probability per class, predicted_class
```

The preprocessing is a numpy port of SuperNNova's on-the-fly classification (SuperNNova pins pandas < 3, so
it is not a dependency) and its `VanillaRNN` is included (MIT). Tests check that probabilities match
SuperNNova's `classify_lcs` to 1e-4.

How the input is prepared matters, and is configurable under `[model.HyraxSNN]`:
- **Flux units.** The models behave as if trained on nJy (the alert stream). In our tests on simulated SNe,
  `elasticc_ia` separated SN Ia much better with nJy input than with the zp-27.5 fluxes Fink's own processor
  sends, so `RUBIN_DIA_SETTINGS` keeps nJy. Models with `cosmo_quantile` normalization (the ones with
  redshift) don't depend on units.
- **`time_window`** (default `[-30, 100]`): use observations within this many days of the max-S/N point, like
  the window the models were trained with.
- **`detection_snr`** (default off): keep only points above this S/N, like the alert detections Fink feeds its
  models, instead of the full forced photometry.
- **`fix_clipped_norm_min`** (default off): SuperNNova stores the flux normalization minimum clipped to −2000
  while its mean/std used the true minimum; the models were trained with the clipped value, so this is off
  by default.
- `hyrax_snn.FINK_EXACT` (+ `flux_scale = FINK_EXACT_FLUX_SCALE`) approximates Fink's own processing, for
  comparison (`classify_rubin_dia.py --method snn --fink-exact`).

### Training SuperNNova (optional)

No pretrained SuperNNova models exist for PLAsTiCC, so to compare with ParSNIP on the same training data,
`hyrax_snn` can also train one with Hyrax (`h.train()`):

```bash
python examples/snn_train.py data/plasticc_hats [--output examples/models/snn_plasticc] [--classes {ia,dp2,all}] \
    [--redshift-column redshift] [--epochs 90] [--validate-fraction 0.1]
```

- SuperNNova's default network and training: bi-LSTM (32 x 2), dropout 0.05, Adam (lr 1e-3, weight decay
  1e-7), cross-entropy weighted by inverse class frequency, random-length truncation, learning rate reduced on
  plateau, 90 epochs, batch 128.
- "global" normalization (log-standardization shared by all bands' fluxes, and by their errors) from the
  training light curves. As in SuperNNova, the flux minimum is floored at −2000 (zeropoint 27.5), so bright
  variable stars don't squash all supernova fluxes to one value; unlike SuperNNova, the mean and std use the
  same floored minimum, so training and inference match exactly.
- A held-out fraction (`--validate-fraction`, default 0.1) drives the learning rate (reduced 10x when the
  validation loss stops improving) and the export: the weights of the epoch with the lowest validation loss
  are kept, as in SuperNNova. Its accuracy and recall per class are printed at the end.
- Classes from the catalog's `type` column: `ia` (default, SNIa vs non-SNIa), `dp2` (SNIa/SNII/SNIbc/other),
  `all`. No redshift by default; `--redshift-column` trains a model that takes it.
- `--resume latest` continues an interrupted run (e.g. a batch job that hit its time limit) from the last
  Hyrax checkpoint in `--results-dir` (or `--resume CHECKPOINT.pt` / a train run directory): weights, optimizer,
  learning-rate schedule and validation history are restored; `--epochs` is the total. The model directory,
  split and batch size must be the same. Hyrax checkpoints each epoch before validating it, so that epoch's
  validation is lost (recorded as `null`; its weights can't be picked as the best).
- About 6 s per epoch for the PLAsTiCC training set on an Apple M-series CPU, so roughly an hour for the
  training set + DDF (40,774 objects) and 90 epochs. `nersc/snn_train.sbatch` runs it at NERSC.

The result is an ordinary SuperNNova model directory (`cli_args.json`, `data_norm.json`, `model.pt`), used
like the built-in names: `pretrained="<dir>"`, or `classify_rubin_dia.py --method snn --model <dir>`.
`cli_args.json` also records the class names, class weights, a training summary and `flux_zeropoint` (27.5
for PLAsTiCC); `hyrax_snn.rubin_dia_settings(model)` gives `RUBIN_DIA_SETTINGS` with the matching flux scale
(nJy rescaled to zeropoint 27.5 for such a model, nJy for the Fink models). In Python:

```python
from hyrax_snn import configure, new_model_dir, training_dataset, SNN_BAND_MAP

h = Hyrax()
h.set_config("split.train", 0.9); h.set_config("split.validate", 0.1)
settings = {"label_column": "type", "label_scheme": "ia", "mwebv_column": "mwebv"}   # + catalog columns
dataset = training_dataset(h, catalog, settings)
new_model_dir("snn_plasticc", dataset, list(SNN_BAND_MAP.values()), h.config["model"]["HyraxSNN"], flux_zeropoint=27.5)
configure(h, catalog, pretrained="snn_plasticc", groups=("train", "validate"), dataset_settings=settings)
h.train().export_snn("snn_plasticc")
```

On our small toy simulation the Fink models performed poorly whichever way the input was prepared (e.g.
`elasticc_ia` AUC 0.3–0.75, against 0.95 for ParSNIP with the PLAsTiCC classifier); validate on the DP2
simulation before relying on them.

## Examples

Shared by both classifiers:
- `examples/classify_rubin_dia.py --method {parsnip,snn}`: classify a Rubin DIA catalog (default: the EDP2
  catalog). `--model` picks the pretrained model (snn: also a model directory from `snn_train.py`), `--redshift-column` is for models with redshift input;
  parsnip: `--classifier` for class probabilities; snn: `--fink-exact` and the input options.
- `examples/evaluate.py`: score predictions from either method against a simulated catalog's truth
  (SN Ia vs non-Ia confusion matrix, or predicted class per true type for models without an SNIa class; recall
  vs S/N and redshift, P(target) and ROC, light curves; predicted vs true redshift and latent space for ParSNIP).
  Probabilities are used as given; `--kfold-lightgbm` trains ParSNIP's classifier on the catalog instead, as a
  quick check only.
- `examples/demo_workflow.ipynb`: the workflow for both models on the DP2 simulation, with SN Ia vs rest side
  by side. It loads `data/dp2_sim_sne` if it exists and simulates otherwise, and trains the PLAsTiCC classifier
  if `examples/models/plasticc_classifier.pkl` doesn't exist yet (training set + DDF, ~35 min;
  `DEMO_PLASTICC_DDF=0` for the training set only, ~7 min). `DEMO_CATALOG` / `DEMO_CLASSIFIER` override the
  paths. Needs the `notebook` extra (and lightcurvelynx to simulate).
  It also shows a SuperNNova trained on PLAsTiCC if `examples/models/snn_plasticc/` exists (`DEMO_SNN_MODEL`
  overrides the path); it doesn't train one.
- `examples/simulate_dp2.py`: the DP2 validation simulation, e.g.
  `python examples/simulate_dp2.py data/dp2_sim_sne --n-per-class 5000` for a larger set.

ParSNIP only:
- `examples/plasticc_to_hats.py`: download PLAsTiCC and write a labeled HATS catalog (`--ddf` adds the
  test-set DDF objects).
- `examples/parsnip_train_classifier.py`: train ParSNIP's LightGBM classifier on a labeled catalog; writes the
  classifier, its `.json` and K-fold results.
- `examples/parsnip_train_then_infer.py`: train or fine-tune ParSNIP itself, export, then run inference.

SuperNNova only:
- `examples/snn_train.py`: train SuperNNova on a labeled catalog (optional; see above).

## Notes

- **Device.** On Apple silicon, Hyrax picks MPS, which neither model is tested on. Both keep the model on the
  CPU by default (`model.HyraxParsnip.device` / `model.HyraxSNN.device = "cpu"`); set `"hyrax"` to follow
  Hyrax's device choice (e.g. CUDA).
- **LightGBM threads** (ParSNIP classifier). On macOS, torch and LightGBM load separate OpenMP runtimes, and
  multithreaded LightGBM segfaults once torch has run. The classifier helpers therefore run LightGBM
  single-threaded (`n_jobs=1`). ParSNIP feature tables are small, so this is cheap.
- **Compatibility shims.** ParSNIP 1.4.3 passes `verbose=` to `LGBMClassifier.fit`, which LightGBM ≥ 4
  removed; `train_classifier` strips it. SuperNNova's pandas-based preprocessing is replaced by the numpy port.

## Tests

```bash
.venv/bin/pytest
```

The tests build a small synthetic nested HATS catalog and cover:
- the dataset, including Rubin-style columns, flux scaling and flags
- ParSNIP: pretrained inference checked against native `parsnip.predict` (all objects, large batches), the
  photo-z model without redshifts, training from scratch, fine-tuning and export, and the classifier
  (including reloading it in a fresh process)
- SuperNNova: a synthetic model (features, batch-order invariance, with/without redshift), training, export
  and reuse as a model directory, and the Fink models
  against SuperNNova itself (`tests/test_snn_reference.py`; skipped unless `supernnova` is installed:
  `.venv/bin/pip install --no-deps supernnova==3.0.51 natsort seaborn`)
- the PLAsTiCC converter (`examples/plasticc_to_hats.py`)
