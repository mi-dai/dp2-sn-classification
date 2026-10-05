# hyrax-parsnip

Run [ParSNIP](https://parsnip.readthedocs.io) transient models through
[Hyrax](https://hyrax.readthedocs.io) on light curves stored in a
[HATS](https://hats.readthedocs.io) catalog.

- **Inference with pretrained models, no training needed.** Use the built-in ParSNIP models
  (`plasticc`, `ps1`) or any ParSNIP `.pt` file to get latent representations
  (`s1..sN`, color, amplitude, luminosity, ...) for every object.
- **Optional training.** Train from scratch or fine-tune a pretrained model with
  `h.train()`, and export the result back to ParSNIP's native format.
- **Classification.** Train ParSNIP's LightGBM classifier on labeled
  predictions, save it, and reuse it on new inference results.
- **SuperNNova (`hyrax_snn`).** Run Fink's pretrained SuperNNova classifiers on the same
  catalogs, with or without redshifts. Inference only; see [SuperNNova](#supernnova-hyrax_snn).

## Install

```bash
python -m venv .venv
.venv/bin/pip install -e ".[dev]"
```

On macOS, LightGBM (required by ParSNIP) needs the OpenMP runtime: `brew install libomp`.

At NERSC, build the venv on top of `desc-td-env`, which provides lightcurvelynx for the demo's simulations:

```bash
/global/common/software/lsst/install/td_env/2026-09-03-32-23/py/envs/td_env/bin/python \
    -m venv --system-site-packages .venv
.venv/bin/pip install -e ".[notebook,dev]"
.venv/bin/python -m ipykernel install --user --name hyrax-parsnip-td --display-name "hyrax-parsnip (td_env)"
```

## Catalog format

A nested HATS catalog with one row per object, e.g. as produced by LSDB / nested-pandas:

| column       | type                                          |
|--------------|-----------------------------------------------|
| `object_id`  | id                                            |
| `redshift`   | float (required by `plasticc` / `ps1` models) |
| `lightcurve` | nested: `mjd`, `flux`, `fluxerr`, `band`      |

All names are configurable under `[data_set.ParsnipHATSDataset]`, and so are the optional
`mwebv_column` and `label_column`. `band_map` maps catalog band names to the sncosmo
bands the model knows. The default is LSST `ugrizy` → `lsstu`…`lssty`, matching the
`plasticc` model. See [`default_config.toml`](src/hyrax_parsnip/default_config.toml).

## Inference with a pretrained model

```python
from hyrax import Hyrax
import hyrax_parsnip

h = Hyrax()
hyrax_parsnip.configure(h, "/path/to/hats_catalog", pretrained="plasticc")

results = h.infer()                                   # no training run needed
predictions = hyrax_parsnip.load_predictions(results) # astropy Table, one row per object
```

`configure` points `infer.model_weights_file` at the pretrained ParSNIP file. Objects
that ParSNIP cannot process (e.g. no observations in the model's time window) get NaN
features rather than being dropped, so object ids stay aligned.

## Without redshifts

The `plasticc` and `ps1` models need a redshift for every object. `plasticc_photoz` predicts it
from the light curve instead:

```python
hyrax_parsnip.configure(h, catalog, pretrained="plasticc_photoz")  # also sets require_redshift = false
h.set_config("data_set.ParsnipHATSDataset.redshift_column", False)
predictions = hyrax_parsnip.load_predictions(h.infer())             # includes predicted_redshift
```

The model was trained with a host-galaxy photo-z as an encoder input. Every object gets
`model.HyraxParsnip.photoz` / `photoz_error` (default 0.5 ± 1.0), a weak prior. On simulated
SN Ia/II-P/Ib/c at z = 0.03–0.2, this gave Δz/(1+z) = +0.075 ± 0.068 (NMAD), and classification
accuracy went from 99.7% (true z) to 98%. Hyrax logs a warning about NaN inputs (the missing
redshifts); the photo-z model ignores them.

### Rubin DIA catalogs

`examples/classify_rubin_dia.py --method parsnip` runs `plasticc_photoz` on a Rubin HATS catalog with nested
`diaObjectForcedSource` photometry: `psfDiffFlux` in nJy (rescaled with `flux_scale`) with flagged
points dropped (`flag_columns`). It writes one row per `diaObjectId`, and adds class probabilities
when you pass `--classifier`. These dataset settings are available as `hyrax_parsnip.RUBIN_DIA_SETTINGS`:

```python
for key, value in hyrax_parsnip.RUBIN_DIA_SETTINGS.items():
    h.set_config(f"data_set.ParsnipHATSDataset.{key}", value)
```

## Training (optional)

```python
h.set_config("train.epochs", 50)
h.set_config("data_loader.batch_size", 128)
hyrax_parsnip.configure(h, catalog, pretrained=False,       # or "plasticc" to fine-tune
                        groups=("train", "infer"), model_weights_file=False)

model = h.train()
model.export_parsnip("my_model.pt")   # usable with parsnip.load_model / pretrained="my_model.pt"
results = h.infer()                    # uses the latest training run
```

- Training uses ParSNIP's own loss, optimizer, augmentation, and ReduceLROnPlateau
  schedule. Hyrax's `[optimizer]`/`[criterion]`/`[scheduler]` settings are ignored
  (Hyrax logs a warning about this).
- A Hyrax epoch is one pass over the catalog. ParSNIP's `fit` counts ~25,000 augmented
  light curves as an epoch, so small catalogs need more Hyrax epochs.
- For a new model, the architecture comes from `[model.HyraxParsnip.settings]`, which
  accepts any key from `parsnip/settings.py`. Reusing a Hyrax `.pth` checkpoint therefore
  needs the same settings (or the same `pretrained`) as the run that produced it.

## Classification

ParSNIP does not ship a pretrained classifier. Train one once on labeled data (e.g. the
PLAsTiCC training set), save it, and reuse it:

```python
labels = hyrax_parsnip.catalog_metadata(catalog, ["type"])
predictions = hyrax_parsnip.load_predictions(h.infer(), metadata=labels)

classifier, out_of_sample = hyrax_parsnip.train_classifier(predictions, label_column="type")
classifier.write("classifier.pkl")

# Later, on new data:
classifier = hyrax_parsnip.load_classifier("classifier.pkl")
probabilities = hyrax_parsnip.classify(classifier, new_predictions)
```

ParSNIP's default LightGBM `min_child_weight=1000` suits PLAsTiCC-sized training sets. With only a few
hundred labeled objects it prevents every split, and all probabilities come out equal. In that case pass a
smaller value, e.g. `train_classifier(..., min_child_weight=10.0)`.

## SuperNNova (`hyrax_snn`)

`hyrax_snn` runs pretrained [SuperNNova](https://supernnova.readthedocs.io) classifiers
through Hyrax, reading catalogs with the same dataset and settings as ParSNIP. There is no
training: the models are complete classifiers trained by the [Fink](https://fink-broker.org)
broker on ELAsTiCC (Fraga et al. 2024, [arXiv:2404.08798](https://arxiv.org/abs/2404.08798)),
published in [fink-science](https://github.com/astrolabsoftware/fink-science) (Apache-2.0).
They are downloaded from a pinned commit on first use (`~/.cache/hyrax_snn/`); run once on a
machine with internet access (e.g. a NERSC login node) before using compute nodes.

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

The preprocessing is a numpy port of SuperNNova's on-the-fly classification (SuperNNova
pins pandas < 3, so it is not a dependency) and its `VanillaRNN` is included (MIT). Tests
check that probabilities match SuperNNova's `classify_lcs` to 1e-4.

How the input is prepared matters, and is configurable under `[model.HyraxSNN]`:
- **Flux units.** The models behave as if trained on nJy (the alert stream). In our tests on
  simulated SNe, `elasticc_ia` separated SN Ia much better with nJy input than with the
  zp-27.5 fluxes Fink's own processor sends, so `RUBIN_DIA_SETTINGS` keeps nJy. Models with
  `cosmo_quantile` normalization (the ones with redshift) don't depend on units.
- **`time_window`** (default `[-30, 100]`): use observations within this many days of the
  max-S/N point, like the window the models were trained with.
- **`detection_snr`** (default off): keep only points above this S/N, like the alert
  detections Fink feeds its models, instead of the full forced photometry.
- **`fix_clipped_norm_min`** (default off): SuperNNova stores the flux normalization minimum
  clipped to −2000 while its mean/std used the true minimum; the models were trained with
  the clipped value, so this is off by default.
- `hyrax_snn.FINK_EXACT` (+ `flux_scale = FINK_EXACT_FLUX_SCALE`) approximates Fink's own
  processing, for comparison.

On our small toy simulation the models performed poorly whichever way the input was
prepared (e.g. `elasticc_ia` AUC 0.3–0.75); validate on the DP2 simulation before relying
on them.

## Examples

- `examples/parsnip_demo_workflow.ipynb`: end-to-end walkthrough on SN Ia / II / Ib/c simulated with lightcurvelynx on the
  real DP2 visits and written in the DP2 DIA catalog schema: HATS catalog → pretrained inference (with and
  without redshifts) → latent space → classifier → optional fine-tuning. Needs lightcurvelynx (see Install for
  NERSC) and the notebook extras (`.venv/bin/pip install -e ".[notebook]"`).
- `examples/simulate_dp2.py`: the demo's simulations as a script, e.g.
  `python examples/simulate_dp2.py data/dp2_sim_sne --n-per-class 5000` for a larger training set.
- `examples/classify_rubin_dia.py --method {parsnip,snn}`: classify a Rubin DIA catalog with ParSNIP or
  SuperNNova (`--model` for the pretrained model, `--redshift-column` for models with redshift input; parsnip:
  `--classifier` for class probabilities; snn: `--fink-exact` and input options). Both write the same table:
  `diaObjectId`, `method`, `model`, `redshift_input`, `p_<class>`, `predicted_class` (plus ParSNIP's features).
- `examples/evaluate.py`: score predictions from either method on a simulated catalog against the truth
  (confusion matrix or predicted class per true type, recall vs S/N and redshift, P(target) and ROC, light
  curves; predicted vs true redshift and latent space for ParSNIP). The simulation is validation-only:
  probabilities are used as given; `--kfold-lightgbm` trains ParSNIP's classifier on the catalog instead, as a
  quick check. It warns when the model was given the redshift (`--redshift-column`).
- `examples/parsnip_train_then_infer.py`: train or fine-tune ParSNIP, export, then run inference.

```bash
python examples/classify_rubin_dia.py data/dp2_sim_sne --method snn --output snn.parquet
python examples/evaluate.py data/dp2_sim_sne snn.parquet --output-dir eval_snn
python examples/classify_rubin_dia.py data/dp2_sim_sne --method parsnip --classifier classifier.pkl --output parsnip.parquet
python examples/evaluate.py data/dp2_sim_sne parsnip.parquet --output-dir eval_parsnip
```

## Notes

- **Device.** On Apple silicon, Hyrax picks MPS, which ParSNIP is not tested on. By
  default the wrapper keeps ParSNIP on the CPU (`model.HyraxParsnip.device = "cpu"`).
  Set `device = "hyrax"` to follow Hyrax's device choice (e.g. CUDA).
- **LightGBM threads.** On macOS, torch and LightGBM load separate OpenMP runtimes, and
  multithreaded LightGBM segfaults once torch has run. The classifier helpers therefore
  run LightGBM single-threaded (`n_jobs=1`). ParSNIP feature tables are small, so this
  is cheap.
- **Compatibility shims.** ParSNIP 1.4.3 passes `verbose=` to `LGBMClassifier.fit`,
  which LightGBM ≥ 4 removed. `train_classifier` strips it while training.

## Tests

```bash
.venv/bin/pytest
```

The tests build a small synthetic nested HATS catalog and cover:
- the dataset
- pretrained inference, checked against native `parsnip.predict`
- training from scratch, fine-tuning, and export
- the classifier, including reloading it in a fresh process
- `hyrax_snn` with a synthetic model (features, batch-order invariance, with/without redshift)
- `hyrax_snn` against SuperNNova itself with the Fink models (`tests/test_snn_reference.py`;
  skipped unless `supernnova` is installed: `.venv/bin/pip install --no-deps supernnova==3.0.51 natsort seaborn`)
