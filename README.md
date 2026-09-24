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

## Install

```bash
python -m venv .venv
.venv/bin/pip install -e ".[dev]"
```

On macOS, LightGBM (required by ParSNIP) needs the OpenMP runtime: `brew install libomp`.

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

## Examples

- `examples/demo_workflow.ipynb`: end-to-end walkthrough on a simulated catalog (SN Ia / II-P / Ib/c from
  sncosmo): HATS catalog → pretrained inference → latent space → classifier → optional fine-tuning.
  Install the extras with `.venv/bin/pip install -e ".[notebook]"`.
- `examples/pretrained_inference.py`: HATS → latents → (train or load) classifier → probabilities.
- `examples/train_then_infer.py`: train or fine-tune, export, then run inference.

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
- `plasticc_photoz` (photometric-redshift model) needs host photo-z inputs that the
  dataset does not provide yet.

## Tests

```bash
.venv/bin/pytest
```

The tests build a small synthetic nested HATS catalog and cover:
- the dataset
- pretrained inference, checked against native `parsnip.predict`
- training from scratch, fine-tuning, and export
- the classifier, including reloading it in a fresh process
