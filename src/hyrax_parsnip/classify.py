"""Classification of ParSNIP predictions produced by ``h.infer()``.

ParSNIP ships pretrained autoencoders but no pretrained classifier. Train one once
on labeled data with `train_classifier`, save it with ``Classifier.write`` and
reload it with `load_classifier` to classify new inference results without
retraining.
"""

import ctypes
import inspect
import json
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from astropy.table import Table, join

from hyrax_parsnip.model import FEATURE_NAMES_FILENAME


def load_predictions(results, metadata: Table | None = None) -> Table:
    """Turn Hyrax inference results into a ParSNIP-style predictions table.

    Parameters
    ----------
    results : hyrax ResultDataset
        The return value of ``h.infer()``. For an older run, use
        ``hyrax.datasets.result_factories.load_results_dataset(h.config, results_dir)``.
    metadata : astropy Table, optional
        Extra per-object columns (e.g. labels from `hyrax_lightcurves.catalog_metadata`) joined on
        ``object_id``.

    Returns
    -------
    Table
        One row per object: ``object_id``, ``original_object_id`` (needed by
        ``parsnip.Classifier`` for K-folding) and every ParSNIP feature column.
    """
    with open(Path(results.data_location) / FEATURE_NAMES_FILENAME) as f:
        names = json.load(f)

    object_ids = np.asarray(results.ids(), dtype=str)
    predictions = Table(np.asarray(results.__get_all__()), names=names)
    predictions.add_column(object_ids, name="object_id", index=0)
    predictions.add_column(object_ids, name="original_object_id", index=1)

    if metadata is not None:
        predictions = join(predictions, metadata, keys="object_id", join_type="left")
    return predictions


def valid_mask(predictions: Table) -> np.ndarray:
    """Rows ParSNIP could process (``infer_batch`` fills unprocessable rows with NaN)."""
    return np.isfinite(np.asarray(predictions["s1"], dtype=np.float64))


@contextmanager
def _lightgbm_compat(n_jobs: int):
    """Adapt LightGBM to parsnip.Classifier.train while it runs.

    - LightGBM >= 4 removed the ``verbose`` argument of ``LGBMClassifier.fit`` that
      ParSNIP passes; drop it.
    - Force ``n_jobs``. On macOS, torch and LightGBM each load their own OpenMP
      runtime, and multithreaded LightGBM segfaults once torch has run.
    """
    import lightgbm

    original_fit = lightgbm.LGBMClassifier.fit
    accepts_verbose = "verbose" in inspect.signature(original_fit).parameters

    def fit(self, *args, verbose=None, **kwargs):
        self.set_params(n_jobs=n_jobs)
        if accepts_verbose:
            kwargs["verbose"] = verbose
        return original_fit(self, *args, **kwargs)

    lightgbm.LGBMClassifier.fit = fit
    try:
        yield
    finally:
        lightgbm.LGBMClassifier.fit = original_fit


def _limit_lightgbm_threads(n_jobs: int):
    """Cap LightGBM's threads process-wide, including model loading, which ignores
    ``n_jobs`` (see `_lightgbm_compat` for why). -1 restores the default."""
    import lightgbm

    set_max_threads = getattr(lightgbm.basic._LIB, "LGBM_SetMaxThreads", None)
    if set_max_threads is not None:
        set_max_threads(ctypes.c_int(n_jobs))


def _set_n_jobs(classifier, n_jobs: int):
    _limit_lightgbm_threads(n_jobs)
    for model in classifier.classifiers:
        model.set_params(n_jobs=n_jobs)


def train_classifier(
    predictions: Table, label_column: str = "type", num_folds: int = 10, n_jobs: int = 1, **kwargs
):
    """Train a ``parsnip.Classifier`` (LightGBM) on labeled predictions.

    Rows ParSNIP could not process are skipped. Extra kwargs go to
    ``Classifier.train`` (e.g. ``min_child_weight``, ``target_label``).

    ``n_jobs`` is LightGBM's thread count. Keep it at 1 on macOS (see
    `_lightgbm_compat`); ParSNIP feature tables are small, so this is cheap.

    Returns
    -------
    classifier : parsnip.Classifier
    out_of_sample : Table
        K-fold out-of-sample class probabilities for the training objects.
    """
    import parsnip

    _limit_lightgbm_threads(n_jobs)
    predictions = predictions[valid_mask(predictions)]
    classifier = parsnip.Classifier()
    with _lightgbm_compat(n_jobs):
        out_of_sample = classifier.train(
            predictions,
            num_folds=num_folds,
            labels=np.asarray(predictions[label_column]).astype(str),
            **kwargs,
        )
    return classifier, out_of_sample


def load_classifier(path, n_jobs: int = 1):
    """Load a classifier saved with ``Classifier.write``.

    ``n_jobs`` is LightGBM's thread count (see `train_classifier`).
    """
    import parsnip

    _limit_lightgbm_threads(n_jobs)
    classifier = parsnip.Classifier.load(str(path))
    _set_n_jobs(classifier, n_jobs)
    return classifier


def classify(classifier, predictions: Table, n_jobs: int = 1) -> Table:
    """Class probabilities for each object; NaN for objects ParSNIP could not process.

    ``n_jobs`` overrides the LightGBM thread count stored in the classifier (see
    `train_classifier`).
    """
    _set_n_jobs(classifier, n_jobs)
    mask = valid_mask(predictions)
    probabilities = Table({"object_id": predictions["object_id"]})
    for name in classifier.class_names:
        probabilities[str(name)] = np.full(len(predictions), np.nan)

    if mask.any():
        classified = classifier.classify(predictions[mask])
        for name in classifier.class_names:
            probabilities[str(name)][mask] = classified[str(name)]
    return probabilities
