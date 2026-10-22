"""SCONE model directories.

SCONE publishes no trained weights, so every model is trained here. A model directory holds
``scone_model.json`` (classes, redshift input, heatmap settings, flux zeropoint, class weights) and,
once trained, ``model.pt`` (the `SconeCNN` state dict). `new_model_dir` writes the first from a
labeled catalog read with `training_dataset`; training (``h.train()``) then fills ``model.pt`` via
``HyraxSCONE.export_scone``.
"""

import json
from pathlib import Path

import numpy as np

from hyrax_lightcurves.training import class_weights

MODEL_INFO_FILENAME = "scone_model.json"


def load_model_info(model_dir) -> dict:
    path = Path(model_dir) / MODEL_INFO_FILENAME
    if not path.exists():
        raise FileNotFoundError(f"{model_dir} is not a SCONE model directory (no {MODEL_INFO_FILENAME})")
    return json.loads(path.read_text())


def weights_file(model_dir) -> Path:
    return Path(model_dir) / "model.pt"


def training_dataset(h, catalog_path, dataset_settings: dict, with_z: bool = False):
    """The labeled catalog with its SCONE heatmaps, as `new_model_dir` reads it: SCONE's band map,
    the heatmap defaults and `dataset_settings` (``[data_set.LightCurveHATSDataset]`` keys, which need
    ``label_column``; ``label_scheme`` maps the types to classes; ``[data_set.SconeHATSDataset]`` keys
    such as ``cache_dir`` are passed under ``"scone"``). `hyrax_scone.configure` sets the rest."""
    from hyrax_scone.config import configure_dataset
    from hyrax_scone.dataset import SconeHATSDataset

    configure_dataset(h, catalog_path, ("train",), dataset_settings, require_redshift=with_z)
    return SconeHATSDataset(h.config, data_location=str(catalog_path))


def new_model_dir(
    path,
    dataset,
    with_z: bool = False,
    redshift_error: float = 0.0,
    flux_zeropoint: float | None = 27.5,
) -> dict:
    """Write ``scone_model.json`` for a new model trained on `dataset`.

    Parameters
    ----------
    path : str or Path
        New model directory.
    dataset : SconeHATSDataset
        Labeled training data (``label_column`` set); its classes become the model's classes and its
        heatmap settings are used at inference too.
    with_z : bool
        Whether the model takes the redshift (and `redshift_error`, the same for every object) as input.
    flux_zeropoint : float, optional
        Zeropoint of the training fluxes (27.5 for PLAsTiCC), so inference rescales other fluxes to it.
    """
    if dataset.label_classes is None:
        raise ValueError("The training dataset needs a label_column")
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    usable = dataset.has_heatmap()
    if with_z:
        usable &= np.isfinite([dataset.get_redshift(i) for i in range(len(dataset))])
    if not usable.any():
        raise ValueError("No usable light curves in the training dataset")
    labels = np.array([dataset.get_label_index(i) for i in np.flatnonzero(usable)])
    weights, counts = class_weights(labels, len(dataset.label_classes))

    info = {
        "class_names": list(dataset.label_classes),
        "with_z": with_z,
        "redshift_error": redshift_error,
        "flux_zeropoint": flux_zeropoint,
        "band_map": dict(dataset.settings["band_map"]),
        "heatmap_settings": dataset.heatmap_settings,
        "class_weights": weights,
        "training_counts": dict(zip(dataset.label_classes, counts)),
        "trained_with": "hyrax_scone",
    }
    (path / MODEL_INFO_FILENAME).write_text(json.dumps(info, indent=2))
    return info
