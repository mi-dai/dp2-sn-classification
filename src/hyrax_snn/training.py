"""New SuperNNova models for training with hyrax_snn.

A trained model is an ordinary SuperNNova model directory (``cli_args.json``,
``data_norm.json``, ``model.pt``), so it is used for inference exactly like the Fink
models (``pretrained=<directory>``). `new_model_dir` writes the first two from a labeled
catalog (read with `training_dataset`); training (``h.train()``) then fills ``model.pt``
via ``HyraxSNN.export_snn``.
"""

import json
from itertools import combinations
from pathlib import Path

import numpy as np

from hyrax_parsnip import ParsnipHATSDataset
from hyrax_parsnip.config import DATASET_CLASS
from hyrax_snn.config import MODEL_NAME, SNN_BAND_MAP
from hyrax_snn.features import build_features, select_observations
from hyrax_snn.pretrained import load_settings

HOST_FEATURES = ["HOSTGAL_PHOTOZ", "HOSTGAL_PHOTOZ_ERR", "HOSTGAL_SPECZ", "HOSTGAL_SPECZ_ERR"]


def training_dataset(h, catalog_path, dataset_settings: dict, redshift: str = "none") -> ParsnipHATSDataset:
    """The labeled catalog as `new_model_dir` reads it, with the SuperNNova band map and
    `dataset_settings` (which need ``label_column``; ``label_scheme`` maps the types to classes).
    Also merges hyrax_snn's default config into `h`; `hyrax_snn.configure` sets the rest."""
    h.set_config("model.name", MODEL_NAME)
    # Referencing the dataset class merges hyrax_parsnip's dataset defaults.
    data = {"dataset_class": DATASET_CLASS, "data_location": str(catalog_path), "primary_id_field": "object_id"}
    h.set_config("data_request", {"train": {"data": data}})
    for key, value in {"band_map": SNN_BAND_MAP, **dataset_settings}.items():
        h.set_config(f"data_set.ParsnipHATSDataset.{key}", value)
    h.set_config("data_set.ParsnipHATSDataset.require_redshift", redshift == "zspe")
    return ParsnipHATSDataset(h.config, data_location=str(catalog_path))


def feature_lists(filters: list[str], redshift: str = "none") -> dict:
    """SuperNNova's feature lists for `filters`, in the layout of the Fink models."""
    combos = sorted("".join(c) for n in range(1, len(filters) + 1) for c in combinations(filters, n))
    to_normalize = [f"FLUXCAL_{f}" for f in filters] + [f"FLUXCALERR_{f}" for f in filters] + ["delta_time"]
    training = to_normalize + ["MWEBV"] + combos
    if redshift == "zspe":
        training += ["HOSTGAL_SPECZ", "HOSTGAL_SPECZ_ERR"]
    return {
        "all_features": to_normalize + HOST_FEATURES + ["MWEBV"] + combos,
        "training_features": training,
        "training_features_to_normalize": to_normalize,
    }


def log_standardization(values: np.ndarray, min_clip: float = -2000.0) -> dict:
    """SuperNNova's log-standardization statistics, ``log(x - min + 1e-5)`` standardized.

    As in SuperNNova, the minimum is clipped at `min_clip` (in FLUXCAL units, zeropoint 27.5):
    otherwise the most negative flux of a bright variable star sets it, and supernova fluxes
    all map to nearly the same value. Unlike SuperNNova, the mean and std are computed with the
    clipped minimum (values below it clipped, as at inference), so they match the stored minimum.
    """
    arr_min = max(float(np.min(values)), min_clip)
    arr_log = np.log(np.clip(values, arr_min, None) - arr_min + 1e-5)
    return {"min": arr_min, "mean": float(arr_log.mean()), "std": float(arr_log.std())}


def new_model_dir(
    path,
    dataset,
    band_names,
    model_settings: dict,
    redshift: str = "none",
    flux_zeropoint: float | None = None,
    hidden_dim: int = 32,
    num_layers: int = 2,
    dropout: float = 0.05,
    bidirectional: bool = True,
    layer_type: str = "lstm",
    rnn_output_option: str = "mean",
):
    """Write ``cli_args.json`` and ``data_norm.json`` for a new model trained on `dataset`.

    Parameters
    ----------
    path : str or Path
        New model directory.
    dataset : ParsnipHATSDataset
        Labeled training data (``label_column`` set); its classes become the model's classes.
    band_names : list of str
        SuperNNova filter name of each dataset band index (the ``band_map`` values).
    model_settings : dict
        ``[model.HyraxSNN]`` settings; ``time_window``, ``detection_snr`` and ``redshift_error``
        select and build the features exactly as in training and inference.
    redshift : "none" or "zspe"
        Whether the model takes the redshift (and its error) as input.
    flux_zeropoint : float, optional
        Zeropoint of the training fluxes (e.g. 27.5 for PLAsTiCC), so inference can rescale
        (also scales the normalization's flux floor; 27.5 is assumed if not given).
    """
    if dataset.label_classes is None:
        raise ValueError("The training dataset needs a label_column")
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    band_names = np.asarray(band_names)
    filters = list(dict.fromkeys(str(b) for b in band_names))
    lists = feature_lists(filters, redshift)
    cli = {
        "model": "vanilla",
        "list_filters": filters,
        **lists,
        "norm": "global",
        "nb_classes": len(dataset.label_classes),
        "class_names": list(dataset.label_classes),
        "redshift": redshift,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "dropout": dropout,
        "bidirectional": bidirectional,
        "layer_type": layer_type,
        "rnn_output_option": rnn_output_option,
        "flux_zeropoint": flux_zeropoint,
        "trained_with": "hyrax_snn",
    }
    # Provisional identity normalization, to build the raw features with the final settings.
    identity = {f: {"min": 0.0, "mean": 0.0, "std": 1.0} for f in lists["training_features_to_normalize"]}
    (path / "cli_args.json").write_text(json.dumps(cli, indent=2))
    (path / "data_norm.json").write_text(json.dumps(identity, indent=2))
    settings = load_settings(path)

    time_window = tuple(model_settings["time_window"]) if model_settings["time_window"] else None
    detection_snr = float(model_settings["detection_snr"]) if model_settings["detection_snr"] else None
    columns = [settings.all_features.index(f) for f in settings.features_to_normalize]
    rows, used = [], []
    for i in range(len(dataset)):
        lc = dataset.get_lightcurve(i)
        observations = select_observations(
            lc[:, 0], lc[:, 1], lc[:, 2], band_names[lc[:, 3].astype(int)], time_window, detection_snr
        )
        redshift_value = float(dataset.get_redshift(i))
        if redshift == "zspe" and not np.isfinite(redshift_value):
            continue
        x = build_features(
            *observations, settings, mwebv=float(dataset.get_mwebv(i)),
            redshift=redshift_value if np.isfinite(redshift_value) else 0.0,
            redshift_error=float(model_settings["redshift_error"]), normalize_features=False,
        )
        if x is not None:
            rows.append(x[:, columns])
            used.append(i)
    if not rows:
        raise ValueError("No usable light curves in the training dataset")
    values = np.concatenate(rows)

    # "global" normalization: one set of statistics for all fluxes, one for all errors (as SuperNNova).
    n = len(filters)
    # SuperNNova's -2000 floor is for zeropoint 27.5; scale it to the zeropoint of the fluxes.
    min_clip = -2000.0 * 10 ** (0.4 * ((flux_zeropoint or 27.5) - 27.5))
    flux_stats = log_standardization(values[:, :n], min_clip)
    err_stats = log_standardization(values[:, n : 2 * n])
    time_stats = log_standardization(values[:, -1])
    norm = {f"FLUXCAL_{f}": flux_stats for f in filters}
    norm.update({f"FLUXCALERR_{f}": err_stats for f in filters})
    norm["delta_time"] = time_stats
    (path / "data_norm.json").write_text(json.dumps(norm, indent=2))

    # Class weights for the cross-entropy: inverse class frequency, mean 1.
    labels = np.array([dataset.get_label_index(i) for i in used])
    counts = np.bincount(labels, minlength=cli["nb_classes"]).astype(float)
    weights = np.where(counts > 0, counts.sum() / (len(counts) * np.maximum(counts, 1)), 0.0)
    cli["class_weights"] = [float(w) for w in weights]
    cli["training_counts"] = {name: int(c) for name, c in zip(cli["class_names"], counts)}
    (path / "cli_args.json").write_text(json.dumps(cli, indent=2))
    return load_settings(path)
