from pathlib import Path

from hyrax_lightcurves.config import DATASET_CLASS
from hyrax_lightcurves.config import RUBIN_DIA_SETTINGS as LIGHTCURVE_RUBIN_DIA_SETTINGS
from hyrax_scone.training import load_model_info, weights_file

MODEL_NAME = "hyrax_scone.model.HyraxSCONE"
SCONE_DATASET_CLASS = "hyrax_scone.dataset.SconeHATSDataset"

# SCONE's LSST band names (keys of hyrax_scone.heatmaps.LSST_WAVELENGTHS); Rubin catalogs use "y".
SCONE_BAND_MAP = {"u": "u", "g": "g", "r": "r", "i": "i", "z": "z", "y": "Y"}

# `[data_set.LightCurveHATSDataset]` settings for Rubin DIA catalogs with SCONE: the Rubin DIA
# columns and flags, with nJy fluxes converted to zeropoint 27.5 (SCONE's FLUXCAL units, and
# PLAsTiCC's). `rubin_dia_settings` gives the scale for a model trained on other fluxes.
RUBIN_DIA_SETTINGS = {
    **LIGHTCURVE_RUBIN_DIA_SETTINGS,
    "flux_scale": 10 ** (-0.4 * (31.4 - 27.5)),
    "band_map": SCONE_BAND_MAP,
}


def configure_dataset(h, catalog_path, groups, dataset_settings: dict | None = None, require_redshift=False):
    """Point `h` at a HATS catalog served by `SconeHATSDataset` (with hyrax_scone's defaults merged).

    `dataset_settings` are ``[data_set.LightCurveHATSDataset]`` overrides, plus ``"scone"``: a dict of
    ``[data_set.SconeHATSDataset]`` overrides (heatmap settings, ``cache_dir``, ``n_jobs``).
    """
    # Setting model.name first makes Hyrax merge hyrax_scone/default_config.toml, and referencing the
    # shared dataset class merges hyrax_lightcurves' (SconeHATSDataset reads both sections).
    h.set_config("model.name", MODEL_NAME)
    h.set_config(
        "data_request",
        {"train": {"data": {"dataset_class": DATASET_CLASS, "data_location": str(catalog_path),
                            "primary_id_field": "object_id"}}},
    )
    data_request = {
        group: {
            "data": {
                "dataset_class": SCONE_DATASET_CLASS,
                "data_location": str(catalog_path),
                # Training groups also need the class index of each object.
                "fields": ["heatmap", "redshift"] + (["label_index"] if group in ("train", "validate") else []),
                "primary_id_field": "object_id",
            }
        }
        for group in groups
    }
    h.set_config("data_request", data_request, over_write=True)
    # The heatmaps are already held in memory (or memory-mapped from the cache).
    h.set_config("data_set.use_cache", False)

    settings = dict(dataset_settings or {})
    scone = settings.pop("scone", {})
    h.set_config("data_set.LightCurveHATSDataset.band_map", SCONE_BAND_MAP)
    for key, value in settings.items():
        h.set_config(f"data_set.LightCurveHATSDataset.{key}", value)
    for key, value in scone.items():
        h.set_config(f"data_set.SconeHATSDataset.{key}", value)
    h.set_config("data_set.LightCurveHATSDataset.require_redshift", bool(require_redshift))
    return h


def configure(
    h,
    catalog_path: str | Path,
    *,
    model_dir: str | Path,
    groups: tuple[str, ...] = ("infer",),
    dataset_settings: dict | None = None,
):
    """Point a Hyrax instance at a HATS catalog and a SCONE model directory (trained, or new to train).

    Parameters
    ----------
    h : hyrax.Hyrax
        Hyrax instance to configure.
    catalog_path : str or Path
        HATS catalog with a nested light-curve column.
    model_dir : str or Path
        SCONE model directory (``scone_model.json``; see `hyrax_scone.new_model_dir`). Its heatmap
        settings and band map are used for the catalog, so heatmaps are made as in training.
    groups : tuple of str
        data_request groups to create; "train" / "validate" also request ``label_index``
        (set ``label_column`` and ``label_scheme`` in `dataset_settings`).
    dataset_settings : dict, optional
        `[data_set.LightCurveHATSDataset]` overrides, e.g. `rubin_dia_settings(model_dir)`, and
        ``"scone"``: ``[data_set.SconeHATSDataset]`` overrides such as ``cache_dir`` and ``n_jobs``.
    """
    info = load_model_info(model_dir)
    settings = dict(dataset_settings or {})
    settings["scone"] = {**settings.get("scone", {}), **info["heatmap_settings"]}
    configure_dataset(h, catalog_path, groups, settings, require_redshift=info["with_z"])
    h.set_config("data_set.LightCurveHATSDataset.band_map", info["band_map"])
    h.set_config("model.HyraxSCONE.model_dir", str(model_dir))
    # A new model (no weights yet) is trained first; inference then uses the latest training run.
    weights = weights_file(model_dir)
    h.set_config("infer.model_weights_file", str(weights) if weights.exists() else False)
    return h


def rubin_dia_settings(model_dir) -> dict:
    """`RUBIN_DIA_SETTINGS` with Rubin nJy fluxes rescaled to the model's training zeropoint."""
    settings = dict(RUBIN_DIA_SETTINGS)
    zeropoint = load_model_info(model_dir).get("flux_zeropoint")
    if zeropoint is not None:
        settings["flux_scale"] = 10 ** (-0.4 * (31.4 - float(zeropoint)))
    return settings
