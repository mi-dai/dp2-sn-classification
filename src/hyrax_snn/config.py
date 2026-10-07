from pathlib import Path

from hyrax_lightcurves.config import DATASET_CLASS, MODEL_FIELDS, NJY_TO_ZP27_5
from hyrax_lightcurves.config import RUBIN_DIA_SETTINGS as LIGHTCURVE_RUBIN_DIA_SETTINGS
from hyrax_snn.pretrained import resolve

MODEL_NAME = "hyrax_snn.model.HyraxSNN"

# SuperNNova's LSST filter names; Rubin catalogs use "y".
SNN_BAND_MAP = {"u": "u", "g": "g", "r": "r", "i": "i", "z": "z", "y": "Y"}

# `[data_set.LightCurveHATSDataset]` settings for Rubin DIA catalogs with SuperNNova: the
# Rubin DIA columns and flags, with fluxes kept in nJy. The Fink ELAsTiCC models
# were trained on the alert stream (nJy) and, in our tests, separate SN Ia much better
# with nJy input than with Fink's zp-27.5 conversion. Models with `cosmo_quantile`
# normalization rescale each light curve and don't depend on the units.
RUBIN_DIA_SETTINGS = {**LIGHTCURVE_RUBIN_DIA_SETTINGS, "flux_scale": 1.0, "band_map": SNN_BAND_MAP}

# Model settings that approximate Fink's own Rubin processor (fink_science/rubin/snn):
# alert detections only (S/N > 5 stands in for diaSource + prvDiaSources), no time
# window, clipped normalization minimum. Pair with flux_scale = FINK_EXACT_FLUX_SCALE (zp 27.5).
FINK_EXACT = {"fix_clipped_norm_min": False, "time_window": False, "detection_snr": 5.0}
FINK_EXACT_FLUX_SCALE = NJY_TO_ZP27_5


def configure(
    h,
    catalog_path: str | Path,
    *,
    pretrained: str = "elasticc_ia",
    groups: tuple[str, ...] = ("infer",),
    dataset_settings: dict | None = None,
):
    """Point a Hyrax instance at a HATS catalog and a SuperNNova model (pretrained or to train).

    Parameters
    ----------
    h : hyrax.Hyrax
        Hyrax instance to configure.
    catalog_path : str or Path
        HATS catalog with a nested light-curve column.
    pretrained : str
        Built-in model name (see `hyrax_snn.PRETRAINED_MODELS`) or a SuperNNova model
        directory. Built-in models are downloaded on first use.
    groups : tuple of str
        data_request groups to create; "train" / "validate" also request ``label_index``
        (set ``label_column`` and ``label_scheme`` in `dataset_settings`).
    dataset_settings : dict, optional
        `[data_set.LightCurveHATSDataset]` overrides, e.g. `RUBIN_DIA_SETTINGS` plus a
        `redshift_column`. They are applied after the SuperNNova band map.
    """
    # Setting model.name first makes Hyrax merge hyrax_snn/default_config.toml.
    h.set_config("model.name", MODEL_NAME)
    h.set_config("model.HyraxSNN.pretrained", str(pretrained))

    data_request = {
        group: {
            "data": {
                "dataset_class": DATASET_CLASS,
                "data_location": str(catalog_path),
                # Training groups also need the class index of each object.
                "fields": list(MODEL_FIELDS) + (["label_index"] if group in ("train", "validate") else []),
                "primary_id_field": "object_id",
            }
        }
        for group in groups
    }
    h.set_config("data_request", data_request)
    h.set_config("data_set.LightCurveHATSDataset.band_map", SNN_BAND_MAP)
    for key, value in (dataset_settings or {}).items():
        h.set_config(f"data_set.LightCurveHATSDataset.{key}", value)

    settings = h.config["model"]["HyraxSNN"]
    snn = resolve(str(pretrained), cache_dir=settings["cache_dir"] or None)
    # Models that take a redshift input need one per object; the others don't.
    h.set_config("data_set.LightCurveHATSDataset.require_redshift", snn.redshift != "none")
    # A new model (no model.pt yet) is trained first; inference then uses the latest training run.
    weights = snn.model_dir / "model.pt"
    h.set_config("infer.model_weights_file", str(weights) if weights.exists() else False)
    return h


def rubin_dia_settings(pretrained: str = "elasticc_ia", cache_dir=None) -> dict:
    """`RUBIN_DIA_SETTINGS` with the flux scale `pretrained` expects: nJy for the Fink models, and
    nJy rescaled to the model's ``flux_zeropoint`` for models trained with hyrax_snn on other
    fluxes (e.g. zp 27.5 for PLAsTiCC)."""
    settings = dict(RUBIN_DIA_SETTINGS)
    zeropoint = resolve(str(pretrained), cache_dir=cache_dir).flux_zeropoint
    if zeropoint is not None:
        settings["flux_scale"] = 10 ** (-0.4 * (31.4 - float(zeropoint)))
    return settings
