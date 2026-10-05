from pathlib import Path

from hyrax_parsnip.config import DATASET_CLASS, MODEL_FIELDS, NJY_TO_ZP27_5
from hyrax_parsnip.config import RUBIN_DIA_SETTINGS as PARSNIP_RUBIN_DIA_SETTINGS
from hyrax_snn.pretrained import resolve

MODEL_NAME = "hyrax_snn.model.HyraxSNN"

# SuperNNova's LSST filter names; Rubin catalogs use "y".
SNN_BAND_MAP = {"u": "u", "g": "g", "r": "r", "i": "i", "z": "z", "y": "Y"}

# `[data_set.ParsnipHATSDataset]` settings for Rubin DIA catalogs with SuperNNova: the same
# columns and flags as for ParSNIP, but fluxes stay in nJy. The Fink ELAsTiCC models
# were trained on the alert stream (nJy) and, in our tests, separate SN Ia much better
# with nJy input than with Fink's zp-27.5 conversion. Models with `cosmo_quantile`
# normalization rescale each light curve and don't depend on the units.
RUBIN_DIA_SETTINGS = {**PARSNIP_RUBIN_DIA_SETTINGS, "flux_scale": 1.0, "band_map": SNN_BAND_MAP}

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
    """Point a Hyrax instance at a HATS catalog and a pretrained SuperNNova model.

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
        data_request groups to create.
    dataset_settings : dict, optional
        `[data_set.ParsnipHATSDataset]` overrides, e.g. `RUBIN_DIA_SETTINGS` plus a
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
                "fields": list(MODEL_FIELDS),
                "primary_id_field": "object_id",
            }
        }
        for group in groups
    }
    h.set_config("data_request", data_request)
    h.set_config("data_set.ParsnipHATSDataset.band_map", SNN_BAND_MAP)
    for key, value in (dataset_settings or {}).items():
        h.set_config(f"data_set.ParsnipHATSDataset.{key}", value)

    settings = h.config["model"]["HyraxSNN"]
    snn = resolve(str(pretrained), cache_dir=settings["cache_dir"] or None)
    # Models that take a redshift input need one per object; the others don't.
    h.set_config("data_set.ParsnipHATSDataset.require_redshift", snn.redshift != "none")
    h.set_config("infer.model_weights_file", str(snn.model_dir / "model.pt"))
    return h
