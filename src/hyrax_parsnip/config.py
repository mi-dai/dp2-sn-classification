from pathlib import Path

from hyrax_lightcurves.config import DATASET_CLASS, MODEL_FIELDS, NJY_TO_ZP27_5
from hyrax_lightcurves.config import RUBIN_DIA_SETTINGS as LIGHTCURVE_RUBIN_DIA_SETTINGS
from hyrax_parsnip.model import pretrained_model_path

MODEL_NAME = "hyrax_parsnip.model.HyraxParsnip"

# Catalog band -> sncosmo band understood by the ParSNIP models (set by `configure`).
PARSNIP_BAND_MAP = {"u": "lsstu", "g": "lsstg", "r": "lsstr", "i": "lssti", "z": "lsstz", "y": "lssty"}

# `[data_set.LightCurveHATSDataset]` settings for Rubin DIA catalogs with ParSNIP: the Rubin DIA
# columns and flags, with fluxes rescaled from nJy to the PLAsTiCC zeropoint (27.5).
RUBIN_DIA_SETTINGS = {**LIGHTCURVE_RUBIN_DIA_SETTINGS, "flux_scale": NJY_TO_ZP27_5}


def configure(
    h,
    catalog_path: str | Path,
    *,
    pretrained: str | bool = "plasticc",
    groups: tuple[str, ...] = ("infer",),
    model_weights_file: str | Path | None = None,
):
    """Point a Hyrax instance at a HATS catalog and the ParSNIP model.

    Parameters
    ----------
    h : hyrax.Hyrax
        Hyrax instance to configure.
    catalog_path : str or Path
        HATS catalog with a nested light-curve column.
    pretrained : str or False
        ParSNIP model to start from: built-in name ("plasticc", "ps1", ...), path to a
        ParSNIP .pt file, or False for a new untrained model.
    groups : tuple of str
        data_request groups to create, e.g. ("infer",) or ("train", "validate").
    model_weights_file : str or Path, optional
        Weights used by ``h.infer()``. By default the pretrained ParSNIP file is used
        (so no training run is needed). If `pretrained` is False and this is None,
        Hyrax falls back to the most recent training run.
    """
    # Setting model.name first makes Hyrax merge hyrax_parsnip/default_config.toml.
    h.set_config("model.name", MODEL_NAME)
    h.set_config("model.HyraxParsnip.pretrained", pretrained)

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
    h.set_config("data_set.LightCurveHATSDataset.band_map", PARSNIP_BAND_MAP)

    # Models that predict redshift can run on objects without one.
    h.set_config("data_set.LightCurveHATSDataset.require_redshift", not _predicts_redshift(h, pretrained))

    if model_weights_file is None and pretrained:
        model_weights_file = pretrained_model_path(str(pretrained))
    h.set_config("infer.model_weights_file", str(model_weights_file) if model_weights_file else False)
    return h


def _predicts_redshift(h, pretrained) -> bool:
    if not pretrained:
        return bool(h.config["model"]["HyraxParsnip"]["settings"].get("predict_redshift", False))

    import parsnip

    return bool(parsnip.load_model(pretrained_model_path(str(pretrained)), threads=1).settings["predict_redshift"])
