from pathlib import Path

from hyrax_parsnip.model import pretrained_model_path

MODEL_NAME = "hyrax_parsnip.model.HyraxParsnip"
DATASET_CLASS = "hyrax_parsnip.dataset.ParsnipHATSDataset"

# Dataset fields the model consumes; see HyraxParsnip.prepare_inputs.
MODEL_FIELDS = ["lightcurve", "redshift", "mwebv", "photoz"]

# Rubin fluxes are in nJy (AB zeropoint 31.4); the PLAsTiCC models use zeropoint 27.5.
NJY_TO_ZP27_5 = 10 ** (-0.4 * (31.4 - 27.5))

# `[data_set.ParsnipHATSDataset]` settings for Rubin DIA catalogs: forced photometry on
# difference images (`diaObjectForcedSource`), with flagged observations dropped.
RUBIN_DIA_SETTINGS = {
    "id_column": "diaObjectId",
    "redshift_column": False,
    "lightcurve_column": "diaObjectForcedSource",
    "time_column": "midpointMjdTai",
    "flux_column": "psfDiffFlux",
    "fluxerr_column": "psfDiffFluxErr",
    "band_column": "band",
    "flux_scale": NJY_TO_ZP27_5,
    "flag_columns": [
        "psfDiffFlux_flag",
        "invalidPsfFlag",
        "pixelFlags_saturatedCenter",
        "pixelFlags_crCenter",
        "pixelFlags_nodata",
        "diff_PixelFlags_nodataCenter",
    ],
}


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

    # Models that predict redshift can run on objects without one.
    h.set_config("data_set.ParsnipHATSDataset.require_redshift", not _predicts_redshift(h, pretrained))

    if model_weights_file is None and pretrained:
        model_weights_file = pretrained_model_path(str(pretrained))
    h.set_config("infer.model_weights_file", str(model_weights_file) if model_weights_file else False)
    return h


def _predicts_redshift(h, pretrained) -> bool:
    if not pretrained:
        return bool(h.config["model"]["HyraxParsnip"]["settings"].get("predict_redshift", False))

    import parsnip

    return bool(parsnip.load_model(pretrained_model_path(str(pretrained)), threads=1).settings["predict_redshift"])
