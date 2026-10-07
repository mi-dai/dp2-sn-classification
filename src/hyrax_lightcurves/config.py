DATASET_CLASS = "hyrax_lightcurves.dataset.LightCurveHATSDataset"

# Dataset fields the models consume; see `hyrax_lightcurves.inputs.prepare_inputs`.
MODEL_FIELDS = ["lightcurve", "redshift", "mwebv", "photoz"]

# Rubin fluxes are in nJy (AB zeropoint 31.4); the PLAsTiCC models use zeropoint 27.5.
NJY_TO_ZP27_5 = 10 ** (-0.4 * (31.4 - 27.5))

# `[data_set.LightCurveHATSDataset]` settings for Rubin DIA catalogs: forced photometry on
# difference images (`diaObjectForcedSource`), with flagged observations dropped. Fluxes
# stay in nJy here; each model package sets the flux scale (and band map) it needs.
RUBIN_DIA_SETTINGS = {
    "id_column": "diaObjectId",
    "redshift_column": False,
    "lightcurve_column": "diaObjectForcedSource",
    "time_column": "midpointMjdTai",
    "flux_column": "psfDiffFlux",
    "fluxerr_column": "psfDiffFluxErr",
    "band_column": "band",
    "flux_scale": 1.0,
    "flag_columns": [
        "psfDiffFlux_flag",
        "invalidPsfFlag",
        "pixelFlags_saturatedCenter",
        "pixelFlags_crCenter",
        "pixelFlags_nodata",
        "diff_PixelFlags_nodataCenter",
    ],
}
