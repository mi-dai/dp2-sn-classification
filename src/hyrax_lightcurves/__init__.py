"""Light curves from HATS catalogs for Hyrax models (shared by hyrax_parsnip and hyrax_snn).

The dataset serves each object's cleaned light curve and per-object values; each model
package turns them into its own input and sets the dataset's band map and flux scale.
"""

from hyrax_lightcurves.catalog import catalog_metadata
from hyrax_lightcurves.config import DATASET_CLASS, MODEL_FIELDS, NJY_TO_ZP27_5, RUBIN_DIA_SETTINGS
from hyrax_lightcurves.dataset import LightCurveHATSDataset
from hyrax_lightcurves.inputs import prepare_inputs

__all__ = [
    "DATASET_CLASS",
    "MODEL_FIELDS",
    "NJY_TO_ZP27_5",
    "RUBIN_DIA_SETTINGS",
    "LightCurveHATSDataset",
    "catalog_metadata",
    "prepare_inputs",
]
