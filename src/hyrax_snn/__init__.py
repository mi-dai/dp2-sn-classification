"""Run pretrained SuperNNova classifiers through Hyrax on HATS light-curve catalogs."""

from hyrax_snn.config import FINK_EXACT, FINK_EXACT_FLUX_SCALE, RUBIN_DIA_SETTINGS, SNN_BAND_MAP, configure
from hyrax_snn.model import HyraxSNN
from hyrax_snn.predictions import load_predictions
from hyrax_snn.pretrained import PRETRAINED_MODELS, pretrained_model_dir

__all__ = [
    "FINK_EXACT",
    "FINK_EXACT_FLUX_SCALE",
    "PRETRAINED_MODELS",
    "RUBIN_DIA_SETTINGS",
    "SNN_BAND_MAP",
    "HyraxSNN",
    "configure",
    "load_predictions",
    "pretrained_model_dir",
]
