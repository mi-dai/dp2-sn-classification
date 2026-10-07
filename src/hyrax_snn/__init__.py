"""Run SuperNNova classifiers through Hyrax on HATS light-curve catalogs: Fink's pretrained models,
or models trained here (optional, see `hyrax_snn.training`)."""

from hyrax_snn.config import (
    FINK_EXACT,
    FINK_EXACT_FLUX_SCALE,
    RUBIN_DIA_SETTINGS,
    SNN_BAND_MAP,
    configure,
    rubin_dia_settings,
)
from hyrax_snn.model import HyraxSNN
from hyrax_snn.predictions import load_predictions
from hyrax_snn.pretrained import PRETRAINED_MODELS, pretrained_model_dir
from hyrax_snn.training import new_model_dir, training_dataset

__all__ = [
    "FINK_EXACT",
    "FINK_EXACT_FLUX_SCALE",
    "PRETRAINED_MODELS",
    "RUBIN_DIA_SETTINGS",
    "SNN_BAND_MAP",
    "HyraxSNN",
    "configure",
    "load_predictions",
    "new_model_dir",
    "pretrained_model_dir",
    "rubin_dia_settings",
    "training_dataset",
]
