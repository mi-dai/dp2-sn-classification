"""Run SCONE (Qu et al. 2021) through Hyrax on HATS light-curve catalogs: GP heatmaps and a PyTorch
port of SCONE's CNN. SCONE publishes no trained weights: models are trained here (e.g. on PLAsTiCC, see
examples/scone_train.py), and models trained with SCONE itself (TensorFlow) are converted on the fly."""

from hyrax_scone.config import RUBIN_DIA_SETTINGS, SCONE_BAND_MAP, configure, rubin_dia_settings
from hyrax_scone.convert import resolve_model_dir
from hyrax_scone.dataset import SconeHATSDataset
from hyrax_scone.model import HyraxSCONE
from hyrax_scone.predictions import load_predictions
from hyrax_scone.training import load_model_info, new_model_dir, training_dataset

__all__ = [
    "RUBIN_DIA_SETTINGS",
    "SCONE_BAND_MAP",
    "HyraxSCONE",
    "SconeHATSDataset",
    "configure",
    "load_model_info",
    "load_predictions",
    "new_model_dir",
    "resolve_model_dir",
    "rubin_dia_settings",
    "training_dataset",
]
