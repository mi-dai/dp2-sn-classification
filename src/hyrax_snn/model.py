import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from hyrax.models import hyrax_model
from hyrax.models.model_registry import _torch_save

from hyrax_parsnip.model import HyraxParsnip
from hyrax_snn.features import apply_time_window, build_features
from hyrax_snn.pretrained import resolve
from hyrax_snn.rnn import VanillaRNN

logger = logging.getLogger(__name__)

# Written next to every saved weights file (including each infer results dir) so the
# columns of the inference output can be named later.
CLASS_NAMES_FILENAME = "snn_classes.json"


def _to_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


@hyrax_model
class HyraxSNN(nn.Module):
    """Hyrax model running a pretrained SuperNNova classifier (inference only).

    The model (``model.HyraxSNN.pretrained``) is a built-in Fink ELAsTiCC model name
    (see `hyrax_snn.PRETRAINED_MODELS`) or a SuperNNova model directory. Inference
    returns one class probability per output class for each object.
    """

    # Same batch layout as HyraxParsnip: (lightcurve, lengths, time_offset, redshift, mwebv).
    prepare_inputs = staticmethod(HyraxParsnip.prepare_inputs)

    def __init__(self, config, data_sample=None):
        super().__init__()
        self.config = config
        settings = config["model"]["HyraxSNN"]
        self.snn = resolve(
            str(settings["pretrained"]),
            cache_dir=settings["cache_dir"] or None,
            fix_clipped_min=bool(settings["fix_clipped_norm_min"]),
        )
        self.class_names = self.snn.class_names
        self.band_names = np.array([str(b) for b in config["data_set"]["ParsnipHATSDataset"]["band_map"].values()])
        self.time_window = tuple(settings["time_window"]) if settings["time_window"] else None
        self.detection_snr = float(settings["detection_snr"]) if settings["detection_snr"] else None
        self.redshift_error = float(settings["redshift_error"])
        self.device_setting = str(settings["device"])

        unknown = sorted(set(self.band_names) - set(self.snn.filters))
        if unknown:
            logger.warning(
                f"Bands {unknown} from data_set.ParsnipHATSDataset.band_map are not used by the SuperNNova "
                f"model, which uses {self.snn.filters}; their observations are ignored."
            )

        self.rnn = VanillaRNN(
            input_size=len(self.snn.model_features),
            nb_classes=len(self.class_names),
            hidden_dim=self.snn.hidden_dim,
            num_layers=self.snn.num_layers,
            dropout=self.snn.dropout,
            bidirectional=self.snn.bidirectional,
            layer_type=self.snn.layer_type,
            rnn_output_option=self.snn.rnn_output_option,
        )
        self._load_snn_state(torch.load(self.snn.model_dir / "model.pt", map_location="cpu", weights_only=True))

    @property
    def uses_redshift(self) -> bool:
        return self.snn.redshift != "none"

    def _load_snn_state(self, state):
        self.rnn.load_state_dict(state)

    def _device(self):
        if self.device_setting == "hyrax":
            return next(self.parameters()).device
        self.rnn.to(self.device_setting)
        return torch.device(self.device_setting)

    def forward(self, batch):
        return self.infer_batch(batch)

    def features(self, batch) -> list:
        """SuperNNova input array for each object in a batch (None where there is nothing to classify)."""
        lightcurve, lengths, time_offset, redshift, mwebv = (_to_numpy(b) for b in batch)
        out = []
        for i, n_obs in enumerate(lengths):
            rows = lightcurve[i, : int(n_obs)].astype(np.float64)
            time = rows[:, 0] + float(time_offset[i])
            flux, fluxerr = rows[:, 1], rows[:, 2]
            bands = self.band_names[rows[:, 3].astype(int)]
            if self.detection_snr is not None:
                mask = flux / fluxerr > self.detection_snr
                time, flux, fluxerr, bands = time[mask], flux[mask], fluxerr[mask], bands[mask]
            if len(time) and self.time_window:
                mask = apply_time_window(time, flux, fluxerr, self.time_window)
                time, flux, fluxerr, bands = time[mask], flux[mask], fluxerr[mask], bands[mask]
            if self.uses_redshift and not np.isfinite(redshift[i]):
                out.append(None)
                continue
            out.append(
                build_features(
                    time,
                    flux,
                    fluxerr,
                    bands,
                    self.snn,
                    mwebv=float(mwebv[i]),
                    redshift=float(redshift[i]) if np.isfinite(redshift[i]) else 0.0,
                    redshift_error=self.redshift_error,
                )
            )
        return out

    def infer_batch(self, batch):
        """(batch, n_classes) class probabilities; NaN rows where there is nothing to classify."""
        features = self.features(batch)
        output = np.full((len(features), len(self.class_names)), np.nan)
        valid = [i for i, x in enumerate(features) if x is not None]
        if valid:
            device = self._device()
            sequences = [torch.from_numpy(features[i]) for i in valid]
            lengths = torch.tensor([len(s) for s in sequences])
            padded = nn.utils.rnn.pad_sequence(sequences)  # (time, batch, features)
            # enforce_sorted=False keeps the batch order (SuperNNova sorts and unsorts by length).
            packed = nn.utils.rnn.pack_padded_sequence(padded, lengths, enforce_sorted=False).to(device)
            self.rnn.eval()
            with torch.no_grad():
                probabilities = torch.softmax(self.rnn(packed), dim=-1)
            output[valid] = probabilities.cpu().numpy()
        return torch.from_numpy(output)

    def train_batch(self, batch):
        raise NotImplementedError(
            "hyrax_snn runs pretrained SuperNNova models for inference only; training is not supported."
        )


def _save(self, save_path: Path):
    """Save Hyrax weights plus the class names for the inference output."""
    save_path = Path(save_path)
    _torch_save(self, save_path)
    with open(save_path.parent / CLASS_NAMES_FILENAME, "w") as f:
        json.dump(self.class_names, f)


def _load(self, load_path: Path):
    """Load weights from a Hyrax .pth file or a SuperNNova model.pt (VanillaRNN state dict)."""
    state = torch.load(Path(load_path), map_location="cpu", weights_only=True)
    if any(key.startswith("rnn_layer.") for key in state):
        self._load_snn_state(state)
    else:
        self.load_state_dict(state)


# @hyrax_model installs Hyrax's generic torch save/load; replace them.
HyraxSNN.save = _save
HyraxSNN.load = _load
