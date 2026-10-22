import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from hyrax.models import hyrax_model
from hyrax.models.model_registry import _torch_save

from hyrax_lightcurves.training import ValidationScheduler, ValidationTracking, move_optimizer_state
from hyrax_snn.features import build_features, select_observations
from hyrax_snn.pretrained import resolve, weights_file
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
class HyraxSNN(ValidationTracking, nn.Module):
    """Hyrax model running a SuperNNova classifier.

    The model (``model.HyraxSNN.pretrained``) is a built-in Fink ELAsTiCC model name
    (see `hyrax_snn.PRETRAINED_MODELS`) or a SuperNNova model directory. Inference
    returns one class probability per output class for each object.

    Training is optional: a model directory made by `hyrax_snn.training.new_model_dir`
    (no ``model.pt`` yet) starts from random weights and is trained with ``h.train()``
    on a dataset with a ``label_index`` field; `export_snn` then writes ``model.pt``.
    """

    @staticmethod
    def prepare_inputs(data_dict):
        """The shared ``(lightcurve, lengths, time_offset, redshift, mwebv)`` (see
        `hyrax_lightcurves.inputs.prepare_inputs`) plus the class index (``label_index``, -1
        when the dataset has no labels, e.g. for inference)."""
        import numpy as np

        from hyrax_lightcurves.inputs import prepare_inputs

        inputs = prepare_inputs(data_dict)
        data = data_dict["data"]
        n_objects = len(inputs[1])
        labels = np.asarray(data["label_index"], dtype=np.int64) if "label_index" in data else np.full(n_objects, -1)
        return (*inputs[:5], labels)  # without photoz (ParSNIP only)

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
        self.band_names = np.array([str(b) for b in config["data_set"]["LightCurveHATSDataset"]["band_map"].values()])
        self.time_window = tuple(settings["time_window"]) if settings["time_window"] else None
        self.detection_snr = float(settings["detection_snr"]) if settings["detection_snr"] else None
        self.redshift_error = float(settings["redshift_error"])
        self.device_setting = str(settings["device"])

        if not set(self.band_names) & set(self.snn.filters):
            raise ValueError(
                f"None of the bands {sorted(self.band_names)} from data_set.LightCurveHATSDataset.band_map are "
                f"filters of the SuperNNova model ({self.snn.filters}): no observation could be classified."
            )
        unknown = sorted(set(self.band_names) - set(self.snn.filters))
        if unknown:
            logger.warning(
                f"Bands {unknown} from data_set.LightCurveHATSDataset.band_map are not used by the SuperNNova "
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
        weights = weights_file(self.snn.model_dir)
        if weights.exists():
            self._load_snn_state(torch.load(weights, map_location="cpu", weights_only=True))
        else:
            logger.info(f"No {weights}: starting from random weights (new model to train)")

        # Training (SuperNNova's defaults): Adam, weighted cross-entropy, random-length truncation,
        # the learning rate reduced 10x when the validation loss stops improving, and the weights of
        # the epoch with the lowest validation loss kept for export.
        self.random_length = bool(settings["random_length"])
        self.optimizer = torch.optim.Adam(
            self.rnn.parameters(), lr=float(settings["learning_rate"]), weight_decay=float(settings["weight_decay"])
        )
        self.criterion = nn.CrossEntropyLoss()
        plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, factor=0.1, patience=10, min_lr=float(settings["learning_rate"]) * 1e-3
        )
        self.scheduler = ValidationScheduler(self, plateau)
        self._init_validation_tracking()

    @property
    def uses_redshift(self) -> bool:
        return self.snn.redshift != "none"

    def _load_snn_state(self, state):
        self.rnn.load_state_dict(state)

    def _trained_module(self):
        return self.rnn

    def _device(self):
        if self.device_setting == "hyrax":
            device = next(self.parameters()).device
        else:
            self.rnn.to(self.device_setting)
            device = torch.device(self.device_setting)
        move_optimizer_state(self.optimizer, device)
        return device

    def forward(self, batch):
        return self.infer_batch(batch)

    def features(self, batch) -> list:
        """SuperNNova input array for each object in a batch (None where there is nothing to classify)."""
        lightcurve, lengths, time_offset, redshift, mwebv = (_to_numpy(b) for b in batch[:5])
        out = []
        for i, n_obs in enumerate(lengths):
            rows = lightcurve[i, : int(n_obs)].astype(np.float64)
            time, flux, fluxerr, bands = select_observations(
                rows[:, 0] + float(time_offset[i]), rows[:, 1], rows[:, 2], self.band_names[rows[:, 3].astype(int)],
                self.time_window, self.detection_snr,
            )
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
            packed = self._pack([features[i] for i in valid])
            self.rnn.eval()
            with torch.no_grad():
                probabilities = torch.softmax(self.rnn(packed), dim=-1)
            output[valid] = probabilities.cpu().numpy()
        return torch.from_numpy(output)

    def _pack(self, arrays):
        """Pack variable-length (time, features) arrays, keeping their order."""
        sequences = [torch.from_numpy(np.ascontiguousarray(x)) for x in arrays]
        lengths = torch.tensor([len(s) for s in sequences])
        padded = nn.utils.rnn.pad_sequence(sequences)  # (time, batch, features)
        # enforce_sorted=False keeps the batch order (SuperNNova sorts and unsorts by length).
        return nn.utils.rnn.pack_padded_sequence(padded, lengths, enforce_sorted=False).to(self._device())

    def _loss(self, batch, augment: bool):
        """Weighted cross-entropy and accuracy over the labeled objects with features."""
        features = self.features(batch)
        labels = _to_numpy(batch[5])
        keep = [i for i, x in enumerate(features) if x is not None and labels[i] >= 0]
        if not keep:
            return None, 0, 0.0
        arrays = []
        for i in keep:
            x = features[i]
            if augment and self.random_length and len(x) > 3:
                x = x[: np.random.randint(3, len(x) + 1)]  # SuperNNova's random-length augmentation
            arrays.append(x)
        logits = self.rnn(self._pack(arrays))
        target = torch.as_tensor(labels[keep], device=logits.device)
        weight = None
        if self.snn.class_weights:
            weight = torch.tensor(self.snn.class_weights, dtype=logits.dtype, device=logits.device)
        loss = nn.functional.cross_entropy(logits, target, weight=weight)
        accuracy = (logits.argmax(dim=1) == target).float().mean().item()
        return loss, len(keep), accuracy

    def train_batch(self, batch):
        self.rnn.train()
        self.optimizer.zero_grad()
        loss, count, accuracy = self._loss(batch, augment=True)
        if loss is None:
            return {"loss": 0.0}
        loss.backward()
        self.optimizer.step()
        self._add_training_loss(loss.item(), count)
        return {"loss": loss.item(), "accuracy": accuracy}

    def validate_batch(self, batch):
        self.rnn.eval()
        with torch.no_grad():
            loss, count, accuracy = self._loss(batch, augment=False)
        if loss is None:
            return {"loss": 0.0}
        self._add_validation(loss.item(), count, accuracy)
        return {"loss": loss.item(), "accuracy": accuracy}

    def export_snn(self, model_dir=None, best=True):
        """Write the weights as ``model.pt`` (a SuperNNova VanillaRNN state dict) into the model
        directory, so it can be used with ``pretrained=<directory>``. With `best` (and a
        validation split), the weights of the epoch with the lowest validation loss, as SuperNNova."""
        model_dir = Path(model_dir) if model_dir else self.snn.model_dir
        torch.save(self._export_state(best), model_dir / "model.pt")
        return model_dir


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
