import json
import logging
from pathlib import Path

import torch
import torch.nn as nn
from hyrax.models import hyrax_model
from hyrax.models.model_registry import _torch_save

from hyrax_lightcurves.training import ValidationScheduler, ValidationTracking, move_optimizer_state
from hyrax_scone.cnn import SconeCNN
from hyrax_scone.training import load_model_info, weights_file

logger = logging.getLogger(__name__)

# Written next to every saved weights file (including each infer results dir) so the
# columns of the inference output can be named later.
CLASS_NAMES_FILENAME = "scone_classes.json"


@hyrax_model
class HyraxSCONE(ValidationTracking, nn.Module):
    """Hyrax model running SCONE's CNN on GP heatmaps (`hyrax_scone.SconeHATSDataset`).

    ``model.HyraxSCONE.model_dir`` is a SCONE model directory (``scone_model.json``, made by
    `hyrax_scone.new_model_dir`, and ``model.pt`` once trained). Inference returns one class
    probability per class for each object (NaN where the object has no heatmap). Without
    ``model.pt`` the network starts from random weights and is trained with ``h.train()`` on a
    dataset with a ``label_index`` field; `export_scone` then writes ``model.pt``.
    """

    @staticmethod
    def prepare_inputs(data_dict):
        """``(heatmap, redshift, label_index)``: heatmaps (batch, 2, n_wavelengths, n_times), the
        redshift (NaN without one) and the class index (-1 when the dataset has no labels)."""
        import numpy as np

        data = data_dict["data"]
        heatmap = np.asarray(data["heatmap"], dtype=np.float32)
        n_objects = len(heatmap)
        redshift = np.asarray(data.get("redshift", np.full(n_objects, np.nan)), dtype=np.float32)
        labels = np.asarray(data["label_index"], dtype=np.int64) if "label_index" in data else np.full(n_objects, -1)
        return heatmap, redshift, labels

    def __init__(self, config, data_sample=None):
        super().__init__()
        self.config = config
        settings = config["model"]["HyraxSCONE"]
        self.model_dir = Path(settings["model_dir"])
        self.info = load_model_info(self.model_dir)
        self.class_names = list(self.info["class_names"])
        self.with_z = bool(self.info["with_z"])
        self.redshift_error = float(self.info["redshift_error"])
        self.class_weights = self.info.get("class_weights")
        self.device_setting = str(settings["device"])

        heatmap = self.info["heatmap_settings"]
        self.cnn = SconeCNN(
            len(self.class_names), int(heatmap["n_wavelengths"]), int(heatmap["n_times"]), with_z=self.with_z
        )
        weights = weights_file(self.model_dir)
        if weights.exists():
            self.cnn.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
        else:
            logger.info(f"No {weights}: starting from random weights (new model to train)")

        # Training as SCONE: Adam with a constant learning rate (5e-5); class-weighted cross-entropy
        # instead of SCONE's downsampling to the smallest class. The weights of the epoch with the
        # lowest validation loss are kept for export.
        self.optimizer = torch.optim.Adam(self.cnn.parameters(), lr=float(settings["learning_rate"]))
        self.criterion = nn.CrossEntropyLoss()
        self.scheduler = ValidationScheduler(self)
        self._init_validation_tracking()

    def _trained_module(self):
        return self.cnn

    def _device(self):
        if self.device_setting == "hyrax":
            device = next(self.parameters()).device
        else:
            self.cnn.to(self.device_setting)
            device = torch.device(self.device_setting)
        move_optimizer_state(self.optimizer, device)
        return device

    def forward(self, batch):
        return self.infer_batch(batch)

    def _valid(self, heatmap, redshift):
        """Objects that can be classified: a heatmap, and a redshift for models that take one."""
        valid = torch.isfinite(heatmap).all(dim=(1, 2, 3))
        if self.with_z:
            valid &= torch.isfinite(redshift)
        return valid

    def _logits(self, heatmap, redshift):
        device = self._device()
        heatmap, redshift = heatmap.to(device), redshift.to(device)
        if not self.with_z:
            return self.cnn(heatmap)
        return self.cnn(heatmap, redshift, torch.full_like(redshift, self.redshift_error))

    def infer_batch(self, batch):
        """(batch, n_classes) class probabilities; NaN rows where there is nothing to classify."""
        heatmap, redshift = (torch.as_tensor(b) for b in batch[:2])
        valid = self._valid(heatmap, redshift).cpu()
        output = torch.full((len(heatmap), len(self.class_names)), float("nan"), dtype=torch.float64)
        if valid.any():
            self.cnn.eval()
            with torch.no_grad():
                probabilities = torch.softmax(self._logits(heatmap[valid], redshift[valid]), dim=-1)
            output[valid] = probabilities.cpu().double()
        return output

    def _loss(self, batch):
        """Class-weighted cross-entropy and accuracy over the labeled objects with a heatmap."""
        heatmap, redshift, labels = (torch.as_tensor(b) for b in batch[:3])
        keep = (self._valid(heatmap, redshift) & (labels >= 0)).cpu()
        if not keep.any():
            return None, 0, 0.0
        logits = self._logits(heatmap[keep.to(heatmap.device)], redshift[keep.to(redshift.device)])
        target = labels[keep.to(labels.device)].to(logits.device)
        weight = None
        if self.class_weights:
            weight = torch.tensor(self.class_weights, dtype=logits.dtype, device=logits.device)
        loss = nn.functional.cross_entropy(logits, target, weight=weight)
        accuracy = (logits.argmax(dim=1) == target).float().mean().item()
        return loss, int(keep.sum()), accuracy

    def train_batch(self, batch):
        self.cnn.train()
        self.optimizer.zero_grad()
        loss, count, accuracy = self._loss(batch)
        if loss is None:
            return {"loss": 0.0}
        loss.backward()
        self.optimizer.step()
        self._add_training_loss(loss.item(), count)
        return {"loss": loss.item(), "accuracy": accuracy}

    def validate_batch(self, batch):
        self.cnn.eval()
        with torch.no_grad():
            loss, count, accuracy = self._loss(batch)
        if loss is None:
            return {"loss": 0.0}
        self._add_validation(loss.item(), count, accuracy)
        return {"loss": loss.item(), "accuracy": accuracy}

    def export_scone(self, model_dir=None, best=True):
        """Write the CNN weights as ``model.pt`` into the model directory, so it can be used for
        inference (``model_dir=<directory>``). With `best` (and a validation split), the weights of
        the epoch with the lowest validation loss."""
        model_dir = Path(model_dir) if model_dir else self.model_dir
        torch.save(self._export_state(best), model_dir / "model.pt")
        return model_dir


def _save(self, save_path: Path):
    """Save Hyrax weights plus the class names for the inference output."""
    save_path = Path(save_path)
    _torch_save(self, save_path)
    with open(save_path.parent / CLASS_NAMES_FILENAME, "w") as f:
        json.dump(self.class_names, f)


def _load(self, load_path: Path):
    """Load weights from a Hyrax .pth file or a model directory's model.pt (SconeCNN state dict)."""
    state = torch.load(Path(load_path), map_location="cpu", weights_only=True)
    if not any(key.startswith("cnn.") for key in state):
        self.cnn.load_state_dict(state)
    else:
        self.load_state_dict(state)


# @hyrax_model installs Hyrax's generic torch save/load; replace them.
HyraxSCONE.save = _save
HyraxSCONE.load = _load
