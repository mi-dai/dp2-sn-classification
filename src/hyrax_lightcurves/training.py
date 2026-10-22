"""Training helpers shared by the Hyrax models trained here (hyrax_snn, hyrax_scone).

Hyrax reports only the last validation batch and steps the scheduler before validating, so
`ValidationTracking` accumulates the loss over all validation objects each epoch, keeps the
weights of the best epoch, and `ValidationScheduler` carries that state through Hyrax's
checkpoints (``train.resume``).
"""

from pathlib import Path

import numpy as np


class ValidationScheduler:
    """Hyrax's once-per-epoch ``scheduler.step()``, optionally ReduceLROnPlateau on the validation loss.

    Hyrax passes no loss to ``step()`` and steps before validating, so the plateau uses the model's
    latest full validation loss (from the previous epoch; ``validate_post_epoch``), or the training
    loss of the epoch when there is no validation split. Without `plateau` the learning rate is constant.
    """

    def __init__(self, model, plateau=None):
        self.scheduler = plateau
        self.model = model

    def get_last_lr(self):
        return [group["lr"] for group in self.model.optimizer.param_groups]

    def step(self):
        if self.scheduler is not None:
            loss = self.model._last_validation_loss
            self.scheduler.step(self.model._epoch_loss if loss is None else loss)
        self.model._epoch_loss = 0.0
        self.model._epochs_trained += 1

    def state_dict(self):
        # Hyrax saves the scheduler in each checkpoint, so the model's training state (validation
        # history and best weights) rides along and is restored on `train.resume`.
        model = self.model
        return {
            "plateau": self.scheduler.state_dict() if self.scheduler is not None else None,
            "epochs_trained": model._epochs_trained,
            "validation_history": list(model.validation_history),
            "last_validation_loss": model._last_validation_loss,
            "best_state": model._best_state,
        }

    def load_state_dict(self, state_dict):
        if "epochs_trained" not in state_dict:  # a plain ReduceLROnPlateau state
            self.scheduler.load_state_dict(state_dict)
            return
        if self.scheduler is not None and state_dict["plateau"] is not None:
            self.scheduler.load_state_dict(state_dict["plateau"])
        self.model._epochs_trained = state_dict["epochs_trained"]
        history = [tuple(h) for h in state_dict["validation_history"]]
        # Hyrax saves the checkpoint before validating its last epoch: that validation is lost.
        history += [(np.nan, np.nan)] * (self.model._epochs_trained - len(history))
        self.model.validation_history = history
        self.model._last_validation_loss = state_dict["last_validation_loss"]
        best = state_dict["best_state"]
        self.model._best_state = {k: v.cpu() for k, v in best.items()} if best else None


class ValidationTracking:
    """Mixin for Hyrax models: full validation loss per epoch and the weights of the best epoch.

    The model calls `_init_validation_tracking` in ``__init__``, `_add_validation` from
    ``validate_batch`` and `_add_training_loss` from ``train_batch``, and defines `_trained_module`
    (the network whose weights are kept).
    """

    def _init_validation_tracking(self):
        self._epoch_loss = 0.0
        self._validation_sum = self._validation_correct = self._validation_count = 0.0
        self._last_validation_loss = None
        self.validation_history = []  # (loss, accuracy) of each validation epoch, over all batches
        self._epochs_trained = 0
        self._best_state = None

    def _trained_module(self):
        raise NotImplementedError

    def _add_training_loss(self, loss: float, count: int):
        self._epoch_loss += loss * count

    def _add_validation(self, loss: float, count: int, accuracy: float):
        self._validation_sum += loss * count
        self._validation_correct += accuracy * count
        self._validation_count += count

    def validate_post_epoch(self):
        """Hyrax hook after each validation pass: record the loss over all validation objects
        (Hyrax only reports the last batch) and keep the best weights so far."""
        import logging

        if not self._validation_count:
            return
        loss = self._validation_sum / self._validation_count
        accuracy = self._validation_correct / self._validation_count
        self._validation_sum = self._validation_correct = self._validation_count = 0.0
        self._last_validation_loss = loss
        self.validation_history.append((loss, accuracy))
        if loss <= np.nanmin([h[0] for h in self.validation_history]):
            state = self._trained_module().state_dict()
            self._best_state = {k: v.detach().cpu().clone() for k, v in state.items()}
        logging.getLogger(type(self).__module__).info(
            f"Validation epoch {len(self.validation_history)}: loss {loss:.4f}, accuracy {accuracy:.3f}"
        )

    @property
    def best_epoch(self):
        """Validation epoch (1-based) with the lowest loss, or None without validation."""
        losses = np.array([h[0] for h in self.validation_history], dtype=float)
        if not np.isfinite(losses).any():
            return None
        return int(np.nanargmin(losses)) + 1

    def _export_state(self, best: bool = True) -> dict:
        """Weights to export: those of the best validation epoch (with `best` and a validation split),
        otherwise the current ones."""
        state = self._best_state if best and self._best_state is not None else self._trained_module().state_dict()
        return {k: v.cpu() for k, v in state.items()}


def move_optimizer_state(optimizer, device):
    """Hyrax loads a resumed checkpoint's optimizer state onto its own device (e.g. MPS/CUDA)."""
    import torch

    for param_state in optimizer.state.values():
        for key, value in param_state.items():
            if torch.is_tensor(value) and value.device != device:
                param_state[key] = value.to(device)


def find_checkpoint(resume, results_dir) -> Path:
    """The Hyrax checkpoint to resume training from.

    `resume` is a checkpoint file, a Hyrax train run directory (its latest
    ``checkpoint_epoch_N.pt``), or ``"latest"``: the newest train run under `results_dir`
    that has one.
    """

    def latest_in(run_dir):
        checkpoints = list(Path(run_dir).glob("checkpoint_epoch_*.pt"))
        return max(checkpoints, key=lambda p: int(p.stem.rsplit("_", 1)[1])) if checkpoints else None

    if str(resume) == "latest":
        runs = sorted(Path(results_dir).glob("*-train-*"), key=lambda p: p.stat().st_mtime, reverse=True)
        found = next((c for c in map(latest_in, runs) if c is not None), None)
        if found is None:
            raise FileNotFoundError(f"No Hyrax train run with a checkpoint under {results_dir}")
        return found
    path = Path(resume)
    if path.is_dir():
        found = latest_in(path)
        if found is None:
            raise FileNotFoundError(f"No checkpoint_epoch_*.pt in {path}")
        return found
    if not path.is_file():
        raise FileNotFoundError(f"No checkpoint {path}")
    return path


def class_weights(labels, n_classes: int) -> tuple[list[float], list[int]]:
    """Inverse-class-frequency weights (mean 1 over the classes present) and the class counts."""
    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=n_classes).astype(float)
    weights = np.where(counts > 0, counts.sum() / (len(counts) * np.maximum(counts, 1)), 0.0)
    return [float(w) for w in weights], [int(c) for c in counts]
