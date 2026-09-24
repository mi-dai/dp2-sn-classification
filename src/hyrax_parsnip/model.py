import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from hyrax.models import hyrax_model
from hyrax.models.model_registry import _torch_save

from hyrax_parsnip.lightcurves import batch_to_tables

logger = logging.getLogger(__name__)

# Written next to every saved weights file (including each infer results dir) so the
# columns of the inference output can be named later.
FEATURE_NAMES_FILENAME = "parsnip_features.json"


def feature_names(settings: dict) -> list[str]:
    """Names of the per-object values `HyraxParsnip.infer_batch` returns, in order.

    These are the numeric columns of ``ParsnipModel.predict_dataset``.
    """
    names = [
        "reference_time",
        "reference_time_error",
        "color",
        "color_error",
        "amplitude",
        "amplitude_error",
    ]
    for idx in range(settings["latent_size"]):
        names += [f"s{idx + 1}", f"s{idx + 1}_error"]
    if settings["predict_redshift"]:
        names += ["predicted_redshift", "predicted_redshift_error"]
    names += [
        "total_s2n",
        "count",
        "count_s2n_3",
        "count_s2n_5",
        "count_s2n_3_pre",
        "count_s2n_3_rise",
        "count_s2n_3_fall",
        "count_s2n_3_post",
        "model_chisq",
        "model_dof",
        "luminosity",
        "luminosity_error",
    ]
    return names


def pretrained_model_path(name_or_path: str) -> str:
    """Resolve a built-in ParSNIP model name (e.g. "plasticc") to its .pt file path."""
    import importlib.resources

    if Path(name_or_path).suffix:
        return str(name_or_path)
    path = importlib.resources.files("parsnip") / "models" / f"{name_or_path}.pt"
    if not path.is_file():
        raise ValueError(f"No built-in ParSNIP model named '{name_or_path}'")
    return str(path)


def _unwrap(value):
    """Convert tomlkit containers to plain Python objects."""
    return value.unwrap() if hasattr(value, "unwrap") else value


class ParsnipPlateauScheduler:
    """Adapts ParSNIP's ReduceLROnPlateau to Hyrax's once-per-epoch ``scheduler.step()``.

    ReduceLROnPlateau needs the epoch loss, which Hyrax does not pass, so the model
    accumulates it in ``_epoch_loss`` during ``train_batch``.
    """

    def __init__(self, scheduler, model):
        self.scheduler = scheduler
        self.model = model

    def get_last_lr(self):
        return [group["lr"] for group in self.scheduler.optimizer.param_groups]

    def step(self):
        self.scheduler.step(self.model._epoch_loss)
        self.model._epoch_loss = 0.0

    def state_dict(self):
        return self.scheduler.state_dict()

    def load_state_dict(self, state_dict):
        self.scheduler.load_state_dict(state_dict)


@hyrax_model
class HyraxParsnip(nn.Module):
    """Hyrax model wrapping a ParSNIP ``ParsnipModel``.

    Inference returns the ParSNIP latent representation and derived features for each
    object (see `feature_names`). Training is optional: the model can start from a
    built-in or saved ParSNIP model (``model.HyraxParsnip.pretrained``) and be used
    directly for inference, fine-tuned, or trained from scratch with
    ``pretrained = false``.
    """

    def __init__(self, config, data_sample=None):
        super().__init__()
        self.config = config
        settings = config["model"]["HyraxParsnip"]
        self.band_names = [str(b) for b in config["data_set"]["ParsnipHATSDataset"]["band_map"].values()]
        self.augment_train = bool(settings["augment_train"])
        self.device_setting = str(settings["device"])
        self.photoz = float(settings["photoz"])
        self.photoz_error = float(settings["photoz_error"])

        self.parsnip = self._build_parsnip(settings)

        unknown = sorted(set(self.band_names) - set(self.parsnip.settings["bands"]))
        if unknown:
            raise ValueError(
                f"Bands {unknown} from data_set.ParsnipHATSDataset.band_map are not supported by the "
                f"ParSNIP model, which uses {list(self.parsnip.settings['bands'])}."
            )

        self.feature_names = feature_names(self.parsnip.settings)

        # Use ParSNIP's own optimizer, loss and LR schedule instead of Hyrax's config.
        self.optimizer = self.parsnip.optimizer
        self.criterion = self.parsnip.loss_function
        self.scheduler = ParsnipPlateauScheduler(self.parsnip.scheduler, self)
        self._epoch_loss = 0.0

    def _build_parsnip(self, settings: dict):
        import parsnip

        device = "cpu" if self.device_setting == "hyrax" else self.device_setting
        threads = int(settings["threads"])
        pretrained = settings["pretrained"]
        if pretrained:
            logger.info(f"Loading ParSNIP model '{pretrained}'")
            return parsnip.load_model(str(pretrained), device=device, threads=threads)

        logger.info("Building a new, untrained ParSNIP model")
        return parsnip.ParsnipModel(
            path="parsnip_model.pt",
            bands=self.band_names,
            device=device,
            threads=threads,
            settings=dict(_unwrap(settings["settings"]) or {}),
        )

    @staticmethod
    def prepare_inputs(data_dict):
        """Unpack the collated batch into float32 arrays
        ``(lightcurve, lengths, time_offset, redshift, mwebv)``.

        Hyrax may place inputs on devices without float64 support (e.g. MPS), so each
        light curve's times are split into an integer-day ``time_offset`` (exact in
        float32) and residuals with seconds-level precision over multi-year light curves.
        """
        import numpy as np

        data = data_dict["data"]
        lengths = np.asarray(data["lengths"], dtype=np.int64)
        n_objects = len(lengths)
        lightcurve = np.array(data["lightcurve"], dtype=np.float64)

        time_offset = np.zeros(n_objects)
        for i, n_obs in enumerate(lengths):
            if n_obs:
                time_offset[i] = np.floor(lightcurve[i, :n_obs, 0].min())
                lightcurve[i, :n_obs, 0] -= time_offset[i]

        redshift = np.asarray(data.get("redshift", np.full(n_objects, np.nan)))
        mwebv = np.asarray(data.get("mwebv", np.zeros(n_objects)))
        return (
            lightcurve.astype(np.float32),
            lengths,
            time_offset.astype(np.float32),
            redshift.astype(np.float32),
            mwebv.astype(np.float32),
        )

    def forward(self, batch):
        return self.infer_batch(batch)

    def _sync_device(self):
        """Keep ParSNIP on its configured device.

        Hyrax moves the model with Module.to(), which bypasses ParSNIP's own `to()`
        (that also moves its non-parameter tensors) and may pick a device ParSNIP
        wasn't set up for (e.g. MPS). With ``device = "hyrax"`` ParSNIP follows
        Hyrax's choice instead.
        """
        param_device = str(next(self.parameters()).device)
        target = param_device if self.device_setting == "hyrax" else self.device_setting
        if param_device != target or str(self.parsnip.device) != target:
            self.parsnip.to(target, force=True)

    def _preprocess(self, batch) -> list:
        """Rebuild and ParSNIP-preprocess each light curve; invalid ones become None."""
        from parsnip import preprocess_light_curve

        self._sync_device()
        tables = batch_to_tables(*batch, band_names=self.band_names)
        if self.parsnip.settings["predict_redshift"]:
            # Photo-z models read these PLAsTiCC-style keys. The host photo-z is an
            # encoder input only; the redshift itself is predicted from the light curve.
            for table in tables:
                table.meta["hostgal_specz"] = table.meta["redshift"]
                table.meta["hostgal_photoz"] = self.photoz
                table.meta["hostgal_photoz_err"] = self.photoz_error
        return [preprocess_light_curve(t, self.parsnip.settings, raise_on_invalid=False) for t in tables]

    def infer_batch(self, batch):
        """Return a (batch, n_features) tensor of ParSNIP predictions; NaN rows for
        light curves ParSNIP cannot process."""
        import lcdata

        preprocessed = self._preprocess(batch)
        output = np.full((len(preprocessed), len(self.feature_names)), np.nan)

        valid = [i for i, lc in enumerate(preprocessed) if lc is not None]
        if valid:
            with torch.no_grad():
                predictions = self.parsnip.predict_dataset(
                    lcdata.from_light_curves([preprocessed[i] for i in valid])
                )
            # lcdata sorts light curves by object_id as a string ("0", "1", "10", ...), so
            # place rows by their object_id (the batch position), not by input order.
            rows = np.asarray(predictions["object_id"]).astype(int)
            output[rows] = np.stack(
                [np.asarray(predictions[name], dtype=np.float64) for name in self.feature_names], axis=1
            )

        return torch.from_numpy(output)

    def _loss(self, batch, augment: bool):
        light_curves = [lc for lc in self._preprocess(batch) if lc is not None]
        if not light_curves:
            return None, 0
        if augment:
            import lcdata

            # ParSNIP's augmentation expects the light curves (and meta objects) of
            # an lcdata Dataset, as produced in ParsnipModel.fit.
            light_curves = lcdata.from_light_curves(light_curves).light_curves
            light_curves = self.parsnip.augment_light_curves(light_curves, as_table=False)
        result = self.parsnip.forward(light_curves)
        return self.parsnip.loss_function(result), len(light_curves)

    def train_batch(self, batch):
        from parsnip.utils import replace_nan_grads

        self.optimizer.zero_grad()
        loss, count = self._loss(batch, augment=self.augment_train)
        if loss is None:
            return {"loss": 0.0}

        loss.backward()
        replace_nan_grads(self.parameters())
        self.optimizer.step()

        self._epoch_loss += loss.item()
        return {"loss": loss.item() / count}

    def validate_batch(self, batch):
        with torch.no_grad():
            loss, count = self._loss(batch, augment=False)
        return {"loss": loss.item() / count if loss is not None else 0.0}

    def export_parsnip(self, path):
        """Write the model in ParSNIP's native format, usable by ``parsnip.load_model``."""
        self.parsnip.path = str(Path(path).resolve())
        self.parsnip.save()


def _save(self, save_path: Path):
    """Save Hyrax weights plus the feature names for the inference output."""
    save_path = Path(save_path)
    _torch_save(self, save_path)
    with open(save_path.parent / FEATURE_NAMES_FILENAME, "w") as f:
        json.dump(self.feature_names, f)


def _load(self, load_path: Path):
    """Load weights from a Hyrax .pth file or a native ParSNIP .pt file.

    The architecture must already match; it is set by ``pretrained`` / ``settings``.
    """
    load_path = Path(load_path)
    try:
        state = torch.load(load_path, map_location="cpu", weights_only=True)
    except Exception:
        # Native ParSNIP files hold numpy arrays in their settings; this matches
        # what parsnip.load_model itself does.
        state = torch.load(load_path, map_location="cpu", weights_only=False)

    if isinstance(state, (list, tuple)) and len(state) == 2 and isinstance(state[0], dict):
        self.parsnip.load_state_dict(state[1])
    else:
        self.load_state_dict(state)
    self._sync_device()


# @hyrax_model installs Hyrax's generic torch save/load; replace them with the
# ParSNIP-aware versions.
HyraxParsnip.save = _save
HyraxParsnip.load = _load
