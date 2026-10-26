"""Convert models trained with SCONE itself (TensorFlow/Keras SavedModels) into hyrax_scone model directories.

The Keras model (``SconeClassifier._define_and_compile_model``) has ten layers with weights, saved as
``layer_with_weights-0`` ... ``-9``: four convolutions each followed by a BatchNormalization, a dense
layer and the output. They map one to one onto `SconeCNN`:

- Conv2D kernel (h, w, in, out) → ``Conv2d.weight`` (out, in, h, w); bias as is.
- BatchNormalization gamma, beta, moving_mean, moving_variance → weight, bias, running_mean, running_var.
- Dense kernel (in, out) → ``Linear.weight`` (out, in). The flattened image features come in the same
  order (one channel), and ``with_z`` models append the redshift and its error in the same order.
- A binary model's single sigmoid unit, P(Ia) = sigmoid(w·x + b), becomes the logits (w·x + b, 0) for
  the classes (SNIa, non-SNIa): their softmax is the same probability.

The heatmaps are then made as upstream SCONE made them (`hyrax_scone.heatmaps.UPSTREAM_SETTINGS`, with the
model's band wavelengths), except for the time window, which is centred on the S/N²-weighted detection
time instead of the simulation's true peak.

`resolve_model_dir` converts on the fly: wherever a model directory is expected, a SavedModel can be given
and is converted once into ``~/.cache/hyrax_scone/converted/<name>``.
"""

import hashlib
import json
import os
import re
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch

from hyrax_scone.config import SCONE_BAND_MAP
from hyrax_scone.heatmaps import DEFAULT_SETTINGS, LSST_WAVELENGTHS, UPSTREAM_SETTINGS
from hyrax_scone.tf_checkpoint import read_checkpoint
from hyrax_scone.training import MODEL_INFO_FILENAME

# layer_with_weights index of each part of the Keras model.
CONV_LAYERS = (0, 2, 4, 6)
NORM_LAYERS = (1, 3, 5, 7)
DENSE_LAYER, OUTPUT_LAYER = 8, 9
# Written by SCONE next to the model: the band wavelengths its heatmaps were made with.
FILTER_WAVE_FILE = "filter_mean_wavelength.dat"
# SCONE's name for the SavedModel directory.
SAVED_MODEL_DIRNAME = "trained_model"
# Changes when converted models would differ, so cached conversions are redone.
CONVERTER_VERSION = 1


def keras_weights(tensors: dict) -> dict[int, dict[str, np.ndarray]]:
    """``{layer index: {"kernel": ..., "bias": ..., ...}}`` from checkpoint variables."""
    layers: dict[int, dict[str, np.ndarray]] = {}
    for name, value in tensors.items():
        match = re.fullmatch(r"layer_with_weights-(\d+)/(\w+)/\.ATTRIBUTES/VARIABLE_VALUE", name)
        if match:
            layers.setdefault(int(match[1]), {})[match[2]] = np.asarray(value)
    if sorted(layers) != list(range(10)):
        raise ValueError(f"Expected SCONE's 10 layers with weights, found {sorted(layers)}")
    return layers


def keras_state_dict(layers: dict[int, dict[str, np.ndarray]], with_z: bool = False) -> tuple[dict, dict]:
    """`SconeCNN` state dict and its shape (``n_classes``, ``n_wavelengths``, ``n_times``, ``with_z``,
    ``binary``) from SCONE's Keras weights (`keras_weights`) of a model with or without the redshift input."""
    t = lambda a: torch.as_tensor(np.ascontiguousarray(a), dtype=torch.float32)  # noqa: E731
    blocks = ("features.0", "features.1", "features.3", "features.4")
    state = {}
    for block, conv, norm in zip(blocks, CONV_LAYERS, NORM_LAYERS):
        state[f"{block}.conv.weight"] = t(layers[conv]["kernel"].transpose(3, 2, 0, 1))
        state[f"{block}.conv.bias"] = t(layers[conv]["bias"])
        state[f"{block}.norm.weight"] = t(layers[norm]["gamma"])
        state[f"{block}.norm.bias"] = t(layers[norm]["beta"])
        state[f"{block}.norm.running_mean"] = t(layers[norm]["moving_mean"])
        state[f"{block}.norm.running_var"] = t(layers[norm]["moving_variance"])
        state[f"{block}.norm.num_batches_tracked"] = torch.tensor(0)

    n_wavelengths = layers[CONV_LAYERS[0]]["kernel"].shape[0]
    # The dense layer takes (n_wavelengths / 4) x (n_times / 4) image features, plus z and z_err.
    n_image_features = layers[DENSE_LAYER]["kernel"].shape[0] - (2 if with_z else 0)
    if n_image_features % (n_wavelengths // 4):
        raise ValueError(f"Dense layer input {n_image_features} doesn't fit {n_wavelengths}-wavelength heatmaps")
    n_times = 4 * n_image_features // (n_wavelengths // 4)
    state["dense.weight"] = t(layers[DENSE_LAYER]["kernel"].T)
    state["dense.bias"] = t(layers[DENSE_LAYER]["bias"])

    kernel, bias = layers[OUTPUT_LAYER]["kernel"], layers[OUTPUT_LAYER]["bias"]
    binary = kernel.shape[1] == 1
    if binary:  # sigmoid(w·x + b) = softmax(w·x + b, 0)[0]
        kernel = np.concatenate([kernel, np.zeros_like(kernel)], axis=1)
        bias = np.concatenate([bias, np.zeros_like(bias)])
    state["output.weight"] = t(kernel.T)
    state["output.bias"] = t(bias)
    shape = {
        "n_classes": kernel.shape[1],
        "n_wavelengths": int(n_wavelengths),
        "n_times": int(n_times),
        "with_z": with_z,
        "binary": binary,
    }
    return state, shape


def read_band_wavelengths(path) -> dict[str, float]:
    """SCONE's ``filter_mean_wavelength.dat`` (lines ``band: wavelength``; ``#`` comments)."""
    wavelengths = {}
    for line in Path(path).read_text().splitlines():
        line = line.split("#")[0].strip()
        if line:
            band, wave = line.split(":")
            wavelengths[band.strip()] = float(wave)
    return wavelengths


def import_saved_model(
    saved_model,
    output,
    class_names: list[str] | None = None,
    redshift_error: float = 0.0,
) -> dict:
    """Convert a SCONE SavedModel directory (``variables/``, optionally ``filter_mean_wavelength.dat``) into
    a hyrax_scone model directory (``model.pt``, ``scone_model.json``) at `output`.

    Binary models get the classes ``["SNIa", "non-SNIa"]`` (SCONE's label 1 is Ia); multi-class models
    need `class_names`, in the order of SCONE's ``types``. `redshift_error` is given with the redshift to
    models that take one.
    """
    from hyrax_scone.cnn import SconeCNN

    saved_model = Path(saved_model)
    tensors = read_checkpoint(saved_model / "variables" / "variables")
    # SCONE names the inputs "image", and "z", "z_err" for models with redshift (with_z).
    with_z = b"z_err" in (saved_model / "saved_model.pb").read_bytes()
    state, shape = keras_state_dict(keras_weights(tensors), with_z=with_z)
    if shape["binary"]:
        if class_names and list(class_names) != ["SNIa", "non-SNIa"]:
            raise ValueError("A binary SCONE model's classes are SNIa, non-SNIa (P(Ia) is its output)")
        class_names = ["SNIa", "non-SNIa"]
    elif not class_names or len(class_names) != shape["n_classes"]:
        raise ValueError(f"The model has {shape['n_classes']} classes: give their names, in SCONE's order")

    model = SconeCNN(shape["n_classes"], shape["n_wavelengths"], shape["n_times"], with_z=shape["with_z"])
    model.load_state_dict(state)  # checks every name and shape

    wave_file = saved_model / FILTER_WAVE_FILE
    band_wavelengths = read_band_wavelengths(wave_file) if wave_file.exists() else dict(LSST_WAVELENGTHS)
    missing = sorted(set(SCONE_BAND_MAP.values()) - set(band_wavelengths))
    if missing:
        raise ValueError(f"{wave_file} has no wavelength for bands {missing}")
    heatmap_settings = {
        **DEFAULT_SETTINGS,
        **UPSTREAM_SETTINGS,
        "n_wavelengths": shape["n_wavelengths"],
        "n_times": shape["n_times"],
        "band_wavelengths": band_wavelengths,
    }
    optimizer = {
        name.split("/")[1].lstrip("_"): float(value)
        for name, value in tensors.items()
        if name in ("optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUE", "optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUE")
    }

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output / "model.pt")
    info = {
        "class_names": list(class_names),
        "with_z": shape["with_z"],
        "redshift_error": redshift_error,
        "flux_zeropoint": 27.5,  # SCONE works in SNANA's FLUXCAL
        "band_map": dict(SCONE_BAND_MAP),
        "heatmap_settings": heatmap_settings,
        "class_weights": None,
        "trained_with": "scone (tensorflow)",
        "converted_from": str(saved_model.resolve()),
        "source_sha1": source_sha1(saved_model),
        "band_wavelengths_from": str(wave_file) if wave_file.exists() else "SCONE's legacy LSST values",
        "tensorflow_optimizer": optimizer,
    }
    (output / MODEL_INFO_FILENAME).write_text(json.dumps(info, indent=2))
    return info


def is_saved_model(path) -> bool:
    path = Path(path)
    return (path / "saved_model.pb").is_file() and (path / "variables").is_dir()


def source_sha1(saved_model) -> str:
    """Hash of everything a conversion depends on: the SavedModel's files and `CONVERTER_VERSION`."""
    saved_model = Path(saved_model)
    digest = hashlib.sha1(f"converter {CONVERTER_VERSION}".encode())
    files = [saved_model / "saved_model.pb", saved_model / FILTER_WAVE_FILE]
    files += sorted((saved_model / "variables").iterdir())
    for path in files:
        if path.is_file():
            digest.update(path.name.encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def default_cache_dir() -> Path:
    """``$XDG_CACHE_HOME/hyrax_scone``, or ``~/.cache/hyrax_scone``."""
    base = os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache"
    return Path(base) / "hyrax_scone"


def resolve_model_dir(path, cache_dir=None) -> Path:
    """A hyrax_scone model directory for `path`, converting SCONE TensorFlow models on the fly.

    - A hyrax_scone model directory (``scone_model.json``) is returned as it is.
    - A SavedModel written by SCONE (``saved_model.pb``, ``variables/``), or a directory holding one as
      ``trained_model/`` (e.g. ``data/scone_model_example``), is converted into
      ``<cache_dir>/converted/<name>``, named after the given directory (its parent's name when it is
      ``trained_model/`` itself). The conversion is reused while the model and the converter are
      unchanged (``source_sha1`` in its ``scone_model.json``) and redone otherwise.

    `cache_dir` defaults to `default_cache_dir`. Multi-class SavedModels need their class names: convert
    them with ``examples/scone_import_tf.py --classes``.
    """
    path = Path(path)
    if (path / MODEL_INFO_FILENAME).is_file():
        return path
    saved_model = path / SAVED_MODEL_DIRNAME if is_saved_model(path / SAVED_MODEL_DIRNAME) else path
    if not is_saved_model(saved_model):
        raise FileNotFoundError(
            f"{path} is neither a SCONE model directory ({MODEL_INFO_FILENAME}, model.pt) nor a TensorFlow "
            f"SavedModel written by SCONE (saved_model.pb, variables/), nor a directory holding one as "
            f"{SAVED_MODEL_DIRNAME}/"
        )
    resolved = saved_model.resolve()
    name = resolved.parent.name if resolved.name == SAVED_MODEL_DIRNAME else resolved.name
    target = Path(cache_dir or default_cache_dir()) / "converted" / name

    digest = source_sha1(saved_model)
    info_path = target / MODEL_INFO_FILENAME
    if info_path.is_file() and json.loads(info_path.read_text()).get("source_sha1") == digest:
        return target

    # Convert next to the target and move it into place, so concurrent jobs never see a partial copy.
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{name}-", dir=target.parent))
    staging.chmod(0o755)  # mkdtemp makes it private; model directories are normally readable
    try:
        import_saved_model(saved_model, staging)
        if target.exists():
            shutil.rmtree(target, ignore_errors=True)
        os.replace(staging, target)
    except ValueError as error:
        raise ValueError(f"{error} (convert it with examples/scone_import_tf.py)") from error
    except OSError:
        # Another process put its conversion in place first; use it if it is current.
        if not (info_path.is_file() and json.loads(info_path.read_text()).get("source_sha1") == digest):
            raise
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
    return target
