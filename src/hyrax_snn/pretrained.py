"""Pretrained SuperNNova models from the Fink broker, and their settings.

The ELAsTiCC models are trained by Fink (Fraga et al. 2024, A&A, arXiv:2404.08798) and
published in https://github.com/astrolabsoftware/fink-science (Apache-2.0) under
``fink_science/data/models/snn_models/``. Each model directory holds ``cli_args.json``
(SuperNNova settings), ``data_norm.json`` (feature normalization) and ``model.pt``
(a VanillaRNN state dict). They are downloaded from a pinned commit into a local cache.
"""

import json
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

FINK_SCIENCE_COMMIT = "52bb6d5a7e401bc6bf02630c735643432030727e"
FINK_MODELS_URL = (
    f"https://raw.githubusercontent.com/astrolabsoftware/fink-science/{FINK_SCIENCE_COMMIT}"
    "/fink_science/data/models/snn_models"
)
MODEL_FILES = ("cli_args.json", "data_norm.json", "model.pt")

# Built-in models: name -> (directory in fink-science, class names by output index).
# For binary models SuperNNova's class 0 is the group its `sntypes` tags "Ia" (which
# Fink reuses for the target group, e.g. all SNe in SN_vs_other); multi-class models
# put that group first. The names here are what the classes actually are.
PRETRAINED_MODELS = {
    # Without redshift
    "elasticc_ia": ("elasticc_ia", ["SNIa", "other"]),
    "SN_vs_other": ("elasticc_binary_broad/SN_vs_other", ["SN", "other"]),
    # With redshift (HOSTGAL_SPECZ) and Milky Way E(B-V)
    "elasticc_broad": ("elasticc_broad", ["SN", "Fast", "Long", "Periodic", "NonPeriodic"]),
    "Fast_vs_other": ("elasticc_binary_broad/Fast_vs_other", ["Fast", "other"]),
    "Long_vs_other": ("elasticc_binary_broad/Long_vs_other", ["Long", "other"]),
    "Periodic_vs_other": ("elasticc_binary_broad/Periodic_vs_other", ["Periodic", "other"]),
    "NonPeriodic_vs_other": ("elasticc_binary_broad/NonPeriodic_vs_other", ["NonPeriodic", "other"]),
}


def default_cache_dir() -> Path:
    base = os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache"
    return Path(base) / "hyrax_snn" / FINK_SCIENCE_COMMIT[:12]


def pretrained_model_dir(name_or_path: str, cache_dir: str | Path | None = None) -> Path:
    """Directory of a pretrained model: a built-in name (downloaded and cached) or a local dir.

    Run this once on a machine with internet access (e.g. a NERSC login node) to fill
    the cache before running on nodes without it.
    """
    if name_or_path not in PRETRAINED_MODELS:
        path = Path(name_or_path)
        if not (path / "cli_args.json").exists():
            raise ValueError(
                f"{name_or_path!r} is neither a built-in model ({sorted(PRETRAINED_MODELS)}) "
                "nor a directory with cli_args.json, data_norm.json and model.pt."
            )
        return path

    repo_dir = PRETRAINED_MODELS[name_or_path][0]
    target = Path(cache_dir) if cache_dir else default_cache_dir()
    target = target / repo_dir
    for filename in MODEL_FILES:
        path = target / filename
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(path.suffix + ".part")
            urllib.request.urlretrieve(f"{FINK_MODELS_URL}/{repo_dir}/{filename}", tmp)
            tmp.rename(path)
    return target


@dataclass
class SNNSettings:
    """What inference needs from a SuperNNova model directory."""

    model_dir: Path
    filters: list[str]
    all_features: list[str]
    training_features: list[str]
    features_to_normalize: list[str]
    norm: str
    arr_norm: np.ndarray  # (len(features_to_normalize), 3): min, mean, std
    redshift: str  # "none", "zspe" or "zpho"
    class_names: list[str]
    hidden_dim: int
    num_layers: int
    dropout: float
    bidirectional: bool
    layer_type: str
    rnn_output_option: str
    # Models trained with hyrax_snn (snn_train.py) record these; the Fink models don't.
    flux_zeropoint: float | None = None  # zeropoint of the training fluxes (None: nJy, as for Fink)
    class_weights: list[float] | None = None  # cross-entropy weights used in training

    @property
    def filter_combinations(self) -> list[str]:
        """Every non-empty filter subset as a string, in `filters` order (SuperNNova's one-hot labels)."""
        from itertools import combinations

        return ["".join(c) for n in range(1, len(self.filters) + 1) for c in combinations(self.filters, n)]

    @property
    def model_features(self) -> list[str]:
        """Model inputs in order: `all_features` restricted to `training_features`, as in
        SuperNNova's on-the-fly classification."""
        training = set(self.training_features)
        return [f for f in self.all_features if f in training]


def load_settings(model_dir: str | Path, class_names: list[str] | None = None, fix_clipped_min: bool = False) -> SNNSettings:
    """Read cli_args.json and data_norm.json.

    `fix_clipped_min`: SuperNNova computes the log-normalization mean/std with the true
    minimum of the training data, but stores the minimum clipped to -2000, and trains
    with the clipped value. With the fix, min = -exp(mean) is restored for those
    features, so the normalization matches its mean/std instead of what the network was
    trained on. Off by default (training-consistent). Models trained with hyrax_snn store a
    minimum consistent with their mean/std, so it doesn't apply to them.
    """
    model_dir = Path(model_dir)
    with open(model_dir / "cli_args.json") as f:
        cli = json.load(f)
    with open(model_dir / "data_norm.json") as f:
        norm_json = json.load(f)

    features_to_normalize = list(cli["training_features_to_normalize"])
    arr_norm = np.array(
        [[norm_json[f]["min"], norm_json[f]["mean"], norm_json[f]["std"]] for f in features_to_normalize],
        dtype=np.float64,
    )
    if fix_clipped_min and cli.get("trained_with") != "hyrax_snn":
        clipped = (arr_norm[:, 0] == -2000) & (arr_norm[:, 1] > np.log(4000))
        arr_norm[clipped, 0] = -np.exp(arr_norm[clipped, 1])

    if class_names is None:
        class_names = cli.get("class_names") or [f"class{i}" for i in range(int(cli["nb_classes"]))]
    if len(class_names) != int(cli["nb_classes"]):
        raise ValueError(f"{len(class_names)} class names for a model with {cli['nb_classes']} classes")

    return SNNSettings(
        model_dir=model_dir,
        filters=list(cli["list_filters"]),
        all_features=list(cli["all_features"]),
        training_features=list(cli["training_features"]),
        features_to_normalize=features_to_normalize,
        norm=cli["norm"],
        arr_norm=arr_norm,
        redshift=cli.get("redshift") or "none",
        class_names=list(class_names),
        hidden_dim=int(cli["hidden_dim"]),
        num_layers=int(cli["num_layers"]),
        dropout=float(cli["dropout"]),
        bidirectional=bool(cli["bidirectional"]),
        layer_type=cli["layer_type"],
        rnn_output_option=cli["rnn_output_option"],
        flux_zeropoint=cli.get("flux_zeropoint"),
        class_weights=cli.get("class_weights"),
    )


def resolve(name_or_path: str, cache_dir=None, fix_clipped_min: bool = False) -> SNNSettings:
    """Model directory and settings for a built-in name or a local model directory."""
    model_dir = pretrained_model_dir(name_or_path, cache_dir)
    class_names = PRETRAINED_MODELS[name_or_path][1] if name_or_path in PRETRAINED_MODELS else None
    return load_settings(model_dir, class_names, fix_clipped_min)
