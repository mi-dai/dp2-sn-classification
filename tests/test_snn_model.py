"""hyrax_snn with a synthetic SuperNNova model directory (no network, no supernnova)."""

import itertools
import json

import numpy as np
import pytest
import torch
from conftest import N_OBJECTS

from hyrax_snn import configure, load_predictions
from hyrax_snn.features import apply_time_window, build_features
from hyrax_snn.pretrained import load_settings
from hyrax_snn.rnn import VanillaRNN

FILTERS = ["u", "g", "r", "i", "z", "Y"]


def write_model_dir(path, redshift="none", norm="global", nb_classes=2, seed=0):
    """A SuperNNova-style model directory with random weights."""
    combos = ["".join(c) for n in range(1, 7) for c in itertools.combinations(FILTERS, n)]
    to_normalize = [f"FLUXCAL_{f}" for f in FILTERS] + [f"FLUXCALERR_{f}" for f in FILTERS] + ["delta_time"]
    host = ["HOSTGAL_PHOTOZ", "HOSTGAL_PHOTOZ_ERR", "HOSTGAL_SPECZ", "HOSTGAL_SPECZ_ERR"]
    all_features = to_normalize + host + ["MWEBV"] + sorted(combos)
    training = to_normalize + ["MWEBV"] + sorted(combos)
    if redshift == "zspe":
        training += ["HOSTGAL_SPECZ", "HOSTGAL_SPECZ_ERR"]
    cli = {
        "list_filters": FILTERS,
        "all_features": all_features,
        "training_features": training,
        "training_features_to_normalize": to_normalize,
        "norm": norm,
        "nb_classes": nb_classes,
        "redshift": redshift,
        "hidden_dim": 8,
        "num_layers": 2,
        "dropout": 0.05,
        "bidirectional": True,
        "layer_type": "lstm",
        "rnn_output_option": "mean",
    }
    norm_json = {f: {"min": -2000.0, "mean": 15.0, "std": 0.01} for f in to_normalize if f.startswith("FLUXCAL_")}
    norm_json.update({f: {"min": 0.0, "mean": 3.0, "std": 1.0} for f in to_normalize if f.startswith("FLUXCALERR_")})
    norm_json["delta_time"] = {"min": 0.0, "mean": 0.0, "std": 3.0}

    path.mkdir(parents=True)
    (path / "cli_args.json").write_text(json.dumps(cli))
    (path / "data_norm.json").write_text(json.dumps(norm_json))
    torch.manual_seed(seed)
    n_inputs = len([f for f in all_features if f in set(training)])
    rnn = VanillaRNN(n_inputs, nb_classes, hidden_dim=8)
    torch.save(rnn.state_dict(), path / "model.pt")
    return path


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory):
    return write_model_dir(tmp_path_factory.mktemp("snn") / "no_z")


@pytest.fixture(scope="module")
def zspe_model_dir(tmp_path_factory):
    return write_model_dir(tmp_path_factory.mktemp("snn") / "zspe", redshift="zspe", norm="cosmo_quantile", nb_classes=3)


def infer(h, catalog, model, **model_settings):
    configure(h, catalog, pretrained=str(model))
    for key, value in model_settings.items():
        h.set_config(f"model.HyraxSNN.{key}", value)
    return load_predictions(h.infer())


# ── features ──────────────────────────────────────────────────────────────────


def test_build_features_groups_and_one_hot(model_dir):
    settings = load_settings(model_dir)
    # Two nights; the first has g twice (keep the smaller error) and r 0.2 d later.
    time = [100.0, 100.1, 100.3, 105.0]
    flux = [10.0, 20.0, 30.0, 40.0]
    fluxerr = [2.0, 1.0, 3.0, 4.0]
    bands = ["g", "g", "r", "Y"]
    x = build_features(time, flux, fluxerr, bands, settings)

    assert x.shape == (2, len(settings.model_features))
    columns = settings.model_features
    # One-hot: first step observed in g and r, second in Y.
    assert x[0, columns.index("gr")] == 1
    assert x[1, columns.index("Y")] == 1
    assert x[:, [columns.index(c) for c in settings.filter_combinations]].sum() == 2


def test_build_features_ignores_unknown_bands_and_empty(model_dir):
    settings = load_settings(model_dir)
    assert build_features([1.0], [1.0], [1.0], ["w"], settings) is None
    assert build_features([1.0, 2.0], [1.0, 2.0], [1.0, 1.0], ["w", "g"], settings).shape[0] == 1


def test_time_window_around_max_snr():
    time = np.array([0.0, 50.0, 100.0, 300.0])
    mask = apply_time_window(time, np.array([1.0, 100.0, 5.0, 1.0]), np.ones(4), (-30, 100))
    np.testing.assert_array_equal(mask, [False, True, True, False])


def test_fix_clipped_min(model_dir):
    assert load_settings(model_dir).arr_norm[0, 0] == -2000
    np.testing.assert_allclose(load_settings(model_dir, fix_clipped_min=True).arr_norm[0, 0], -np.exp(15.0))


# ── Hyrax inference ─────────────────────────────────────────────────────────


def test_infer_without_redshift_keeps_objects_without_one(hyrax_instance, hats_catalog, model_dir):
    predictions = infer(hyrax_instance, hats_catalog, model_dir)

    # Every object with usable bands, including "no_redshift"; class names default to class<i>.
    assert len(predictions) == N_OBJECTS + 1
    assert "no_redshift" in set(predictions["object_id"])
    assert predictions.colnames == ["object_id", "class0", "class1", "predicted_class"]
    probabilities = np.stack([predictions["class0"], predictions["class1"]], axis=1)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1, rtol=1e-5)


def test_infer_with_redshift_requires_one(hyrax_instance, hats_catalog, zspe_model_dir):
    predictions = infer(hyrax_instance, hats_catalog, zspe_model_dir)

    assert len(predictions) == N_OBJECTS
    assert "no_redshift" not in set(predictions["object_id"])
    assert np.isfinite(np.asarray(predictions["class2"])).all()


def test_batch_size_does_not_change_results(hats_catalog, model_dir, tmp_path):
    from hyrax import Hyrax

    results = []
    for batch_size in (1, N_OBJECTS + 1):
        h = Hyrax()
        h.set_config("general.results_dir", str(tmp_path / f"b{batch_size}"))
        h.set_config("data_loader.batch_size", batch_size)
        predictions = infer(h, hats_catalog, model_dir)
        predictions.sort("object_id")
        results.append(predictions)
    np.testing.assert_array_equal(results[0]["object_id"], results[1]["object_id"])
    np.testing.assert_allclose(results[0]["class0"], results[1]["class0"], rtol=1e-5)


def test_detection_snr_and_window_options_run(hyrax_instance, hats_catalog, model_dir):
    predictions = infer(hyrax_instance, hats_catalog, model_dir, detection_snr=5.0, time_window=False)
    assert np.isfinite(np.asarray(predictions["class0"])).sum() > 0


# ── training ────────────────────────────────────────────────────────────────


def test_train_export_and_infer(hats_catalog, tmp_path):
    from hyrax import Hyrax

    from hyrax_snn import new_model_dir, training_dataset
    from hyrax_snn.config import SNN_BAND_MAP

    dataset_settings = {"label_column": "type"}  # scheme "all": fast, slow
    output = tmp_path / "trained"
    h = Hyrax()
    h.set_config("general.results_dir", str(tmp_path / "results"))
    h.set_config("data_loader.batch_size", 8)
    h.set_config("train.epochs", 2)
    h.set_config("split.train", 0.75)
    h.set_config("split.validate", 0.25)
    dataset = training_dataset(h, hats_catalog, dataset_settings)
    settings = new_model_dir(
        output, dataset, list(SNN_BAND_MAP.values()), h.config["model"]["HyraxSNN"], hidden_dim=8
    )

    # Classes, class weights and global normalization from the training set.
    assert settings.class_names == ["fast", "slow"]
    # Inverse class frequency, mean 1: 13 fast (incl. no_redshift, kept without redshift), 12 slow.
    np.testing.assert_allclose(settings.class_weights, [25 / 26, 25 / 24])
    norm = json.loads((output / "data_norm.json").read_text())
    assert norm["FLUXCAL_u"] == norm["FLUXCAL_Y"] != norm["FLUXCALERR_u"]
    assert norm["FLUXCAL_g"]["min"] < 0 and norm["delta_time"]["min"] == 0
    assert not (output / "model.pt").exists()

    configure(h, hats_catalog, pretrained=str(output), groups=("train", "validate"), dataset_settings=dataset_settings)
    model = h.train()
    # Validation loss over all validation objects each epoch; the best epoch's weights are exported.
    assert len(model.validation_history) == 2 and model.best_epoch in (1, 2)
    model.export_snn(output)
    assert (output / "model.pt").exists()

    # The trained directory is used like a pretrained model.
    h_infer = Hyrax()
    h_infer.set_config("general.results_dir", str(tmp_path / "results"))
    predictions = infer(h_infer, hats_catalog, output)
    assert len(predictions) == N_OBJECTS + 1
    probabilities = np.stack([predictions["fast"], predictions["slow"]], axis=1)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1, rtol=1e-5)


def test_log_standardization_clips_min():
    from hyrax_snn.training import log_standardization

    values = np.array([-1e5, -10.0, 0.0, 50.0, 500.0])
    stats = log_standardization(values)
    assert stats["min"] == -2000.0  # as SuperNNova; the bright variable's -1e5 doesn't set it
    logs = np.log(np.clip(values, -2000.0, None) + 2000.0 + 1e-5)
    assert stats["mean"] == pytest.approx(logs.mean()) and stats["std"] == pytest.approx(logs.std())
    assert log_standardization(np.array([1.0, 2.0]))["min"] == 1.0
