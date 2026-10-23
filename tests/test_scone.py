"""hyrax_scone: the CNN port, GP heatmaps, the heatmap dataset and training through Hyrax."""

import json

import numpy as np
import pytest
import torch
from conftest import N_OBJECTS, _bazin

from hyrax_scone.cnn import SconeCNN
from hyrax_scone.heatmaps import ccm89_extinction, default_n_jobs, extinction_correct, make_heatmap, peak_time

# ── CNN ───────────────────────────────────────────────────────────────────────


def test_cnn_output_shapes():
    heatmaps = torch.randn(5, 2, 32, 180)
    assert SconeCNN(2)(heatmaps).shape == (5, 2)
    assert SconeCNN(4)(heatmaps).shape == (5, 4)
    z = torch.rand(5)
    assert SconeCNN(2, with_z=True)(heatmaps, z, torch.zeros(5)).shape == (5, 2)
    with pytest.raises(ValueError):
        SconeCNN(2, n_times=181)


def test_cnn_parameters_match_keras_model():
    """Trainable parameters of SCONE's Keras model (32 x 180 heatmaps), counted layer by layer."""
    conv = (32 * 3 * 2 * 32 + 32) + (32 * 3 * 1 * 32 + 32) + 2 * (16 * 3 * 1 * 16 + 16)
    batch_norm = 4 * 2  # gamma and beta of each 1-channel BatchNormalization
    dense = 8 * 45 * 32 + 32
    for n_classes in (2, 3):
        model = SconeCNN(n_classes)
        n = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert n == conv + batch_norm + dense + 32 * n_classes + n_classes
    # Upstream's binary model has one sigmoid output (33 parameters) instead of two logits.
    assert conv + batch_norm + dense + 33 == 22441
    # With the redshift and its error concatenated to the 360 image features.
    assert SconeCNN(2, with_z=True).dense.in_features == 362


def test_cnn_keras_layout_transpose():
    """Each block's output rows are the conv filters (SCONE's transpose), one channel."""
    block = SconeCNN(2).features[0]
    out = block(torch.randn(3, 2, 32, 180))
    assert out.shape == (3, 1, 32, 180)
    block.eval()
    x = torch.randn(1, 2, 32, 180)
    conv = torch.nn.functional.elu(block.conv(torch.nn.functional.pad(x, (1, 1))))  # (1, 32, 1, 180)
    expected = block.norm(conv[:, :, 0, :][:, None])  # filters → rows
    torch.testing.assert_close(block(x), expected)


# ── heatmaps ──────────────────────────────────────────────────────────────────


def upstream_extinction(ebv, wave):
    """SCONE's get_extinction (create_heatmaps/helpers.py), verbatim apart from formatting."""
    avu = 3.1 * ebv
    x = 10000.0 / wave
    y = x - 1.82
    aval, bval = [], []
    for i in range(len(x)):
        if 0.3 <= x[i] < 1.1:
            a, b = 0.574 * pow(x[i], 1.61), -0.527 * pow(x[i], 1.61)
        elif 1.1 <= x[i] < 3.3:
            a = (1.0 + 0.17699 * y[i] - 0.50447 * y[i] ** 2 - 0.02427 * y[i] ** 3 + 0.72085 * y[i] ** 4
                 + 0.01979 * y[i] ** 5 - 0.77530 * y[i] ** 6 + 0.32999 * y[i] ** 7)
            b = (1.41338 * y[i] + 2.28305 * y[i] ** 2 + 1.07233 * y[i] ** 3 - 5.38434 * y[i] ** 4
                 - 0.62251 * y[i] ** 5 + 5.30260 * y[i] ** 6 - 2.09002 * y[i] ** 7)
        elif 3.3 <= x[i] < 8.0:
            fa = -0.04473 * (x[i] - 5.9) ** 2 - 0.009779 * (x[i] - 5.9) ** 3 if x[i] >= 5.9 else 0.0
            fb = 0.21300 * (x[i] - 5.9) ** 2 + 0.120700 * (x[i] - 5.9) ** 3 if x[i] >= 5.9 else 0.0
            a = 1.752 - 0.316 * x[i] - 0.104 / ((x[i] - 4.67) ** 2 + 0.341) + fa
            b = -3.090 + 1.825 * x[i] + 1.206 / ((x[i] - 4.62) ** 2 + 0.263) + fb
        elif 8.0 <= x[i] <= 10.0:
            a = -1.073 - 0.628 * (x[i] - 8.0) + 0.137 * (x[i] - 8.0) ** 2 - 0.070 * (x[i] - 8.0) ** 3
            b = 13.670 + 4.257 * (x[i] - 8.0) + -0.420 * (x[i] - 8.0) ** 2 + 0.374 * (x[i] - 8.0) ** 3
        else:
            a = b = 0.0
        aval.append(a)
        bval.append(b)
    return avu * (np.array(aval) + np.array(bval) / 3.1)


def test_extinction_matches_upstream():
    wave = np.concatenate([np.linspace(1000, 10100, 200), [3000.0, 30000.0, 2e5]])
    np.testing.assert_allclose(ccm89_extinction(0.07, wave), upstream_extinction(0.07, wave), rtol=1e-12)


def test_extinction_correction_brightens_blue_most():
    wave = np.linspace(3000, 10100, 32)
    mean, std = extinction_correct(np.ones((32, 4)), np.full((32, 4), 0.5), 0.1, wave)
    assert (mean > 1).all() and np.all(np.diff(mean[:, 0]) < 0)
    np.testing.assert_allclose(std, mean / 2)
    unchanged, _ = extinction_correct(np.ones((32, 4)), np.ones((32, 4)), 0.0, wave)
    np.testing.assert_allclose(unchanged, 1)


def toy_lightcurve(seed=0, n_obs=120, t0=60150.0):
    rng = np.random.default_rng(seed)
    time = np.sort(60000 + rng.uniform(0, 400, n_obs))
    bands = rng.choice(list("ugrizY"), n_obs)
    fluxerr = np.full(n_obs, 20.0)
    flux = _bazin(time, t0, 1000.0, 4.0, 30.0) + rng.normal(0, 20, n_obs)
    return time, flux, fluxerr, bands


def test_peak_time():
    time = np.array([0.0, 10.0, 20.0, 30.0])
    flux = np.array([0.0, 300.0, 100.0, -900.0])
    fluxerr = np.full(4, 10.0)
    # S/N^2-weighted over detections (positive S/N > 5): (10*900 + 20*100) / 1000.
    assert peak_time(time, flux, fluxerr) == pytest.approx(11.0)
    # No detection: the highest-S/N observation.
    assert peak_time(time, flux / 100, fluxerr) == 10.0


def test_default_n_jobs_follows_slurm(monkeypatch):
    # A full-node Slurm job has an affinity of every thread; SLURM_CPUS_PER_TASK is what was asked for.
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "3")
    assert default_n_jobs() == 3
    monkeypatch.delenv("SLURM_CPUS_PER_TASK")
    assert default_n_jobs() >= 1


def test_make_heatmap():
    time, flux, fluxerr, bands = toy_lightcurve()
    heatmap = make_heatmap(time, flux, fluxerr, bands, mwebv=0.0)
    assert heatmap.shape == (2, 32, 180) and heatmap.dtype == np.float32
    assert heatmap[0].max() == pytest.approx(1.0) and (heatmap[1] > 0).all()
    assert heatmap[1].max() <= 2.0  # max_std
    capped = make_heatmap(time, flux, fluxerr, bands, settings={"max_std": 0.01})
    assert capped[1].max() == pytest.approx(0.01) and np.array_equal(capped[0], heatmap[0])
    # The window is -30..+150 d around the peak time: the light curve's maximum lies in it, early on.
    peak_bin = np.unravel_index(np.argmax(heatmap[0]), heatmap[0].shape)[1]
    assert 5 < peak_bin < 60

    # A mostly negative light curve (e.g. a variable star in difference flux) is scaled by its largest |flux|.
    negative = make_heatmap(time, 50.0 - flux, fluxerr, bands)
    assert np.isfinite(negative).all() and np.abs(negative[0]).max() == pytest.approx(1.0)

    # Bands SCONE doesn't know are ignored; too few observations or no flux give NaN.
    other = make_heatmap(
        np.append(time, 60100.0), np.append(flux, 1e6), np.append(fluxerr, 1.0), np.append(bands, "w")
    )
    np.testing.assert_allclose(other, heatmap, rtol=1e-4, atol=1e-5)
    assert np.isnan(make_heatmap(time[:1], flux[:1], fluxerr[:1], bands[:1])).all()
    assert np.isnan(make_heatmap(time, np.zeros_like(flux), fluxerr, bands)).all()


# ── dataset and model through Hyrax ───────────────────────────────────────────


def scone_settings(cache_dir, **overrides):
    # Small heatmaps keep the toy tests fast.
    return {"label_column": "type", "scone": {"cache_dir": str(cache_dir), "n_jobs": 2, "n_times": 40, **overrides}}


def test_dataset_heatmaps_cached(hyrax_instance, hats_catalog, tmp_path):
    from hyrax_scone import training_dataset

    cache = tmp_path / "cache"
    dataset = training_dataset(hyrax_instance, hats_catalog, scone_settings(cache))
    assert len(dataset) == N_OBJECTS + 1  # no_redshift is kept: no redshift needed
    heatmap = dataset.get_heatmap(0)
    assert heatmap.shape == (2, 32, 40) and dataset.has_heatmap().all()
    files = list(cache.glob("heatmaps-*.npy"))
    assert len(files) == 1

    # Same light curves and settings: read back from the cache (no GP fit).
    import hyrax_scone.dataset

    def no_fit(*args, **kwargs):
        raise AssertionError("heatmaps were recomputed")

    original = hyrax_scone.dataset.make_heatmaps
    hyrax_scone.dataset.make_heatmaps = no_fit
    try:
        again = training_dataset(hyrax_instance, hats_catalog, scone_settings(cache))
    finally:
        hyrax_scone.dataset.make_heatmaps = original
    np.testing.assert_array_equal(again.get_heatmap(3), dataset.get_heatmap(3))

    # Other settings: a new cache file.
    training_dataset(hyrax_instance, hats_catalog, scone_settings(cache, time_window=[-20.0, 140.0]))
    assert len(list(cache.glob("heatmaps-*.npy"))) == 2


def test_train_export_and_infer(hats_catalog, tmp_path):
    from hyrax import Hyrax

    from hyrax_lightcurves.training import find_checkpoint
    from hyrax_scone import configure, load_model_info, load_predictions, new_model_dir, training_dataset

    def new_hyrax(epochs=None):
        h = Hyrax()
        h.set_config("general.results_dir", str(tmp_path / "results"))
        h.set_config("data_loader.batch_size", 8)
        if epochs:
            h.set_config("train.epochs", epochs)
            h.set_config("split.train", 0.75)
            h.set_config("split.validate", 0.25)
        return h

    settings = scone_settings(tmp_path / "cache")
    output = tmp_path / "trained"
    h = new_hyrax(epochs=2)
    dataset = training_dataset(h, hats_catalog, settings)
    info = new_model_dir(output, dataset)
    assert info["class_names"] == ["fast", "slow"]
    # Inverse class frequency, mean 1: 13 fast (incl. no_redshift), 12 slow.
    np.testing.assert_allclose(info["class_weights"], [25 / 26, 25 / 24])
    assert load_model_info(output)["heatmap_settings"]["n_times"] == 40

    configure(h, hats_catalog, model_dir=output, groups=("train", "validate"), dataset_settings=settings)
    model = h.train()
    assert len(model.validation_history) == 2 and model.best_epoch in (1, 2)
    model.export_scone(output)
    assert (output / "model.pt").exists()

    # Resume for one more epoch: the validation history comes back with the checkpoint.
    checkpoint = find_checkpoint("latest", tmp_path / "results")
    h_resume = new_hyrax(epochs=3)
    h_resume.set_config("train.resume", str(checkpoint))
    configure(h_resume, hats_catalog, model_dir=output, groups=("train", "validate"), dataset_settings=settings)
    resumed = h_resume.train()
    assert resumed._epochs_trained == 3 and len(resumed.validation_history) == 3
    resumed.export_scone(output)

    # Inference with the trained directory; the heatmap settings come from the model.
    def infer(batch_size):
        h_infer = new_hyrax()
        h_infer.set_config("data_loader.batch_size", batch_size)
        configure(h_infer, hats_catalog, model_dir=output, dataset_settings={"scone": settings["scone"]})
        return load_predictions(h_infer.infer())

    predictions = infer(8)
    assert len(predictions) == N_OBJECTS + 1
    probabilities = np.stack([predictions["fast"], predictions["slow"]], axis=1)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1, rtol=1e-5)
    assert set(predictions["predicted_class"]) <= {"fast", "slow"}
    # Eval mode (no dropout, running BatchNorm statistics): batches don't change the results.
    other = infer(5)
    np.testing.assert_allclose(other["fast"], predictions["fast"], rtol=1e-5)


def test_with_z_model_needs_redshift(hyrax_instance, hats_catalog, tmp_path):
    from hyrax_scone import configure, load_predictions, new_model_dir, training_dataset

    settings = {**scone_settings(tmp_path / "cache"), "redshift_column": "redshift"}
    dataset = training_dataset(hyrax_instance, hats_catalog, settings, with_z=True)
    assert len(dataset) == N_OBJECTS  # no_redshift dropped
    output = tmp_path / "with_z"
    new_model_dir(output, dataset, with_z=True, redshift_error=0.01)
    torch.save(SconeCNN(2, n_times=40, with_z=True).state_dict(), output / "model.pt")

    configure(hyrax_instance, hats_catalog, model_dir=output, dataset_settings=settings)
    predictions = load_predictions(hyrax_instance.infer())
    assert len(predictions) == N_OBJECTS and (predictions["predicted_class"] != "").all()
    assert json.loads((output / "scone_model.json").read_text())["redshift_error"] == 0.01
