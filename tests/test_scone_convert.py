"""Models trained with SCONE itself: the Keras → PyTorch weight mapping, upstream heatmap settings, and
the example SavedModel (tests skipped unless data/scone_model_example exists, or TensorFlow is installed)."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from test_scone import toy_lightcurve

from hyrax_scone.cnn import SconeCNN
from hyrax_scone.convert import keras_state_dict
from hyrax_scone.heatmaps import UPSTREAM_SETTINGS, ccm89_extinction, extinction_correct, make_heatmap

EXAMPLE = Path(__file__).resolve().parents[1] / "data" / "scone_model_example" / "trained_model"
needs_example = pytest.mark.skipif(not (EXAMPLE / "variables").is_dir(), reason="no data/scone_model_example")

# ── Keras forward pass in numpy (the layer semantics SCONE's Keras model uses) ──


def keras_layers(rng, n_wavelengths, n_times, n_out, with_z):
    """Random weights with Keras shapes, keyed like SavedModel's layer_with_weights-N."""
    rows = n_wavelengths
    layers = {}
    for index, (filters, channels) in zip((0, 2, 4, 6), ((rows, 2), (rows, 1), (rows // 2, 1), (rows // 2, 1))):
        layers[index] = {"kernel": rng.normal(0, 0.3, (filters, 3, channels, filters)), "bias": rng.normal(0, 0.1, filters)}
    for index in (1, 3, 5, 7):
        layers[index] = {
            "gamma": rng.uniform(0.5, 1.5, 1), "beta": rng.normal(0, 0.1, 1),
            "moving_mean": rng.normal(0, 0.1, 1), "moving_variance": rng.uniform(0.5, 2, 1),
        }
    n_features = (rows // 4) * (n_times // 4) + (2 if with_z else 0)
    layers[8] = {"kernel": rng.normal(0, 0.3, (n_features, 32)), "bias": rng.normal(0, 0.1, 32)}
    layers[9] = {"kernel": rng.normal(0, 0.3, (32, n_out)), "bias": rng.normal(0, 0.1, n_out)}
    return layers


def keras_forward(x, layers, z=None):
    """SCONE's Keras model in eval mode on NHWC images x (n, wavelengths, times, 2)."""

    def conv_block(x, conv, norm):
        x = np.pad(x, ((0, 0), (0, 0), (1, 1), (0, 0)))  # ZeroPadding2D(padding=(0, 1))
        kernel, bias = conv["kernel"], conv["bias"]
        kh, kw = kernel.shape[:2]
        n, h, w, _ = x.shape
        out = np.zeros((n, h - kh + 1, w - kw + 1, kernel.shape[3]))
        for j in range(out.shape[2]):  # "valid" Conv2D
            out[:, :, j, :] = np.einsum("nhwc,hwcf->nf", x[:, :kh, j : j + kw, :], kernel)[:, None, :]
        out = out + bias
        out = np.where(out > 0, out, np.expm1(out))  # elu
        out = out.transpose(0, 3, 2, 1)  # SCONE's Reshape layer
        return norm["gamma"] * (out - norm["moving_mean"]) / np.sqrt(norm["moving_variance"] + 1e-3) + norm["beta"]

    def max_pool(x):
        n, h, w, c = x.shape
        return x[:, : h // 2 * 2, : w // 2 * 2].reshape(n, h // 2, 2, w // 2, 2, c).max(axis=(2, 4))

    x = conv_block(x, layers[0], layers[1])
    x = conv_block(x, layers[2], layers[3])
    x = max_pool(x)
    x = conv_block(x, layers[4], layers[5])
    x = conv_block(x, layers[6], layers[7])
    x = max_pool(x).reshape(len(x), -1)  # Flatten (row-major over h, w, c)
    if z is not None:
        x = np.concatenate([x, z], axis=1)
    x = np.maximum(x @ layers[8]["kernel"] + layers[8]["bias"], 0)
    logits = x @ layers[9]["kernel"] + layers[9]["bias"]
    if logits.shape[1] == 1:
        return 1 / (1 + np.exp(-logits[:, 0]))  # sigmoid: P(Ia)
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


@pytest.mark.parametrize("with_z", [False, True])
@pytest.mark.parametrize("n_out", [1, 3])
def test_keras_weights_map_onto_scone_cnn(with_z, n_out):
    rng = np.random.default_rng(1)
    n_wavelengths, n_times = 8, 12
    layers = keras_layers(rng, n_wavelengths, n_times, n_out, with_z)
    state, shape = keras_state_dict(layers, with_z=with_z)
    assert (shape["n_wavelengths"], shape["n_times"], shape["with_z"]) == (n_wavelengths, n_times, with_z)
    assert shape["n_classes"] == (2 if n_out == 1 else n_out)

    model = SconeCNN(shape["n_classes"], n_wavelengths, n_times, with_z=with_z)
    model.load_state_dict(state)
    model.eval()
    x = rng.normal(0, 1, (5, n_wavelengths, n_times, 2))
    z = rng.uniform(0, 1, (5, 2)) if with_z else None
    expected = keras_forward(x, layers, z)

    heatmaps = torch.tensor(x.transpose(0, 3, 1, 2), dtype=torch.float32)
    with torch.no_grad():
        inputs = (torch.tensor(z[:, 0], dtype=torch.float32), torch.tensor(z[:, 1], dtype=torch.float32)) if with_z else ()
        probabilities = torch.softmax(model(heatmaps, *inputs), dim=1).numpy()
    if n_out == 1:  # classes (SNIa, non-SNIa): P(SNIa) is SCONE's sigmoid output
        np.testing.assert_allclose(probabilities[:, 0], expected, rtol=1e-4, atol=1e-6)
    else:
        np.testing.assert_allclose(probabilities, expected, rtol=1e-4, atol=1e-6)


# ── upstream heatmap settings ─────────────────────────────────────────────────


def test_extinction_modes():
    wave = np.linspace(3000, 10100, 32)
    mean, std = np.full((32, 5), 100.0), np.full((32, 5), 10.0)
    extinction = ccm89_extinction(0.1, wave)[:, None]
    added_mean, added_std = extinction_correct(mean, std, 0.1, wave, mode="add")  # upstream: flux + A_λ
    np.testing.assert_allclose(added_mean, mean + extinction)
    np.testing.assert_array_equal(added_std, std)
    scaled_mean, scaled_std = extinction_correct(mean, std, 0.1, wave)  # flux x 10^(0.4 A_λ)
    np.testing.assert_allclose(scaled_mean, mean * 10 ** (0.4 * extinction))
    np.testing.assert_allclose(scaled_std, std * 10 ** (0.4 * extinction))
    with pytest.raises(ValueError):
        extinction_correct(mean, std, 0.1, wave, mode="subtract")


def test_upstream_heatmap_settings():
    time, flux, fluxerr, bands = toy_lightcurve()
    upstream = make_heatmap(time, flux, fluxerr, bands, mwebv=0.1, settings=UPSTREAM_SETTINGS)
    assert np.isfinite(upstream).all() and upstream[0].max() == pytest.approx(1.0)
    assert not np.allclose(upstream, make_heatmap(time, flux, fluxerr, bands, mwebv=0.1))
    # Upstream's additive extinction raises the zero-flux end of the window by A_λ / max flux.
    no_dust = make_heatmap(time, flux, fluxerr, bands, mwebv=0.0, settings=UPSTREAM_SETTINGS)
    assert (upstream[0][:, -1] > no_dust[0][:, -1]).all()


def test_normalization_and_std_cap_options():
    time, flux, fluxerr, bands = toy_lightcurve()
    negative = (time, 50.0 - flux, fluxerr, bands)
    upstream_max = make_heatmap(*negative, settings={"normalization": "max"})
    max_abs = make_heatmap(*negative, settings={"normalization": "max_abs"})
    assert np.abs(max_abs[0]).max() == pytest.approx(1.0)
    assert upstream_max[0].max() == pytest.approx(1.0) and np.abs(upstream_max).max() > 1
    uncapped = make_heatmap(time, flux, fluxerr, bands, settings={"max_std": False, "normalization": "max"})
    capped = make_heatmap(time, flux, fluxerr, bands, settings={"max_std": 0.01, "normalization": "max"})
    assert uncapped[1].max() > 0.01 and capped[1].max() == pytest.approx(0.01)
    with pytest.raises(ValueError):
        make_heatmap(time, flux, fluxerr, bands, settings={"normalization": "median"})


def test_band_wavelengths_setting():
    time, flux, fluxerr, bands = toy_lightcurve()
    shifted = {b: w for b, w in zip("ugrizY", (3710.0, 4805.8, 6221.5, 7559.2, 8680.6, 9757.0))}
    default = make_heatmap(time, flux, fluxerr, bands)
    other = make_heatmap(time, flux, fluxerr, bands, settings={"band_wavelengths": shifted})
    assert other.shape == default.shape and not np.allclose(other, default)
    # Bands missing from band_wavelengths are ignored.
    only_gri = make_heatmap(time, flux, fluxerr, bands, settings={"band_wavelengths": {b: shifted[b] for b in "gri"}})
    assert np.isfinite(only_gri).all()


def test_default_settings_keep_old_cache_key(hyrax_instance, hats_catalog, tmp_path):
    """New heatmap settings at their defaults don't change the cache key (existing caches stay valid)."""
    from test_scone import scone_settings

    from hyrax_scone import training_dataset
    from hyrax_scone.heatmaps import HEATMAP_VERSION

    dataset = training_dataset(hyrax_instance, hats_catalog, scone_settings(tmp_path / "cache"))
    digest = hashlib.sha1()
    for array in (dataset._lightcurve_data, dataset._starts, dataset._stops, dataset._mwebv):
        digest.update(np.ascontiguousarray(array).tobytes())
    old_keys = ("n_wavelengths", "n_times", "wavelength_range", "time_window", "peak_detection_snr",
                "time_scale_guess", "max_std")
    old_settings = {key: dataset.heatmap_settings[key] for key in old_keys}
    digest.update(json.dumps([list(dataset.settings["band_map"].values()), old_settings, HEATMAP_VERSION]).encode())
    assert dataset._cache_key() == digest.hexdigest()[:20]

    other = training_dataset(hyrax_instance, hats_catalog, scone_settings(tmp_path / "cache", extinction="add"))
    assert other._cache_key() != dataset._cache_key()


# ── the example SavedModel ────────────────────────────────────────────────────


@needs_example
def test_read_example_checkpoint_and_import(tmp_path):
    from hyrax_scone import load_model_info
    from hyrax_scone.convert import import_saved_model, keras_weights
    from hyrax_scone.tf_checkpoint import read_checkpoint

    layers = keras_weights(read_checkpoint(EXAMPLE / "variables" / "variables"))
    assert layers[0]["kernel"].shape == (32, 3, 2, 32) and layers[8]["kernel"].shape == (360, 32)
    info = import_saved_model(EXAMPLE, tmp_path / "converted")
    assert info["class_names"] == ["SNIa", "non-SNIa"] and info["with_z"] is False
    assert info["heatmap_settings"]["extinction"] == "add" and info["heatmap_settings"]["band_wavelengths"]["u"] == 3710.0
    assert load_model_info(tmp_path / "converted")["heatmap_settings"]["max_std"] is False
    model = SconeCNN(2)
    model.load_state_dict(torch.load(tmp_path / "converted" / "model.pt", weights_only=True))


@needs_example
def test_example_matches_tensorflow(tmp_path):
    tf = pytest.importorskip("tensorflow")
    from hyrax_scone.convert import import_saved_model

    import_saved_model(EXAMPLE, tmp_path / "converted")
    model = SconeCNN(2)
    model.load_state_dict(torch.load(tmp_path / "converted" / "model.pt", weights_only=True))
    model.eval()
    heatmaps = np.stack(
        [make_heatmap(*toy_lightcurve(seed), settings=UPSTREAM_SETTINGS) for seed in range(8)]
    ).astype(np.float32)
    with torch.no_grad():
        ours = torch.softmax(model(torch.tensor(heatmaps)), dim=1)[:, 0].numpy()
    signature = tf.saved_model.load(str(EXAMPLE)).signatures["serving_default"]
    theirs = signature(image=tf.constant(heatmaps.transpose(0, 2, 3, 1)))["label"].numpy()[:, 0]
    np.testing.assert_allclose(ours, theirs, atol=1e-5)
