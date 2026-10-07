import json
from pathlib import Path

import numpy as np
import parsnip
from conftest import N_OBJECTS

from hyrax_parsnip import configure, feature_names, load_predictions
from hyrax_parsnip.model import FEATURE_NAMES_FILENAME


def test_pretrained_inference_without_training(hyrax_instance, hats_catalog):
    configure(hyrax_instance, hats_catalog, pretrained="plasticc")

    results = hyrax_instance.infer()

    names = feature_names(parsnip.load_model("plasticc", threads=1).settings)
    features = np.asarray(results.__get_all__())
    assert features.shape == (N_OBJECTS, len(names))
    assert np.isfinite(features[:, names.index("s1")]).all()
    assert len(set(results.ids())) == N_OBJECTS

    with open(Path(results.data_location) / FEATURE_NAMES_FILENAME) as f:
        assert json.load(f) == names


def test_matches_native_parsnip(hyrax_instance, hats_catalog):
    """Hyrax inference reproduces ParSNIP's own predict() on the same light curve."""
    from astropy.table import Table

    from hyrax_parsnip import ParsnipHATSDataset

    configure(hyrax_instance, hats_catalog, pretrained="plasticc")
    predictions = load_predictions(hyrax_instance.infer())

    dataset = ParsnipHATSDataset(hyrax_instance.config, data_location=hats_catalog)
    object_id = dataset.get_object_id(3)
    rows = dataset.get_lightcurve(3)
    bands = np.array(["lsstu", "lsstg", "lsstr", "lssti", "lsstz", "lssty"])
    light_curve = Table(
        {"time": rows[:, 0], "flux": rows[:, 1], "fluxerr": rows[:, 2], "band": bands[rows[:, 3].astype(int)]},
        meta={"object_id": object_id, "redshift": float(dataset.get_redshift(3))},
    )
    expected = parsnip.load_model("plasticc", threads=1).predict(light_curve)

    row = predictions[predictions["object_id"] == object_id][0]
    for name in ["s1", "s2", "s3", "color", "amplitude", "reference_time", "luminosity"]:
        np.testing.assert_allclose(row[name], expected[name], rtol=1e-4)


def test_all_objects_match_native_parsnip_in_large_batches(hyrax_instance, hats_catalog):
    """Every object keeps its own features when a batch has more than 10 light curves.

    lcdata orders light curves by object_id as a string ("0", "1", "10", ...), which
    once put features on the wrong objects.
    """
    import lcdata
    from astropy.table import Table

    from hyrax_parsnip import ParsnipHATSDataset

    hyrax_instance.set_config("data_loader.batch_size", N_OBJECTS)
    configure(hyrax_instance, hats_catalog, pretrained="plasticc")
    predictions = load_predictions(hyrax_instance.infer())

    dataset = ParsnipHATSDataset(hyrax_instance.config, data_location=hats_catalog)
    bands = np.array(["lsstu", "lsstg", "lsstr", "lssti", "lsstz", "lssty"])
    light_curves = []
    for idx in range(len(dataset)):
        rows = dataset.get_lightcurve(idx)
        light_curves.append(
            Table(
                {"time": rows[:, 0], "flux": rows[:, 1], "fluxerr": rows[:, 2], "band": bands[rows[:, 3].astype(int)]},
                meta={"object_id": dataset.get_object_id(idx), "redshift": float(dataset.get_redshift(idx))},
            )
        )
    expected = parsnip.load_model("plasticc", threads=1).predict_dataset(lcdata.from_light_curves(light_curves))

    predictions.sort("object_id")
    expected.sort("object_id")
    np.testing.assert_array_equal(np.asarray(predictions["object_id"]), np.asarray(expected["object_id"], dtype=str))
    for name in ["s1", "s2", "s3", "color", "amplitude", "reference_time", "luminosity"]:
        np.testing.assert_allclose(predictions[name], expected[name], rtol=1e-4, err_msg=name)


def test_photoz_model_without_redshift(hyrax_instance, hats_catalog):
    """plasticc_photoz predicts redshift, so objects without one are kept and processed."""
    configure(hyrax_instance, hats_catalog, pretrained="plasticc_photoz")
    hyrax_instance.set_config("data_set.ParsnipHATSDataset.redshift_column", False)
    predictions = load_predictions(hyrax_instance.infer())

    # All objects with a usable band are kept, including "no_redshift".
    assert len(predictions) == N_OBJECTS + 1
    assert "no_redshift" in set(predictions["object_id"])
    redshift = np.asarray(predictions["predicted_redshift"])
    assert np.isfinite(redshift).all()
    assert (redshift >= 0).all()
    assert np.isfinite(np.asarray(predictions["s1"])).all()


def test_photoz_prior_from_column(hats_catalog, tmp_path):
    """A per-object photo-z (photoz_column) replaces the constant prior; objects without one keep it."""
    from hyrax import Hyrax

    results = {}
    for name, column in (("constant", False), ("column", "redshift")):
        h = Hyrax()
        h.set_config("general.results_dir", str(tmp_path / name))
        h.set_config("data_loader.batch_size", 8)
        configure(h, hats_catalog, pretrained="plasticc_photoz")
        h.set_config("data_set.ParsnipHATSDataset.photoz_column", column)
        h.set_config("model.HyraxParsnip.photoz_fractional_error", 0.01)
        predictions = load_predictions(h.infer())
        predictions.sort("object_id")
        results[name] = predictions

    constant, column = results["constant"], results["column"]
    np.testing.assert_array_equal(constant["object_id"], column["object_id"])
    z_constant, z_column = np.asarray(constant["predicted_redshift"]), np.asarray(column["predicted_redshift"])
    has_photoz = np.asarray(constant["object_id"]) != "no_redshift"
    assert np.isfinite(z_column).all()
    assert not np.allclose(z_constant[has_photoz], z_column[has_photoz])
    np.testing.assert_allclose(z_constant[~has_photoz], z_column[~has_photoz], rtol=1e-5)
