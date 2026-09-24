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
