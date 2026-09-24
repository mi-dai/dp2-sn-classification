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
