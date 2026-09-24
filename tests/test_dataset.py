import numpy as np
from conftest import N_OBJECTS, N_OBS

from hyrax_parsnip import ParsnipHATSDataset, configure


def make_dataset(hyrax_instance, catalog):
    configure(hyrax_instance, catalog)
    return ParsnipHATSDataset(hyrax_instance.config, data_location=catalog)


def test_filters_unusable_objects(hyrax_instance, hats_catalog):
    dataset = make_dataset(hyrax_instance, hats_catalog)

    assert len(dataset) == N_OBJECTS
    ids = {dataset.get_object_id(i) for i in range(len(dataset))}
    assert "no_redshift" not in ids
    assert "bad_band" not in ids


def test_getters(hyrax_instance, hats_catalog):
    dataset = make_dataset(hyrax_instance, hats_catalog)

    lightcurve = dataset.get_lightcurve(0)
    assert lightcurve.shape == (N_OBS, 4)
    assert lightcurve.dtype == np.float64
    # Band indices follow band_map order (u, g, r, i, z, y).
    assert set(np.unique(lightcurve[:, 3])) <= set(range(6))
    # MJDs keep sub-day precision.
    assert np.any(lightcurve[:, 0] % 1 > 0)
    assert np.isfinite(dataset.get_redshift(0))
    assert dataset.get_mwebv(0) == 0.0


def test_collate_pads(hyrax_instance, hats_catalog):
    dataset = make_dataset(hyrax_instance, hats_catalog)

    short = dataset.get_lightcurve(0)[:10]
    long = dataset.get_lightcurve(1)
    collated = dataset.collate_lightcurve([{"lightcurve": short}, {"lightcurve": long}])

    assert collated["lightcurve"].shape == (2, len(long), 4)
    np.testing.assert_array_equal(collated["lengths"], [10, len(long)])
    np.testing.assert_array_equal(collated["lightcurve"][0, :10], short)
    assert not collated["lightcurve"][0, 10:].any()


def test_label_column(hyrax_instance, hats_catalog):
    configure(hyrax_instance, hats_catalog)
    hyrax_instance.set_config("data_set.ParsnipHATSDataset.label_column", "type")
    dataset = ParsnipHATSDataset(hyrax_instance.config, data_location=hats_catalog)

    assert {dataset.get_label(i) for i in range(len(dataset))} == {"fast", "slow"}
