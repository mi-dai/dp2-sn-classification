import numpy as np
from conftest import N_OBJECTS, N_OBS

from hyrax_lightcurves import LightCurveHATSDataset
from hyrax_parsnip import configure


def make_dataset(hyrax_instance, catalog):
    configure(hyrax_instance, catalog)
    return LightCurveHATSDataset(hyrax_instance.config, data_location=catalog)


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
    assert np.isnan(dataset.get_photoz(0))  # no photoz_column


def test_photoz_column(hyrax_instance, hats_catalog):
    configure(hyrax_instance, hats_catalog)
    hyrax_instance.set_config("data_set.LightCurveHATSDataset.photoz_column", "redshift")
    dataset = LightCurveHATSDataset(hyrax_instance.config, data_location=hats_catalog)
    assert all(dataset.get_photoz(i) == dataset.get_redshift(i) for i in range(len(dataset)))


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
    hyrax_instance.set_config("data_set.LightCurveHATSDataset.label_column", "type")
    dataset = LightCurveHATSDataset(hyrax_instance.config, data_location=hats_catalog)

    assert {dataset.get_label(i) for i in range(len(dataset))} == {"fast", "slow"}


def test_label_index_and_schemes(hyrax_instance, hats_catalog):
    from hyrax_lightcurves.labels import class_labels, scheme_classes

    configure(hyrax_instance, hats_catalog)
    hyrax_instance.set_config("data_set.LightCurveHATSDataset.label_column", "type")
    dataset = LightCurveHATSDataset(hyrax_instance.config, data_location=hats_catalog)

    # Scheme "all" (default): the catalog types, sorted.
    assert dataset.label_classes == ["fast", "slow"]
    for i in range(len(dataset)):
        assert dataset.label_classes[dataset.get_label_index(i)] == dataset.get_label(i)
    assert dataset.get_label_index(0).dtype == np.int64

    types = ["SNIa", "SNII", "SNIbc", "KN", "SNIa-91bg"]
    assert list(class_labels(types, "ia")) == ["SNIa", "non-SNIa", "non-SNIa", "non-SNIa", "non-SNIa"]
    assert list(class_labels(types, "dp2")) == ["SNIa", "SNII", "SNIbc", "other", "other"]
    assert scheme_classes("ia") == ["SNIa", "non-SNIa"]
    assert scheme_classes("all", types) == sorted(types)


def test_rubin_schema_flux_scale_and_flags(hyrax_instance, tmp_path):
    """Rubin-style catalog: nJy fluxes, quality flags, extra columns, no redshift."""
    import lsdb
    import nested_pandas as npd
    import pandas as pd
    from conftest import make_catalog_frames

    base, flat = make_catalog_frames()
    base = base.iloc[:4].assign(diaObjectId=np.arange(4, dtype=np.int64) + 10**15)[["diaObjectId", "ra", "dec"]]
    flat = flat[flat.index < 4]
    flag = np.zeros(len(flat), dtype=bool)
    flag[::2] = True  # every other observation is flagged
    sources = pd.DataFrame(
        {
            "midpointMjdTai": flat["mjd"].to_numpy(),
            "psfDiffFlux": flat["flux"].to_numpy() / 0.1,  # stored in "nJy" = flux / flux_scale
            "psfDiffFluxErr": flat["fluxerr"].to_numpy() / 0.1,
            "band": flat["band"].to_numpy(),
            "psfDiffFlux_flag": flag,
            "psfMag": np.zeros(len(flat)),  # unused column
        },
        index=flat.index,
    )
    nested = npd.NestedFrame(base).join_nested(sources, "diaObjectForcedSource")
    path = tmp_path / "rubin_like"
    lsdb.from_dataframe(nested, ra_column="ra", dec_column="dec").write_catalog(path)

    configure(hyrax_instance, path)
    for key, value in {
        "id_column": "diaObjectId",
        "redshift_column": False,
        "require_redshift": False,
        "lightcurve_column": "diaObjectForcedSource",
        "time_column": "midpointMjdTai",
        "flux_column": "psfDiffFlux",
        "fluxerr_column": "psfDiffFluxErr",
        "flux_scale": 0.1,
        "flag_columns": ["psfDiffFlux_flag"],
    }.items():
        hyrax_instance.set_config(f"data_set.LightCurveHATSDataset.{key}", value)
    dataset = LightCurveHATSDataset(hyrax_instance.config, data_location=path)

    assert len(dataset) == 4
    idx = [dataset.get_object_id(i) for i in range(4)].index(str(10**15))
    expected = flat[flat.index == 0].iloc[1::2]  # unflagged observations of object 0
    lightcurve = dataset.get_lightcurve(idx)
    np.testing.assert_allclose(lightcurve[:, 0], expected["mjd"])
    np.testing.assert_allclose(lightcurve[:, 1], expected["flux"])
    np.testing.assert_allclose(lightcurve[:, 2], expected["fluxerr"])
    assert np.isnan(dataset.get_redshift(idx))
