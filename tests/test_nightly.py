import numpy as np

from hyrax_lightcurves.nightly import coadd_nightly, night_of, sigma_clip_nightly


def coadd(row, time, flux, fluxerr, band, **kwargs):
    return coadd_nightly(np.array(row), np.array(time), np.array(flux), np.array(fluxerr), np.array(band), **kwargs)


def test_inverse_variance_weights():
    row, time, flux, err, band = coadd([0, 0], [60000.2, 60000.4], [100.0, 130.0], [10.0, 20.0], [1, 1])

    np.testing.assert_array_equal(row, [0])
    np.testing.assert_array_equal(band, [1])
    np.testing.assert_allclose(flux, [106.0])  # weights 4:1
    np.testing.assert_allclose(err, [1.0 / np.sqrt(0.01 + 0.0025)])
    np.testing.assert_allclose(time, [60000.24])


def test_outlier_clipped_only_with_three_points():
    good = [100.0, 101.0, 99.0, 100.0]
    _, _, flux, err, _ = coadd([0] * 5, [60000.2] * 5, good + [1000.0], [10.0] * 5, [0] * 5)
    np.testing.assert_allclose(flux, [100.0])
    np.testing.assert_allclose(err, [10.0 / 2])

    _, _, flux, _, _ = coadd([0] * 2, [60000.2] * 2, [100.0, 1000.0], [10.0] * 2, [0] * 2)
    np.testing.assert_allclose(flux, [550.0])  # two points: not clipped


def test_clip_threshold():
    flux, err = np.array([0.0, 0.0, 0.0, 0.0, 30.0]), np.ones(5)
    assert not sigma_clip_nightly(np.zeros(5, dtype=int), flux, err, sigma_scale=5.0)[-1]
    assert sigma_clip_nightly(np.zeros(5, dtype=int), flux, err, sigma_scale=50.0).all()


def test_group_keeps_at_least_one_point():
    """A group where every point is an outlier from the median keeps the closest one."""
    keep = sigma_clip_nightly(np.zeros(3, dtype=int), np.array([0.0, 100.0, 300.0]), np.ones(3))
    assert keep.tolist() == [False, True, False]


def test_groups_by_object_band_and_night():
    # Night boundary at MJD x.5: 60000.4 and 60000.6 are different nights, 60000.6 and 60001.4 the same.
    assert night_of(60000.6) == night_of(60001.4) != night_of(60000.4)

    row, time, _, _, band = coadd(
        [1, 1, 1, 1, 0],
        [60000.4, 60000.6, 60001.4, 60000.6, 60000.6],
        [1.0, 2.0, 2.0, 3.0, 4.0],
        [1.0] * 5,
        [0, 0, 0, 2, 0],
    )
    np.testing.assert_array_equal(row, [0, 1, 1, 1])  # sorted by object, then time
    np.testing.assert_allclose(time, [60000.6, 60000.4, 60000.6, 60001.0])
    np.testing.assert_array_equal(band, [0, 0, 2, 0])


def test_empty():
    assert all(len(a) == 0 for a in coadd([], [], [], [], []))
