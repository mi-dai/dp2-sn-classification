import numpy as np
import pandas as pd
import pytest

BANDS = list("ugrizy")
# Relative brightness per band, to give the synthetic transients some color.
BAND_SCALE = dict(zip(BANDS, [0.3, 0.8, 1.0, 0.9, 0.7, 0.5]))
N_OBJECTS = 24
N_OBS = 80


def _bazin(t, t0, amplitude, rise, fall):
    return amplitude * np.exp(-(t - t0) / fall) / (1 + np.exp(-(t - t0) / rise))


def make_catalog_frames(seed: int = 42):
    """Per-object and flat light-curve frames for two toy classes of transients."""
    rng = np.random.default_rng(seed)
    objects, observations = [], []
    for i in range(N_OBJECTS):
        label = "fast" if i % 2 == 0 else "slow"
        fall = rng.uniform(15, 30) if label == "fast" else rng.uniform(80, 120)
        t0 = 60100 + rng.uniform(-10, 10)
        amplitude = rng.uniform(500, 2000)

        # Nightly-ish cadence with sub-day offsets, like a real survey.
        time = np.sort(60000 + rng.choice(250, N_OBS, replace=False) + rng.uniform(0.1, 0.3, N_OBS))
        band = rng.choice(BANDS, N_OBS)
        scale = np.array([BAND_SCALE[b] for b in band])
        fluxerr = np.full(N_OBS, 20.0)
        flux = _bazin(time, t0, amplitude, 4.0, fall) * scale + rng.normal(0, fluxerr)

        objects.append(
            {
                "object_id": f"obj{i:03d}",
                "ra": rng.uniform(0, 360),
                "dec": rng.uniform(-60, 20),
                "redshift": rng.uniform(0.05, 0.3),
                "type": label,
            }
        )
        observations.append(
            pd.DataFrame(
                {"mjd": time, "flux": flux, "fluxerr": fluxerr, "band": band},
                index=np.full(N_OBS, i),
            )
        )

    # One object without a redshift and one with an unknown band only; the dataset
    # should drop both.
    objects.append({"object_id": "no_redshift", "ra": 1.0, "dec": 1.0, "redshift": np.nan, "type": "fast"})
    observations.append(observations[0].set_axis(np.full(N_OBS, N_OBJECTS)))
    objects.append({"object_id": "bad_band", "ra": 2.0, "dec": 2.0, "redshift": 0.1, "type": "fast"})
    observations.append(observations[0].assign(band="w").set_axis(np.full(N_OBS, N_OBJECTS + 1)))

    return pd.DataFrame(objects), pd.concat(observations)


@pytest.fixture(scope="session")
def hats_catalog(tmp_path_factory):
    """Path to a small nested HATS catalog with a `lightcurve` column."""
    import lsdb
    import nested_pandas as npd

    base, flat = make_catalog_frames()
    nested = npd.NestedFrame(base).join_nested(flat, "lightcurve")
    catalog = lsdb.from_dataframe(nested, ra_column="ra", dec_column="dec", catalog_name="toy_transients")

    path = tmp_path_factory.mktemp("hats") / "toy_transients"
    catalog.write_catalog(path)
    return path


@pytest.fixture
def hyrax_instance(tmp_path):
    """A Hyrax instance writing results to a temporary directory."""
    from hyrax import Hyrax

    h = Hyrax()
    h.set_config("general.results_dir", str(tmp_path / "results"))
    h.set_config("data_loader.batch_size", 8)
    return h
