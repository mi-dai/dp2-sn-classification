"""Conversion between Hyrax's padded batch arrays and ParSNIP's astropy light curves."""

import numpy as np
from astropy.table import Table


def batch_to_tables(lightcurve, lengths, time_offset, redshift, mwebv, band_names) -> list[Table]:
    """Rebuild one astropy Table per object from a padded light-curve batch.

    Parameters
    ----------
    lightcurve : array-like, shape (batch, max_len, 4)
        Padded ``[time, flux, fluxerr, band_index]`` rows (see ``hyrax_lightcurves.LightCurveHATSDataset``).
    lengths : array-like, shape (batch,)
        Number of real (unpadded) observations per object.
    time_offset : array-like, shape (batch,)
        Added back to each object's times (see ``HyraxParsnip.prepare_inputs``).
    redshift, mwebv : array-like, shape (batch,)
        Per-object metadata.
    band_names : list[str]
        sncosmo band name for each band index.

    Returns
    -------
    list[Table]
        Light curves in the format expected by ``parsnip.preprocess_light_curve``. The
        ``object_id`` meta is the position in the batch.
    """
    lightcurve = _to_numpy(lightcurve)
    lengths = _to_numpy(lengths)
    time_offset = _to_numpy(time_offset).astype(np.float64)
    redshift = _to_numpy(redshift)
    mwebv = _to_numpy(mwebv)
    band_names = np.asarray(band_names)

    tables = []
    for i, n_obs in enumerate(lengths):
        rows = lightcurve[i, : int(n_obs)].astype(np.float64)
        tables.append(
            Table(
                {
                    "time": rows[:, 0] + time_offset[i],
                    "flux": rows[:, 1],
                    "fluxerr": rows[:, 2],
                    "band": band_names[rows[:, 3].astype(int)],
                },
                meta={"object_id": str(i), "redshift": float(redshift[i]), "mwebv": float(mwebv[i])},
            )
        )
    return tables


def _to_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)
