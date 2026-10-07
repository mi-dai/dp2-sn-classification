import numpy as np


def prepare_inputs(data_dict):
    """Unpack a collated `LightCurveHATSDataset` batch into arrays
    ``(lightcurve, lengths, time_offset, redshift, mwebv, photoz)``, float32 except lengths.

    Hyrax may place inputs on devices without float64 support (e.g. MPS), so each
    light curve's times are split into an integer-day ``time_offset`` (exact in
    float32) and residuals with seconds-level precision over multi-year light curves.
    """
    data = data_dict["data"]
    lengths = np.asarray(data["lengths"], dtype=np.int64)
    n_objects = len(lengths)
    lightcurve = np.array(data["lightcurve"], dtype=np.float64)

    time_offset = np.zeros(n_objects)
    for i, n_obs in enumerate(lengths):
        if n_obs:
            time_offset[i] = np.floor(lightcurve[i, :n_obs, 0].min())
            lightcurve[i, :n_obs, 0] -= time_offset[i]

    redshift = np.asarray(data.get("redshift", np.full(n_objects, np.nan)))
    mwebv = np.asarray(data.get("mwebv", np.zeros(n_objects)))
    photoz = np.asarray(data.get("photoz", np.full(n_objects, np.nan)))
    return (
        lightcurve.astype(np.float32),
        lengths,
        time_offset.astype(np.float32),
        redshift.astype(np.float32),
        mwebv.astype(np.float32),
        photoz.astype(np.float32),
    )
