"""SuperNNova input features for one light curve, in numpy.

A port of SuperNNova's on-the-fly preprocessing (``validate_onthefly.format_data``,
``make_dataset.pivot_dataframe_single_from_df`` and ``training_utils.normalize_arr``),
which is pandas-based and pins pandas < 3. The steps and their quirks are kept so the
outputs match SuperNNova's; tests/test_snn_reference.py checks this against the original.
"""

import numpy as np

from hyrax_snn.pretrained import SNNSettings

# Observations closer than this (days) to the start of their group share one time step.
GROUP_WINDOW_DAYS = 0.33


def apply_time_window(time, flux, fluxerr, window) -> np.ndarray:
    """Mask of observations within `window` = (before, after) days of the max-S/N point.

    The Fink models were trained on photometry within (-30, +100) days of an initial peak
    estimate (SuperNNova's `photo_window`); on-the-fly classification has no such cut.
    """
    peak = time[np.argmax(flux / fluxerr)]
    return (time >= peak + window[0]) & (time <= peak + window[1])


def select_observations(time, flux, fluxerr, bands, time_window=None, detection_snr=None):
    """Observations kept for SuperNNova: above `detection_snr` (if set), then within `time_window`
    days of the max-S/N point (if set). The same selection is used for training and inference."""
    if detection_snr is not None:
        mask = flux / fluxerr > detection_snr
        time, flux, fluxerr, bands = time[mask], flux[mask], fluxerr[mask], bands[mask]
    if len(time) and time_window:
        mask = apply_time_window(time, flux, fluxerr, time_window)
        time, flux, fluxerr, bands = time[mask], flux[mask], fluxerr[mask], bands[mask]
    return time, flux, fluxerr, bands


def _group_times(time: np.ndarray) -> np.ndarray:
    """SuperNNova's sequential grouping: a new time step starts at the first observation,
    at any repeated timestamp, or more than GROUP_WINDOW_DAYS after the step's start."""
    grouped = np.empty_like(time)
    last_change = time[0]
    for i, t in enumerate(time):
        dt = 0.0 if i == 0 else t - time[i - 1]
        if dt == 0 or (t - last_change) > GROUP_WINDOW_DAYS:
            grouped[i] = t
            last_change = t
        else:
            grouped[i] = grouped[i - 1]
    return grouped


def build_features(
    time,
    flux,
    fluxerr,
    bands,
    settings: SNNSettings,
    mwebv: float = 0.0,
    redshift: float = 0.0,
    redshift_error: float = 0.0,
    normalize_features: bool = True,
) -> np.ndarray | None:
    """Normalized (n_steps, n_features) float32 input for one light curve, or None if empty.

    With ``normalize_features=False`` the raw features are returned in `settings.all_features`
    order (for computing normalization statistics).

    `bands` are SuperNNova filter names (e.g. "u", ..., "Y"); observations in other bands
    are ignored. Fluxes must be in the units the model expects.
    """
    time = np.asarray(time, dtype=np.float64)
    flux = np.asarray(flux, dtype=np.float64)
    fluxerr = np.asarray(fluxerr, dtype=np.float64)
    bands = np.asarray(bands).astype(str)

    keep = np.isin(bands, settings.filters)
    time, flux, fluxerr, bands = time[keep], flux[keep], fluxerr[keep], bands[keep]
    if len(time) == 0:
        return None

    order = np.argsort(time, kind="stable")
    time, flux, fluxerr, bands = time[order], flux[order], fluxerr[order], bands[order]
    grouped = _group_times(time)

    # One time step per group; within a (group, filter) keep the smallest-error point.
    steps = np.unique(grouped)
    step_index = np.searchsorted(steps, grouped)
    filters = settings.filters
    n_steps, n_filters = len(steps), len(filters)
    step_flux = np.zeros((n_steps, n_filters))
    step_err = np.zeros((n_steps, n_filters))
    best_err = np.full((n_steps, n_filters), np.inf)
    filter_index = np.array([filters.index(b) for b in bands])
    for s, f, fl, er in zip(step_index, filter_index, flux, fluxerr):
        if er < best_err[s, f]:
            best_err[s, f] = er
            step_flux[s, f] = fl
            step_err[s, f] = er
    observed = np.isfinite(best_err)

    delta_time = np.diff(steps, prepend=steps[0])
    combination = ["".join(f for f, o in zip(filters, row) if o) for row in observed]

    # SuperNNova stores its table as float32 before normalizing.
    f32 = lambda a: np.asarray(a, dtype=np.float32).astype(np.float64)  # noqa: E731
    columns = {}
    for j, f in enumerate(filters):
        columns[f"FLUXCAL_{f}"] = f32(step_flux[:, j])
        columns[f"FLUXCALERR_{f}"] = f32(step_err[:, j])
    columns["delta_time"] = f32(delta_time)
    constants = {
        "MWEBV": mwebv,
        "HOSTGAL_SPECZ": redshift if settings.redshift == "zspe" else 0.0,
        "HOSTGAL_SPECZ_ERR": redshift_error if settings.redshift == "zspe" else 0.0,
        "HOSTGAL_PHOTOZ": 0.0,
        "HOSTGAL_PHOTOZ_ERR": 0.0,
    }
    for name, value in constants.items():
        columns[name] = f32(np.full(n_steps, value))
    for combo in settings.filter_combinations:
        columns[combo] = np.array([c == combo for c in combination], dtype=np.float64)

    missing = [f for f in settings.all_features if f not in columns]
    if missing:
        raise ValueError(f"Cannot build SuperNNova features {missing}")
    x = np.stack([columns[f] for f in settings.all_features], axis=1)
    if not normalize_features:
        return x

    x = normalize(x, settings)
    model_columns = [settings.all_features.index(f) for f in settings.model_features]
    return x[:, model_columns].astype(np.float32)


def normalize(x: np.ndarray, settings: SNNSettings) -> np.ndarray:
    """SuperNNova's `normalize_arr` on an (n_steps, all_features) array (columns not normalized are unchanged)."""
    if settings.norm == "none":
        return x
    idx = [settings.all_features.index(f) for f in settings.features_to_normalize]
    arr_min, arr_mean, arr_std = settings.arr_norm.T
    values = np.clip(x[:, idx], arr_min, np.inf)

    if settings.norm == "global":
        normed = (np.log(values - arr_min + 1e-5) - arr_mean) / arr_std
    elif settings.norm in ("cosmo", "cosmo_quantile"):
        # Fluxes and errors (all but the last, delta_time) scaled together per light curve;
        # delta_time log-normalized as in "global".
        scale = values[:, :-1].max() if settings.norm == "cosmo" else np.quantile(values[:, :-1], 0.99)
        normed = values.copy()
        normed[:, :-1] = values[:, :-1] / scale
        normed[:, -1] = (np.log(values[:, -1] - arr_min[-1] + 1e-5) - arr_mean[-1]) / arr_std[-1]
    else:
        raise ValueError(f"Unknown SuperNNova normalization {settings.norm!r}")

    x = x.copy()
    x[:, idx] = normed
    return x
