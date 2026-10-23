"""SCONE's Gaussian-process heatmaps (Qu et al. 2021), ported from helenqu/scone ``create_heatmaps``.

Each light curve is fit with a 2-D Matern-3/2 Gaussian process over (time, wavelength) (george,
as SCONE's ``build_gp``, itself from avocado). The GP mean and standard deviation are predicted on a
grid of `n_wavelengths` wavelengths × `n_times` times around the light curve's peak, corrected for
Milky Way extinction, and divided by the maximum |mean|: a (2, n_wavelengths, n_times) image.

Differences from upstream SCONE:
- The time window is centred on the S/N²-weighted time of the detections (upstream's own estimate
  in ``SaveTriggerToCSV``, with detections at flux/fluxerr > `peak_detection_snr`), not the
  simulation's true peak, which real data don't have.
- Milky Way extinction multiplies the GP flux by 10**(0.4 A_λ); upstream adds A_λ (magnitudes) to
  the flux, see `extinction_correct`.
- Both channels are divided by the maximum of |mean|; upstream divides by the maximum of the mean
  (``data_utils.get_images``). For supernovae, whose positive peak dominates, the two agree; for
  variables with mostly negative difference flux (about 12% of PLAsTiCC) upstream's maximum is a small
  positive number and the heatmap values reach ±1e8 and beyond, which stalls training.
- The std channel is capped at `max_std` (default 2, i.e. twice the largest |flux|). For noise-dominated
  light curves it reaches 1e14; such values wreck BatchNorm's running statistics, so that the trained
  network gives the same output for every object in evaluation mode.
- Light curves the GP can't fit give no heatmap (NaN).

The flux constants (padding error, S/N floor) are SCONE's, in FLUXCAL units (zeropoint 27.5), so
fluxes must be on zeropoint 27.5 (Rubin nJy × `hyrax_lightcurves.NJY_TO_ZP27_5`).
"""

import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

# Effective wavelengths (Å) of the LSST bands in SCONE (``get_band_to_wave("LSST")``; 2017-era bands).
LSST_WAVELENGTHS = {"u": 3670.69, "g": 4826.85, "r": 6223.24, "i": 7545.98, "z": 8590.90, "Y": 9710.28}

# Heatmap settings (``[data_set.SconeHATSDataset]``); SCONE's values.
DEFAULT_SETTINGS = {
    "n_wavelengths": 32,
    "n_times": 180,
    "wavelength_range": [3000.0, 10100.0],
    # Days before and after the peak time covered by the heatmap.
    "time_window": [-30.0, 150.0],
    "peak_detection_snr": 5.0,
    "time_scale_guess": 20.0,
    "max_std": 2.0,
}
HEATMAP_SETTING_KEYS = tuple(DEFAULT_SETTINGS)

# Changes whenever the heatmaps computed for the same settings change, so cached ones are remade.
HEATMAP_VERSION = 3


def peak_time(time, flux, fluxerr, detection_snr: float = 5.0) -> float:
    """S/N²-weighted mean time of the detections (flux/fluxerr > `detection_snr`); the time of the
    highest S/N observation if there is none."""
    snr = flux / fluxerr
    detected = snr > detection_snr
    if not detected.any():
        return float(time[np.argmax(snr)])
    weights = snr[detected] ** 2
    return float(np.sum(time[detected] * weights) / np.sum(weights))


def fit_gp(time, flux, fluxerr, wavelength, time_scale_guess: float = 20.0):
    """SCONE's ``build_gp``: a george GP with a Matern-3/2 kernel in (time, wavelength), fit by maximum
    likelihood. Returns the fitted GP, or None if the fit fails."""
    import george
    from george import kernels
    from scipy.optimize import minimize

    signal_to_noise = np.abs(flux) / np.sqrt(fluxerr**2 + (0.01 * np.max(flux)) ** 2)
    scale = np.abs(flux[np.argmax(signal_to_noise)])
    if not np.isfinite(scale) or scale == 0:
        return None

    kernel = (0.5 * scale) ** 2 * kernels.Matern32Kernel([time_scale_guess**2, 6000.0**2], ndim=2)
    gp = george.GP(kernel)
    guess = gp.get_parameter_vector()
    try:
        gp.compute(np.vstack([time, wavelength]).T, fluxerr)

        def neg_ln_like(p):
            gp.set_parameter_vector(p)
            return -gp.log_likelihood(flux)

        def grad_neg_ln_like(p):
            gp.set_parameter_vector(p)
            return -gp.grad_log_likelihood(flux)

        # As upstream: amplitude within e^±10 of the guess, time scale 1-1000 d, wavelength scale free.
        bounds = [(guess[0] - 10, guess[0] + 10), (0, np.log(1000**2)), (None, None)]
        result = minimize(neg_ln_like, guess, jac=grad_neg_ln_like, bounds=bounds)
        gp.set_parameter_vector(result.x)
    except (ValueError, np.linalg.LinAlgError):
        return None
    return gp


def ccm89_extinction(ebv: float, wavelength) -> np.ndarray:
    """A_λ (magnitudes) for E(B-V) `ebv`, Cardelli, Clayton & Mathis (1989) with R_V = 3.1
    (SCONE's ``get_extinction``)."""
    x = 10000.0 / np.asarray(wavelength, dtype=np.float64)  # inverse microns
    a = np.zeros_like(x)
    b = np.zeros_like(x)

    ir = (x >= 0.3) & (x < 1.1)
    a[ir] = 0.574 * x[ir] ** 1.61
    b[ir] = -0.527 * x[ir] ** 1.61

    optical = (x >= 1.1) & (x < 3.3)
    y = x[optical] - 1.82
    a[optical] = np.polyval([0.32999, -0.77530, 0.01979, 0.72085, -0.02427, -0.50447, 0.17699, 1.0], y)
    b[optical] = np.polyval([-2.09002, 5.30260, -0.62251, -5.38434, 1.07233, 2.28305, 1.41338, 0.0], y)

    uv = (x >= 3.3) & (x < 8.0)
    xu = x[uv]
    fa = np.where(xu >= 5.9, -0.04473 * (xu - 5.9) ** 2 - 0.009779 * (xu - 5.9) ** 3, 0.0)
    fb = np.where(xu >= 5.9, 0.21300 * (xu - 5.9) ** 2 + 0.120700 * (xu - 5.9) ** 3, 0.0)
    a[uv] = 1.752 - 0.316 * xu - 0.104 / ((xu - 4.67) ** 2 + 0.341) + fa
    b[uv] = -3.090 + 1.825 * xu + 1.206 / ((xu - 4.62) ** 2 + 0.263) + fb

    far_uv = (x >= 8.0) & (x <= 10.0)
    xf = x[far_uv] - 8.0
    a[far_uv] = -1.073 - 0.628 * xf + 0.137 * xf**2 - 0.070 * xf**3
    b[far_uv] = 13.670 + 4.257 * xf - 0.420 * xf**2 + 0.374 * xf**3

    r_v = 3.1
    return r_v * ebv * (a + b / r_v)


def extinction_correct(mean, std, ebv: float, wavelength):
    """Correct the GP mean and std (n_wavelengths, n_times) for Milky Way extinction.

    Extinction dims the flux by 10**(-0.4 A_λ), so the intrinsic flux is the observed one times
    10**(0.4 A_λ). Upstream SCONE instead adds A_λ (in magnitudes) to the flux,
    ``predictions + ext`` in ``CreateHeatmapsBase._get_predictions_heatmap``; after the heatmap is
    divided by its maximum flux that adds a near-constant offset of order A_λ / peak flux.
    """
    factor = 10 ** (0.4 * ccm89_extinction(ebv, wavelength))[:, None]
    return mean * factor, std * factor


def make_heatmap(time, flux, fluxerr, bands, mwebv: float = 0.0, settings: dict | None = None) -> np.ndarray:
    """SCONE heatmap of one light curve: float32 (2, n_wavelengths, n_times) with the GP mean and std,
    both divided by the maximum |mean|; all NaN if no heatmap can be made.

    `bands` are SCONE band names (keys of `LSST_WAVELENGTHS`); observations in other bands are dropped.
    """
    settings = {**DEFAULT_SETTINGS, **(settings or {})}
    shape = (2, int(settings["n_wavelengths"]), int(settings["n_times"]))
    failed = np.full(shape, np.nan, dtype=np.float32)

    bands = np.asarray(bands)
    known = np.isin(bands, list(LSST_WAVELENGTHS))
    time, flux, fluxerr, bands = (np.asarray(a)[known] for a in (time, flux, fluxerr, bands))
    if len(time) < 2:
        return failed
    wavelength = np.array([LSST_WAVELENGTHS[b] for b in bands])
    peak = peak_time(time, flux, fluxerr, float(settings["peak_detection_snr"]))

    # As upstream: the GP is fit to the whole light curve plus zero-flux points (error 0.1) 100 days
    # before the first and after the last observation, in the r band.
    pad_time = [time.min() - 100, time.max() + 100]
    time = np.concatenate([time, pad_time])
    flux = np.concatenate([flux, [0.0, 0.0]])
    fluxerr = np.concatenate([fluxerr, [0.1, 0.1]])
    wavelength = np.concatenate([wavelength, [LSST_WAVELENGTHS["r"]] * 2])

    gp = fit_gp(time, flux, fluxerr, wavelength, float(settings["time_scale_guess"]))
    if gp is None:
        return failed

    before, after = settings["time_window"]
    grid_times = np.linspace(peak + before, peak + after, shape[2])
    grid_wavelengths = np.linspace(*settings["wavelength_range"], shape[1])
    grid = np.column_stack([np.tile(grid_times, shape[1]), np.repeat(grid_wavelengths, shape[2])])
    try:
        mean, variance = gp.predict(flux, grid, return_var=True)
    except (ValueError, np.linalg.LinAlgError):
        return failed
    mean = mean.reshape(shape[1], shape[2])
    std = np.sqrt(np.clip(variance, 0, None)).reshape(shape[1], shape[2])
    mean, std = extinction_correct(mean, std, float(mwebv), grid_wavelengths)

    # Both channels divided by the maximum |mean| (SCONE's data_utils.get_images: the maximum of the
    # mean, which is the same for supernovae; see the module docstring).
    scale = np.abs(mean).max()
    if not np.isfinite(scale) or scale <= 0:
        return failed
    with np.errstate(over="ignore"):
        heatmap = (np.stack([mean, std]) / scale).astype(np.float32)
    # A GP whose flux vanishes everywhere overflows: no heatmap.
    if not np.isfinite(heatmap).all():
        return failed
    heatmap[1] = np.minimum(heatmap[1], float(settings["max_std"]))
    return heatmap


def _heatmap_task(args):
    time, flux, fluxerr, bands, mwebv, settings = args
    return make_heatmap(time, flux, fluxerr, bands, mwebv, settings)


def default_n_jobs() -> int:
    """CPUs available to this process: Slurm's ``SLURM_CPUS_PER_TASK`` if set, else the CPU affinity.

    The affinity alone can be the whole node (256 threads on a Perlmutter CPU node) when Slurm gives
    the job a full node, and one GP process per thread ran out of memory."""
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK", "")
    if slurm_cpus.isdigit() and int(slurm_cpus) > 0:
        return int(slurm_cpus)
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1


def make_heatmaps(lightcurves, settings: dict | None = None, n_jobs: int = 1) -> np.ndarray:
    """Heatmaps for many light curves, (n, 2, n_wavelengths, n_times) float32.

    `lightcurves` is a sequence of ``(time, flux, fluxerr, bands, mwebv)``; with `n_jobs` > 1 the
    GP fits run in a process pool.
    """
    tasks = [(*lc, settings) for lc in lightcurves]
    if n_jobs <= 1 or len(tasks) < 2:
        results = [_heatmap_task(t) for t in tasks]
    else:
        with ProcessPoolExecutor(max_workers=n_jobs) as pool:
            results = list(pool.map(_heatmap_task, tasks, chunksize=max(1, len(tasks) // (8 * n_jobs))))
    settings = {**DEFAULT_SETTINGS, **(settings or {})}
    shape = (len(results), 2, int(settings["n_wavelengths"]), int(settings["n_times"]))
    return np.stack(results) if results else np.zeros(shape, dtype=np.float32)
