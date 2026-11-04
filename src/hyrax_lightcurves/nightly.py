"""Nightly coadds of visit-level light curves.

Port of ``coadd_nightly`` and ``sigma_clip_nightly`` from R. Chen's DP2 SN pipeline
(``DP2_SN_samples/pipeline/lclib.py``), on plain arrays. The results are the same, so a coadd of the
same visits reproduces that pipeline's ``lc_nightly``.
"""

import numpy as np
import pandas as pd


def night_of(mjd) -> np.ndarray:
    """Rubin dayObs convention (UTC-12h): a night at Cerro Pachon never straddles a boundary."""
    return np.floor(np.asarray(mjd, dtype=np.float64) - 0.5)


def _group_median(values, group, n_groups) -> np.ndarray:
    """Median of `values` per group id, broadcast back to every row."""
    return pd.Series(values).groupby(group).median().reindex(np.arange(n_groups)).to_numpy()[group]


def sigma_clip_nightly(group, flux, fluxerr, sigma_scale=5.0, maxiter=10, min_n=3):
    """Clip each group (integer ids 0..G-1, one per object/band/night) at a fixed pull.

    Per group: start from the median flux, keep ``|pull| <= sigma_scale``, recentre on the
    inverse-variance weighted mean of the kept points, and repeat until no group changes. Groups with
    fewer than `min_n` points are not clipped; a group that would lose every point keeps the one closest
    to its centre. Returns the boolean mask of kept points.
    """
    group, flux, fluxerr = np.asarray(group), np.asarray(flux, dtype=np.float64), np.asarray(fluxerr, dtype=np.float64)
    if len(group) == 0:
        return np.ones(0, dtype=bool)
    n_groups = group.max() + 1
    clippable = np.bincount(group, minlength=n_groups)[group] >= min_n
    weight = 1.0 / fluxerr**2
    keep = np.ones(len(group), dtype=bool)
    centre = _group_median(flux, group, n_groups)
    for _ in range(maxiter):
        pull = (flux - centre) / fluxerr
        new = (np.abs(pull) <= sigma_scale) | ~clippable
        empty = np.bincount(group, weights=new, minlength=n_groups)[group] == 0
        if empty.any():
            closest = pd.Series(np.where(empty, np.abs(pull), np.inf)).groupby(group).idxmin()
            new[closest.loc[np.unique(group[empty])].to_numpy()] = True
        converged = np.array_equal(new, keep)
        keep = new
        total = np.bincount(group, weights=weight * keep, minlength=n_groups)
        centre = (np.bincount(group, weights=weight * flux * keep, minlength=n_groups) / total)[group]
        if converged:
            break
    return keep


def coadd_nightly(row, time, flux, fluxerr, band_index, clip_sigma=5.0):
    """Inverse-variance weighted coadd of the same-night, same-band points of each object.

    `row` is the object of each point. Points must already be cleaned: no flagged points, finite values
    and positive errors. Points are clipped with `sigma_clip_nightly`, then each night's flux and time are
    the weighted means of the kept points, with error ``1 / sqrt(sum of weights)``.

    Returns ``(row, time, flux, fluxerr, band_index)`` of the coadds, sorted by object, then time.
    """
    row, band_index = np.asarray(row), np.asarray(band_index)
    time, flux, fluxerr = (np.asarray(a, dtype=np.float64) for a in (time, flux, fluxerr))
    if len(row) == 0:
        return row, time, flux, fluxerr, band_index

    points = pd.DataFrame({"row": row, "band": band_index, "night": night_of(time)})
    group = points.groupby(["row", "band", "night"], sort=False).ngroup().to_numpy()
    n_groups = group.max() + 1
    keep = sigma_clip_nightly(group, flux, fluxerr, sigma_scale=clip_sigma)

    weight = keep / fluxerr**2
    total = np.bincount(group, weights=weight, minlength=n_groups)
    coadd_time = np.bincount(group, weights=weight * time, minlength=n_groups) / total
    coadd_flux = np.bincount(group, weights=weight * flux, minlength=n_groups) / total
    coadd_err = 1.0 / np.sqrt(total)
    coadd_row, coadd_band = np.empty(n_groups, dtype=row.dtype), np.empty(n_groups, dtype=band_index.dtype)
    coadd_row[group], coadd_band[group] = row, band_index

    order = np.lexsort((coadd_time, coadd_row))
    return coadd_row[order], coadd_time[order], coadd_flux[order], coadd_err[order], coadd_band[order]
