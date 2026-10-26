import hashlib
import json
import logging
import os
import time
from pathlib import Path

import numpy as np

from hyrax_lightcurves.dataset import LightCurveHATSDataset
from hyrax_scone.heatmaps import (
    DEFAULT_SETTINGS,
    HEATMAP_SETTING_KEYS,
    HEATMAP_VERSION,
    LATER_SETTING_KEYS,
    default_n_jobs,
    make_heatmaps,
)

logger = logging.getLogger(__name__)


class SconeHATSDataset(LightCurveHATSDataset):
    """`LightCurveHATSDataset` plus each object's SCONE heatmap (field ``heatmap``).

    The light curves are read and cleaned as for the other models (``[data_set.LightCurveHATSDataset]``,
    with SCONE's band map and fluxes on zeropoint 27.5). The heatmaps (`hyrax_scone.heatmaps`, settings in
    ``[data_set.SconeHATSDataset]``) are all made when the dataset is created, with the GP fits in
    parallel, because the fits are slow and every training epoch needs them. With a ``cache_dir`` they are
    saved there and reused by any dataset with the same light curves and settings (training runs, resumed
    runs, classification).

    Fields
    ------
    heatmap : float32 array (2, n_wavelengths, n_times), all NaN where no heatmap could be made
    plus the fields of `LightCurveHATSDataset`.
    """

    def __init__(self, config: dict, data_location: Path | str | None = None):
        super().__init__(config, data_location)
        scone = config["data_set"]["SconeHATSDataset"]
        # Plain JSON values (the config holds TOML types), as stored in model directories.
        self.heatmap_settings = json.loads(json.dumps({key: scone[key] for key in HEATMAP_SETTING_KEYS}))
        n_jobs = int(scone["n_jobs"]) or default_n_jobs()
        self._heatmaps = self._load_or_make(scone["cache_dir"] or None, n_jobs)

    def _cache_key(self) -> str:
        """Hash of everything a heatmap depends on: the cleaned light curves (incl. flux scale and
        band map), Milky Way E(B-V), the heatmap settings and the version of the heatmap code."""
        digest = hashlib.sha1()
        for array in (self._lightcurve_data, self._starts, self._stops, self._mwebv):
            digest.update(np.ascontiguousarray(array).tobytes())
        heatmap_settings = {
            key: value
            for key, value in self.heatmap_settings.items()
            if not (key in LATER_SETTING_KEYS and value == DEFAULT_SETTINGS[key])
        }
        settings = [list(self.settings["band_map"].values()), heatmap_settings, HEATMAP_VERSION]
        digest.update(json.dumps(settings).encode())
        return digest.hexdigest()[:20]

    def _load_or_make(self, cache_dir, n_jobs: int) -> np.ndarray:
        path = Path(cache_dir) / f"heatmaps-{self._cache_key()}.npy" if cache_dir else None
        if path is not None and path.exists():
            heatmaps = np.load(path, mmap_mode="r")  # shared by the train and validate datasets
            if len(heatmaps) == len(self):
                logger.info(f"SCONE heatmaps for {len(self)} objects from {path}")
                return heatmaps

        bands = np.array(list(self.settings["band_map"].values()))
        lightcurves = []
        for i in range(len(self)):
            lc = self.get_lightcurve(i)
            lightcurves.append((lc[:, 0], lc[:, 1], lc[:, 2], bands[lc[:, 3].astype(int)], float(self._mwebv[i])))
        start = time.time()
        logger.info(f"Making SCONE heatmaps for {len(self)} objects on {n_jobs} processes")
        heatmaps = make_heatmaps(lightcurves, self.heatmap_settings, n_jobs=n_jobs)
        n_failed = int((~np.isfinite(heatmaps).all(axis=(1, 2, 3))).sum())
        logger.info(
            f"SCONE heatmaps: {len(self) - n_failed} made, {n_failed} failed, in {time.time() - start:.0f} s"
        )

        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            partial = path.with_name(f"{path.stem}.{os.getpid()}.partial.npy")
            np.save(partial, heatmaps)
            partial.replace(path)  # complete files only, even if several processes write
            logger.info(f"Saved SCONE heatmaps to {path}")
        return heatmaps

    def get_heatmap(self, idx: int) -> np.ndarray:
        return np.array(self._heatmaps[idx], dtype=np.float32)

    def has_heatmap(self) -> np.ndarray:
        """Whether each object has a heatmap."""
        return np.isfinite(np.asarray(self._heatmaps)).all(axis=(1, 2, 3))

