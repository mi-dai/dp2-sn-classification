import logging
from pathlib import Path

import numpy as np
from hyrax.datasets import HyraxDataset

logger = logging.getLogger(__name__)

# Columns of the per-object array returned by `get_lightcurve`.
LIGHTCURVE_COLUMNS = ("time", "flux", "fluxerr", "band_index")


class ParsnipHATSDataset(HyraxDataset):
    """Hyrax dataset serving light curves from a nested HATS catalog to ParSNIP.

    The catalog is opened with ``lsdb.open_catalog`` and materialized once. Each object
    must have a nested light-curve column (time, flux, fluxerr, band) plus per-object
    id and redshift columns. Column names and the band mapping come from
    ``config["data_set"]["ParsnipHATSDataset"]``.

    Fields
    ------
    object_id : str
    redshift : float
    mwebv : float (0 when no ``mwebv_column`` is configured)
    label : str (requires ``label_column``)
    lightcurve : float64 array of shape (n_obs, 4) with columns
        ``LIGHTCURVE_COLUMNS``; ``band_index`` indexes into ``band_map``.
    """

    def __init__(self, config: dict, data_location: Path | str | None = None):
        if not data_location:
            raise ValueError("A `data_location` pointing to a HATS catalog must be provided.")

        self.data_location = str(data_location)
        settings = config["data_set"]["ParsnipHATSDataset"]
        self.settings = settings
        self.band_names = list(settings["band_map"].keys())

        frame = self._open_catalog(settings)
        self._build_arrays(frame, settings)

        super().__init__(config)

    def _open_catalog(self, settings: dict):
        import lsdb

        kwargs = dict(settings.get("open_catalog_kwargs") or {})
        if "columns" not in kwargs:
            columns = [settings["id_column"], settings["lightcurve_column"]]
            for key in ("redshift_column", "mwebv_column", "label_column"):
                if settings.get(key):
                    columns.append(settings[key])
            kwargs["columns"] = columns

        catalog = lsdb.open_catalog(self.data_location, **kwargs)
        # The HATS index (_healpix_29) is not unique, so switch to positional rows.
        return catalog.compute().reset_index(drop=True)

    def _build_arrays(self, frame, settings: dict):
        n_objects = len(frame)

        flat = frame[settings["lightcurve_column"]].nest.to_flat()
        row = flat.index.to_numpy()
        time = flat[settings["time_column"]].to_numpy(dtype=np.float64)
        flux = flat[settings["flux_column"]].to_numpy(dtype=np.float64)
        fluxerr = flat[settings["fluxerr_column"]].to_numpy(dtype=np.float64)
        band_lookup = {name: i for i, name in enumerate(self.band_names)}
        band_index = (
            flat[settings["band_column"]].astype(str).map(band_lookup).fillna(-1).to_numpy(dtype=np.int64)
        )

        valid = (
            (band_index >= 0)
            & np.isfinite(time)
            & np.isfinite(flux)
            & np.isfinite(fluxerr)
            & (fluxerr > 0)
        )
        row, time, flux, fluxerr, band_index = (
            row[valid],
            time[valid],
            flux[valid],
            fluxerr[valid],
            band_index[valid],
        )

        # to_flat keeps rows grouped by object, so each object is a contiguous slice.
        offsets = np.searchsorted(row, np.arange(n_objects + 1))
        counts = np.diff(offsets)

        redshift = self._column(frame, settings.get("redshift_column"), np.nan)
        keep = counts >= settings["min_observations"]
        if settings["require_redshift"]:
            keep &= np.isfinite(redshift)

        n_dropped = int(n_objects - keep.sum())
        if n_dropped:
            logger.info(
                f"ParsnipHATSDataset: dropped {n_dropped}/{n_objects} objects with fewer than "
                f"{settings['min_observations']} usable observations or no redshift."
            )

        self._lightcurve_data = np.stack([time, flux, fluxerr, band_index], axis=1)
        self._starts = offsets[:-1][keep]
        self._stops = offsets[1:][keep]
        self._object_ids = frame[settings["id_column"]].astype(str).to_numpy()[keep]
        self._redshift = redshift[keep]
        self._mwebv = self._column(frame, settings.get("mwebv_column"), 0.0)[keep]
        label_column = settings.get("label_column")
        self._labels = frame[label_column].astype(str).to_numpy()[keep] if label_column else None

    @staticmethod
    def _column(frame, column, fill_value) -> np.ndarray:
        if not column:
            return np.full(len(frame), fill_value, dtype=np.float64)
        return frame[column].to_numpy(dtype=np.float64, na_value=np.nan)

    def __len__(self) -> int:
        return len(self._object_ids)

    def get_object_id(self, idx: int) -> str:
        return self._object_ids[idx]

    def get_redshift(self, idx: int) -> float:
        return np.float64(self._redshift[idx])

    def get_mwebv(self, idx: int) -> float:
        return np.float64(self._mwebv[idx])

    def get_label(self, idx: int) -> str:
        if self._labels is None:
            raise RuntimeError("No `label_column` configured for ParsnipHATSDataset.")
        return self._labels[idx]

    def get_lightcurve(self, idx: int) -> np.ndarray:
        return self._lightcurve_data[self._starts[idx] : self._stops[idx]]

    def collate_lightcurve(self, samples: list[dict]) -> dict:
        """Zero-pad variable-length light curves into one (batch, max_len, 4) array."""
        lightcurves = [s["lightcurve"] for s in samples]
        lengths = np.array([len(lc) for lc in lightcurves], dtype=np.int64)

        padded = np.zeros((len(lightcurves), max(lengths.max(), 1), len(LIGHTCURVE_COLUMNS)), dtype=np.float64)
        for i, lc in enumerate(lightcurves):
            padded[i, : len(lc)] = lc

        return {"lightcurve": padded, "lengths": lengths}
