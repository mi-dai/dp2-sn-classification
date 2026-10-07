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
    photoz : float, per-object host photo-z prior for photo-z models (NaN without ``photoz_column``)
    mwebv : float (0 when no ``mwebv_column`` is configured)
    label : str (requires ``label_column``)
    label_index : int, index into `label_classes` of the label mapped by ``label_scheme``
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
            # Load only the nested sub-columns in use; survey tables carry many more.
            nested = [settings[key] for key in ("time_column", "flux_column", "fluxerr_column", "band_column")]
            nested += list(settings.get("flag_columns") or [])
            columns = [settings["id_column"]]
            columns += [f"{settings['lightcurve_column']}.{name}" for name in nested]
            for key in ("redshift_column", "photoz_column", "mwebv_column", "label_column"):
                if settings.get(key) and settings[key] not in columns:  # e.g. photoz_column = redshift_column
                    columns.append(settings[key])
            kwargs["columns"] = columns

        catalog = lsdb.open_catalog(self.data_location, **kwargs)
        # The HATS index (_healpix_29) is not unique, so switch to positional rows.
        return catalog.compute().reset_index(drop=True)

    def _build_arrays(self, frame, settings: dict):
        n_objects = len(frame)

        flat = frame[settings["lightcurve_column"]].nest.to_flat()
        row = flat.index.to_numpy()
        time = flat[settings["time_column"]].to_numpy(dtype=np.float64, na_value=np.nan)
        flux = flat[settings["flux_column"]].to_numpy(dtype=np.float64, na_value=np.nan)
        fluxerr = flat[settings["fluxerr_column"]].to_numpy(dtype=np.float64, na_value=np.nan)
        band_lookup = {name: i for i, name in enumerate(self.band_names)}
        band_index = (
            flat[settings["band_column"]].astype(str).map(band_lookup).fillna(-1).to_numpy(dtype=np.int64)
        )

        # Convert fluxes to the zeropoint the ParSNIP model was trained with.
        flux_scale = float(settings.get("flux_scale", 1.0))
        flux = flux * flux_scale
        fluxerr = fluxerr * flux_scale

        flagged = np.zeros(len(flat), dtype=bool)
        for column in settings.get("flag_columns") or []:
            flagged |= flat[column].fillna(False).to_numpy(dtype=bool)

        valid = (
            ~flagged
            & (band_index >= 0)
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
        self._photoz = self._column(frame, settings.get("photoz_column"), np.nan)[keep]
        self._mwebv = self._column(frame, settings.get("mwebv_column"), 0.0)[keep]
        label_column = settings.get("label_column")
        self._labels = frame[label_column].astype(str).to_numpy()[keep] if label_column else None
        self.label_classes, self._label_index = None, None
        if self._labels is not None:
            # Integer classes for training (Hyrax passes model inputs as numeric arrays).
            from hyrax_parsnip.labels import class_labels, scheme_classes

            scheme = settings.get("label_scheme") or "all"
            mapped = class_labels(self._labels, scheme)
            self.label_classes = list(settings.get("label_classes") or scheme_classes(scheme, mapped))
            lookup = {name: i for i, name in enumerate(self.label_classes)}
            unknown = sorted(set(mapped) - set(lookup))
            if unknown:
                raise ValueError(f"Labels {unknown} are not in label_classes {self.label_classes}")
            self._label_index = np.array([lookup[m] for m in mapped], dtype=np.int64)

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

    def get_photoz(self, idx: int) -> float:
        return np.float64(self._photoz[idx])

    def get_mwebv(self, idx: int) -> float:
        return np.float64(self._mwebv[idx])

    def get_label(self, idx: int) -> str:
        if self._labels is None:
            raise RuntimeError("No `label_column` configured for ParsnipHATSDataset.")
        return self._labels[idx]

    def get_label_index(self, idx: int) -> np.int64:
        """Class index of the label under `label_scheme`, into `label_classes`."""
        if self._label_index is None:
            raise RuntimeError("No `label_column` configured for ParsnipHATSDataset.")
        return self._label_index[idx]

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
