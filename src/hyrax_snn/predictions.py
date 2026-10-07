import json
from pathlib import Path

import numpy as np
from astropy.table import Table, join

from hyrax_snn.model import CLASS_NAMES_FILENAME


def load_predictions(results, metadata: Table | None = None) -> Table:
    """Turn Hyrax inference results into a table of SuperNNova class probabilities.

    Parameters
    ----------
    results : hyrax ResultDataset
        The return value of ``h.infer()``.
    metadata : astropy Table, optional
        Extra per-object columns (e.g. from `hyrax_lightcurves.catalog_metadata`) joined on
        ``object_id``.

    Returns
    -------
    Table
        One row per object: ``object_id``, one probability column per class, and
        ``predicted_class`` (empty where there was nothing to classify).
    """
    with open(Path(results.data_location) / CLASS_NAMES_FILENAME) as f:
        class_names = json.load(f)

    probabilities = np.asarray(results.__get_all__(), dtype=np.float64)
    table = Table(probabilities, names=class_names)
    table.add_column(np.asarray(results.ids(), dtype=str), name="object_id", index=0)

    valid = np.isfinite(probabilities).all(axis=1)
    best = np.argmax(np.nan_to_num(probabilities, nan=-1), axis=1)
    table["predicted_class"] = np.where(valid, np.asarray(class_names)[best], "")

    if metadata is not None:
        table = join(table, metadata, keys="object_id", join_type="left")
    return table
