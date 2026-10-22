import json
from pathlib import Path

import numpy as np
from astropy.table import Table, join


def load_class_predictions(results, class_names_filename: str, metadata: Table | None = None) -> Table:
    """Turn Hyrax inference results of a classifier into a table of class probabilities.

    Parameters
    ----------
    results : hyrax ResultDataset
        The return value of ``h.infer()``.
    class_names_filename : str
        JSON list of the class names, written by the model next to the inference results.
    metadata : astropy Table, optional
        Extra per-object columns (e.g. from `hyrax_lightcurves.catalog_metadata`) joined on
        ``object_id``.

    Returns
    -------
    Table
        One row per object: ``object_id``, one probability column per class, and
        ``predicted_class`` (empty where there was nothing to classify).
    """
    with open(Path(results.data_location) / class_names_filename) as f:
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
