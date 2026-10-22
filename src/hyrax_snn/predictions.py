from astropy.table import Table

from hyrax_lightcurves.predictions import load_class_predictions
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
    return load_class_predictions(results, CLASS_NAMES_FILENAME, metadata)
