from astropy.table import Table

from hyrax_lightcurves.predictions import load_class_predictions
from hyrax_scone.model import CLASS_NAMES_FILENAME


def load_predictions(results, metadata: Table | None = None) -> Table:
    """Turn Hyrax inference results into a table of SCONE class probabilities: ``object_id``, one
    probability column per class and ``predicted_class`` (empty where there was nothing to
    classify), joined with `metadata` on ``object_id`` if given."""
    return load_class_predictions(results, CLASS_NAMES_FILENAME, metadata)
