import numpy as np
from astropy.table import Table


def catalog_metadata(catalog_path, columns: list[str], id_column: str = "object_id") -> Table:
    """Read per-object columns (e.g. a class label) from a HATS catalog.

    The id column is renamed to ``object_id``, like the ids in the models' predictions.
    """
    import lsdb

    frame = lsdb.open_catalog(str(catalog_path), columns=[id_column, *columns]).compute()
    table = Table({"object_id": np.asarray(frame[id_column], dtype=str)})
    for column in columns:
        values = frame[column].to_numpy()
        # pyarrow-backed string columns come back as object arrays.
        table[column] = np.asarray(values, dtype=str) if values.dtype == object else values
    return table
