"""Class schemes for training classifiers on labeled catalogs (e.g. PLAsTiCC types).

- ``ia``: SNIa vs non-SNIa (every other type, including SNIa-91bg and SNIax)
- ``dp2``: SNIa, SNII, SNIbc and "other" (the types of the DP2 simulation, plus everything else)
- ``all``: the catalog types as they are
"""

import numpy as np

SCHEMES = ("ia", "dp2", "all")
DP2_CLASSES = ["SNIa", "SNII", "SNIbc"]


def class_labels(types, scheme: str) -> np.ndarray:
    """Training label of each catalog type under `scheme`."""
    types = np.asarray(types, dtype=str)
    if scheme == "ia":
        return np.where(types == "SNIa", "SNIa", "non-SNIa")
    if scheme == "dp2":
        return np.where(np.isin(types, DP2_CLASSES), types, "other")
    if scheme == "all":
        return types
    raise ValueError(f"Unknown class scheme {scheme!r}; use one of {SCHEMES}")


def scheme_classes(scheme: str, types=None) -> list[str]:
    """Class names of `scheme` in a fixed order (`all` needs the catalog `types`)."""
    if scheme == "ia":
        return ["SNIa", "non-SNIa"]
    if scheme == "dp2":
        return [*DP2_CLASSES, "other"]
    if scheme == "all":
        if types is None:
            raise ValueError("The 'all' scheme needs the catalog types")
        return sorted(set(np.asarray(types, dtype=str)))
    raise ValueError(f"Unknown class scheme {scheme!r}; use one of {SCHEMES}")
