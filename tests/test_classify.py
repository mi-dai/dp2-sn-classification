import numpy as np
from conftest import N_OBJECTS

from hyrax_parsnip import (
    catalog_metadata,
    classify,
    configure,
    load_classifier,
    load_predictions,
    train_classifier,
)


def test_train_save_load_classify(hyrax_instance, hats_catalog, tmp_path):
    configure(hyrax_instance, hats_catalog, pretrained="plasticc")
    labels = catalog_metadata(hats_catalog, ["type"])
    predictions = load_predictions(hyrax_instance.infer(), metadata=labels)

    assert len(predictions) == N_OBJECTS
    assert set(predictions["type"]) == {"fast", "slow"}

    classifier, out_of_sample = train_classifier(
        predictions, label_column="type", num_folds=2, min_child_weight=1.0
    )
    assert len(out_of_sample) == N_OBJECTS

    path = tmp_path / "classifier" / "parsnip_classifier.pkl"
    classifier.write(str(path))
    probabilities = classify(load_classifier(path), predictions)

    assert probabilities.colnames == ["object_id", "fast", "slow"]
    total = np.asarray(probabilities["fast"]) + np.asarray(probabilities["slow"])
    np.testing.assert_allclose(total, 1.0)


def test_classify_marks_unprocessable_rows(hyrax_instance, hats_catalog):
    configure(hyrax_instance, hats_catalog, pretrained="plasticc")
    labels = catalog_metadata(hats_catalog, ["type"])
    predictions = load_predictions(hyrax_instance.infer(), metadata=labels)
    classifier, _ = train_classifier(predictions, label_column="type", num_folds=2, min_child_weight=1.0)

    predictions["s1"][0] = np.nan
    probabilities = classify(classifier, predictions)

    assert np.isnan(probabilities["fast"][0])
    assert np.isfinite(probabilities["fast"][1:]).all()


def test_load_classifier_in_fresh_process(hyrax_instance, hats_catalog, tmp_path):
    """Reloading after torch has run must not crash (conflicting OpenMP runtimes on macOS)."""
    import subprocess
    import sys

    configure(hyrax_instance, hats_catalog, pretrained="plasticc")
    labels = catalog_metadata(hats_catalog, ["type"])
    predictions = load_predictions(hyrax_instance.infer(), metadata=labels)
    classifier, _ = train_classifier(predictions, label_column="type", num_folds=2, min_child_weight=1.0)
    classifier_path = tmp_path / "classifier.pkl"
    predictions_path = tmp_path / "predictions.ecsv"
    classifier.write(str(classifier_path))
    predictions.write(predictions_path)

    script = f"""
import torch
torch.ones(500, 500) @ torch.ones(500, 500)
from astropy.table import Table
from hyrax_parsnip import classify, load_classifier
probabilities = classify(load_classifier({str(classifier_path)!r}), Table.read({str(predictions_path)!r}))
assert len(probabilities) == {N_OBJECTS}
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stderr[-2000:]
