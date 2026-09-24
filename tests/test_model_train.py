import numpy as np
import parsnip
import torch
from conftest import N_OBJECTS

from hyrax_parsnip import configure


def test_train_from_scratch_then_infer(hyrax_instance, hats_catalog, tmp_path):
    h = hyrax_instance
    configure(h, hats_catalog, pretrained=False, groups=("train", "infer"))
    h.set_config("train.epochs", 1)

    model = h.train()
    assert np.isfinite(model.final_training_metrics["loss"])

    # With no pretrained model and no weights file, inference uses the latest training run.
    results = h.infer()
    features = np.asarray(results.__get_all__())
    assert features.shape[0] == N_OBJECTS
    assert np.isfinite(features).any(axis=1).all()

    # The trained model can be exported for use with ParSNIP directly.
    export_path = tmp_path / "exported.pt"
    model.export_parsnip(export_path)
    native = parsnip.load_model(str(export_path), threads=1)
    for name, value in native.state_dict().items():
        torch.testing.assert_close(value.cpu(), model.parsnip.state_dict()[name].cpu())


def test_fine_tune_pretrained(hyrax_instance, hats_catalog):
    h = hyrax_instance
    configure(h, hats_catalog, pretrained="plasticc", groups=("train",))
    h.set_config("train.epochs", 1)

    before = {k: v.clone() for k, v in parsnip.load_model("plasticc", threads=1).state_dict().items()}
    model = h.train()

    assert np.isfinite(model.final_training_metrics["loss"])
    after = model.parsnip.state_dict()
    assert any(not torch.equal(before[k], after[k].cpu()) for k in before if before[k].is_floating_point())
