"""hyrax_snn reproduces SuperNNova's own on-the-fly classification with the Fink models.

Needs `supernnova` (install with ``pip install --no-deps supernnova natsort seaborn``;
its pandas < 3 pin conflicts with this environment) and network access to download the
models. Skipped otherwise.
"""

import numpy as np
import pandas as pd
import pytest

supernnova = pytest.importorskip("supernnova")


def patched_classify_lcs():
    """SuperNNova's classify_lcs with two fixes it needs here:

    - pandas 3 returns read-only arrays from `.values`, which compute_delta_time writes into;
    - the Fink models use MWEBV without listing it in `additional_train_var`, so
      SuperNNova 3.0.51's pivot would drop it.
    """
    from supernnova.utils import data_utils
    from supernnova.validation import validate_onthefly

    def compute_delta_time(df):
        df = df.sort_values(["SNID", "MJD"])
        df["delta_time"] = df["MJD"].diff().fillna(0)
        ids = df.SNID.to_numpy() if "SNID" in df.columns else df.index.to_numpy()
        starts = np.where(ids[:-1] != ids[1:])[0] + 1
        delta_time = df["delta_time"].to_numpy(copy=True)
        delta_time[starts] = 0
        df["delta_time"] = delta_time
        return df

    original_get_settings = validate_onthefly.get_settings

    def get_settings(model_file):
        settings = original_get_settings(model_file)
        if "MWEBV" in settings.training_features and not settings.additional_train_var:
            settings.additional_train_var = ["MWEBV"]
        return settings

    data_utils.compute_delta_time = compute_delta_time
    validate_onthefly.get_settings = get_settings
    return validate_onthefly.classify_lcs


@pytest.mark.parametrize("model", ["elasticc_ia", "SN_vs_other", "elasticc_broad"])
def test_matches_supernnova(hyrax_instance, hats_catalog, model):
    import hyrax_snn
    from hyrax_lightcurves import LightCurveHATSDataset

    try:
        model_dir = hyrax_snn.pretrained_model_dir(model)
    except OSError as e:
        pytest.skip(f"Cannot download {model}: {e}")

    h = hyrax_instance
    redshift_error = 0.001
    hyrax_snn.configure(h, hats_catalog, pretrained=model)
    # SuperNNova's on-the-fly classification uses every observation as given.
    h.set_config("model.HyraxSNN.time_window", False)
    h.set_config("model.HyraxSNN.redshift_error", redshift_error)
    dataset = LightCurveHATSDataset(h.config, data_location=hats_catalog)
    predictions = hyrax_snn.load_predictions(h.infer())

    # The same light curves for SuperNNova (the toy catalog has no MWEBV column, so 0).
    bands = np.array(list(hyrax_snn.SNN_BAND_MAP.values()))
    rows = []
    for i in range(len(dataset)):
        lc = dataset.get_lightcurve(i)
        rows.append(
            pd.DataFrame(
                {
                    "SNID": dataset.get_object_id(i),
                    "MJD": lc[:, 0],
                    "FLUXCAL": lc[:, 1],
                    "FLUXCALERR": lc[:, 2],
                    "FLT": bands[lc[:, 3].astype(int)],
                    "MWEBV": 0.0,
                    "HOSTGAL_SPECZ": float(dataset.get_redshift(i)),
                    "HOSTGAL_SPECZ_ERR": redshift_error,
                }
            )
        )
    ids, expected = patched_classify_lcs()(pd.concat(rows, ignore_index=True), str(model_dir / "model.pt"), "cpu")
    expected = pd.DataFrame(expected[:, 0, :], index=np.asarray(ids).astype(str))

    class_names = hyrax_snn.PRETRAINED_MODELS[model][1]
    ours = predictions.to_pandas().set_index("object_id").loc[expected.index, class_names]
    np.testing.assert_allclose(ours.to_numpy(), expected.to_numpy(), atol=1e-4)
