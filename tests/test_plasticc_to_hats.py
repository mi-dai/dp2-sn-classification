"""examples/plasticc_to_hats.py on small in-memory PLAsTiCC CSVs (no download)."""

import io
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))

import plasticc_to_hats  # noqa: E402

METADATA = """object_id,ra,decl,ddf_bool,hostgal_specz,hostgal_photoz,hostgal_photoz_err,distmod,mwebv,target,true_target,true_submodel,true_z
13,34.45,-5.23,1,0.305,0.319,0.054,41.1,0.019,0,90,1,0.302
14,33.40,-4.33,1,-9.000,0.632,0.018,42.9,0.018,0,995,2,0.610
15,10.00,-40.00,0,-9.000,0.400,0.100,41.0,0.030,0,42,1,0.400
"""

LIGHTCURVES = """object_id,mjd,passband,flux,flux_err,detected_bool
13,59798.32,2,-1.3,1.36,0
13,59799.33,5,25.0,2.00,1
14,59800.10,0,3.0,1.00,0
15,59801.00,1,4.0,1.00,1
99,59802.00,1,4.0,1.00,1
"""


def test_read_metadata_maps_classes_and_missing_specz():
    meta = plasticc_to_hats.read_metadata(io.StringIO(METADATA), "test_ddf")
    assert list(meta["type"]) == ["SNIa", "muLens-String", "SNII"]
    assert np.isnan(meta["hostgal_specz"][1]) and meta["hostgal_specz"][0] == 0.305
    np.testing.assert_allclose(meta["redshift"], [0.302, 0.610, 0.400])
    assert list(meta["dec"]) == [-5.23, -4.33, -40.0]
    assert set(meta["sample"]) == {"test_ddf"}


def test_read_metadata_ddf_only_and_max_objects():
    assert list(plasticc_to_hats.read_metadata(io.StringIO(METADATA), "x", ddf_only=True)["object_id"]) == [13, 14]
    assert len(plasticc_to_hats.read_metadata(io.StringIO(METADATA), "x", max_objects=1)) == 1


def test_read_lightcurves_filters_objects_and_names_bands():
    obs = plasticc_to_hats.read_lightcurves(io.StringIO(LIGHTCURVES), [13, 14], chunksize=2)
    assert list(obs["object_id"]) == [13, 13, 14]
    assert list(obs["band"]) == ["r", "y", "u"]  # passband 5 is y
    assert list(obs["detected"]) == [False, True, False]


def test_nest_keeps_objects_with_observations():
    meta = plasticc_to_hats.read_metadata(io.StringIO(METADATA), "train")
    obs = plasticc_to_hats.read_lightcurves(io.StringIO(LIGHTCURVES), [13, 15])
    frame = plasticc_to_hats.nest(meta, obs)

    assert list(frame["object_id"]) == [13, 15]  # 14 has no observations here
    flat = frame["lightcurve"].nest.to_flat()
    assert list(flat.loc[0, "band"]) == ["r", "y"]
    assert list(flat.loc[1, "band"]) == ["g"]
