"""Download what the jobs need, so they can run on compute nodes without internet access.

    .venv/bin/python nersc/prefetch.py [--skip-plasticc] [--skip-simulation]

Run on a login node (nersc/setup_env.sh calls it). Each step reports and continues on failure:
- the pretrained SuperNNova models (fink-science, ~2 MB) into ~/.cache/hyrax_snn
- the PLAsTiCC files for the classifier training set (~330 MB) into data/plasticc_raw
- the simulation's LSST passbands (lightcurvelynx) and sncosmo templates (SALT3, Vincenzi et al. 2019),
  and a check that the SFD dust map is installed
"""

import argparse
import sys
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "examples"))


def step(name, func):
    print(f"\n== {name}")
    try:
        func()
        print("   ok")
    except Exception:  # report and keep going: one failed download shouldn't block the others
        traceback.print_exc(limit=2)
        print(f"   FAILED: {name}")


def snn_models():
    import hyrax_snn

    for name in hyrax_snn.PRETRAINED_MODELS:
        print("  ", name, hyrax_snn.pretrained_model_dir(name))


def plasticc():
    import plasticc_to_hats

    plasticc_to_hats.download(plasticc_to_hats.TRAIN_FILES + plasticc_to_hats.DDF_FILES, REPO / "data" / "plasticc_raw")


def simulation_assets():
    import sncosmo
    import simulate_dp2  # sets LIGHTCURVELYNX_DATA_DIR to examples/data/lightcurvelynx
    from lightcurvelynx.astro_utils.passbands import PassbandGroup

    PassbandGroup.from_preset("LSST", filters=simulate_dp2.BANDS)
    sncosmo.get_source("salt3")
    for settings in simulate_dp2.CLASSES.values():  # core-collapse classes list their template types
        for name in simulate_dp2.v19_templates(settings.get("types", ())):
            sncosmo.get_source(name)

    from dustmaps.sfd import SFDQuery

    try:
        SFDQuery()
    except Exception as e:
        raise RuntimeError(
            "SFD dust map not found. Fetch it once with: "
            ".venv/bin/python -c 'import dustmaps.sfd; dustmaps.sfd.fetch()' (see the dustmaps docs for data_dir)"
        ) from e


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--skip-plasticc", action="store_true")
    parser.add_argument("--skip-simulation", action="store_true")
    args = parser.parse_args()

    step("SuperNNova pretrained models", snn_models)
    if not args.skip_plasticc:
        step("PLAsTiCC files (Zenodo)", plasticc)
    if not args.skip_simulation:
        step("Simulation passbands, templates and dust map", simulation_assets)


if __name__ == "__main__":
    main()
