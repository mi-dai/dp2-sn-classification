"""Download PLAsTiCC from Zenodo and write it as a labeled HATS catalog.

    python examples/plasticc_to_hats.py [data/plasticc_hats] [--ddf] [--max-objects N]

PLAsTiCC (Kessler et al. 2019; https://zenodo.org/records/2539456, CC-BY-4.0) is the
simulation ParSNIP's ``plasticc`` models were trained on. This writes:

- the official training set (7,848 objects; biased toward bright, low-redshift,
  spectroscopically confirmed objects), and with ``--ddf``
- the Deep Drilling Field objects of the unblinded test set (a deeper, photometric sample;
  all in test light-curve chunk 01), with their true classes.

Downloads ~22 MB, or ~330 MB with ``--ddf``, into ``--raw-dir`` (existing files are reused).
Fluxes are kept as distributed (the values ParSNIP was trained on). The catalog uses the
default ``[data_set.LightCurveHATSDataset]`` columns, so the dataset reads it unchanged:

- per object: ``object_id``, ``ra``, ``dec``, ``type`` (class name), ``redshift`` (true),
  ``hostgal_specz`` (NaN if none), ``hostgal_photoz``, ``hostgal_photoz_err``, ``mwebv``,
  ``ddf``, ``sample`` ("train" or "test_ddf")
- nested ``lightcurve``: ``mjd``, ``band`` (u g r i z y), ``flux``, ``fluxerr``, ``detected``
"""

import argparse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ZENODO_URL = "https://zenodo.org/records/2539456/files/{}?download=1"
TRAIN_FILES = ("plasticc_train_metadata.csv.gz", "plasticc_train_lightcurves.csv.gz")
DDF_FILES = ("plasticc_test_metadata.csv.gz", "plasticc_test_lightcurves_01.csv.gz")

BANDS = np.array(["u", "g", "r", "i", "z", "y"])  # PLAsTiCC passband 0..5
# PLAsTiCC class codes (true_target) -> names, as in lcdata's PLAsTiCC loader.
CLASS_NAMES = {
    6: "muLens-Single",
    15: "TDE",
    16: "EB",
    42: "SNII",
    52: "SNIax",
    53: "Mira",
    62: "SNIbc",
    64: "KN",
    65: "M-dwarf",
    67: "SNIa-91bg",
    88: "AGN",
    90: "SNIa",
    92: "RRL",
    95: "SLSN-I",
    991: "muLens-Binary",
    992: "ILOT",
    993: "CaRT",
    994: "PISN",
    995: "muLens-String",
}


def download(filenames, raw_dir: Path):
    raw_dir.mkdir(parents=True, exist_ok=True)
    for name in filenames:
        path = raw_dir / name
        if path.exists():
            continue
        print(f"Downloading {name} ...")
        tmp = path.with_suffix(path.suffix + ".part")
        urllib.request.urlretrieve(ZENODO_URL.format(name), tmp)
        tmp.rename(path)


def read_metadata(source, sample: str, ddf_only: bool = False, max_objects: int | None = None) -> pd.DataFrame:
    """Per-object table from a PLAsTiCC metadata CSV (path or file-like)."""
    meta = pd.read_csv(source)
    if ddf_only:
        meta = meta[meta["ddf_bool"] == 1]
    if max_objects:
        meta = meta.iloc[:max_objects]
    return pd.DataFrame(
        {
            "object_id": meta["object_id"].astype(np.int64).to_numpy(),
            "ra": meta["ra"].to_numpy(np.float64),
            "dec": meta["decl"].to_numpy(np.float64),
            "type": meta["true_target"].map(CLASS_NAMES).fillna("Unknown").to_numpy(str),
            "redshift": meta["true_z"].to_numpy(np.float64),
            "hostgal_specz": meta["hostgal_specz"].where(meta["hostgal_specz"] >= 0).to_numpy(np.float64),
            "hostgal_photoz": meta["hostgal_photoz"].to_numpy(np.float64),
            "hostgal_photoz_err": meta["hostgal_photoz_err"].to_numpy(np.float64),
            "mwebv": meta["mwebv"].to_numpy(np.float64),
            "ddf": meta["ddf_bool"].astype(bool).to_numpy(),
            "sample": sample,
        }
    )


def read_lightcurves(source, object_ids, chunksize: int = 2_000_000) -> pd.DataFrame:
    """Observations of `object_ids` from a PLAsTiCC light-curve CSV, read in chunks."""
    wanted = pd.Index(np.asarray(object_ids, dtype=np.int64))
    parts = []
    for chunk in pd.read_csv(source, chunksize=chunksize):
        chunk = chunk[chunk["object_id"].isin(wanted)]
        if len(chunk):
            parts.append(chunk)
    obs = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["object_id"])
    return pd.DataFrame(
        {
            "object_id": obs["object_id"].astype(np.int64).to_numpy(),
            "mjd": obs["mjd"].to_numpy(np.float64),
            "band": BANDS[obs["passband"].to_numpy(int)] if len(obs) else np.array([], dtype=str),
            "flux": obs["flux"].to_numpy(np.float32),
            "fluxerr": obs["flux_err"].to_numpy(np.float32),
            "detected": obs["detected_bool"].astype(bool).to_numpy() if len(obs) else np.array([], dtype=bool),
        }
    )


def nest(meta: pd.DataFrame, observations: pd.DataFrame):
    """NestedFrame with one row per object and a nested `lightcurve` column (objects without observations dropped)."""
    import nested_pandas as npd

    meta = meta[meta["object_id"].isin(observations["object_id"])].reset_index(drop=True)
    row = pd.Series(np.arange(len(meta)), index=meta["object_id"])
    observations = observations[observations["object_id"].isin(row.index)]
    observations = observations.set_index(row.loc[observations["object_id"]].to_numpy()).drop(columns="object_id")
    return npd.NestedFrame(meta).join_nested(observations.sort_index(kind="stable"), "lightcurve")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("output", nargs="?", default="data/plasticc_hats", help="Output HATS catalog")
    parser.add_argument("--raw-dir", default="data/plasticc_raw", help="Where the Zenodo files are downloaded")
    parser.add_argument("--ddf", action="store_true", help="Add the DDF objects of the unblinded test set")
    parser.add_argument("--max-objects", type=int, help="Keep only the first N objects of each sample (quick checks)")
    args = parser.parse_args()

    import lsdb

    raw = Path(args.raw_dir)
    download(TRAIN_FILES + (DDF_FILES if args.ddf else ()), raw)

    samples = [("train", TRAIN_FILES, False)] + ([("test_ddf", DDF_FILES, True)] if args.ddf else [])
    frames = []
    for name, (meta_file, lc_file), ddf_only in samples:
        print(f"Reading {name} ...")
        meta = read_metadata(raw / meta_file, name, ddf_only=ddf_only, max_objects=args.max_objects)
        frames.append((meta, read_lightcurves(raw / lc_file, meta["object_id"])))
    meta = pd.concat([m for m, _ in frames], ignore_index=True)
    observations = pd.concat([o for _, o in frames], ignore_index=True)
    frame = nest(meta, observations)

    lsdb.from_dataframe(frame, ra_column="ra", dec_column="dec", catalog_name="plasticc").write_catalog(
        args.output, overwrite=True
    )
    counts = frame.groupby("sample").size().to_dict()
    print(f"Wrote {len(frame):,} objects {counts} to {args.output}")
    print("types:", frame["type"].value_counts().to_dict())


if __name__ == "__main__":
    main()
