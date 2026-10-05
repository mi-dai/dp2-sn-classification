"""Classify a Rubin DIA HATS catalog with a pretrained SuperNNova model (Fink ELAsTiCC).

    python examples/snn_classify_rubin_dia.py \\
        [/global/cfs/cdirs/lsst/groups/TD/SN/EDP2/for_fastdb/subsample_joined.hats] \\
        [--model elasticc_ia] [--output snn_predictions.parquet] \\
        [--redshift-column redshift] [--mwebv-column mwebv] [--fink-exact]

Reads forced photometry on difference images (``diaObjectForcedSource``, nJy), drops
flagged observations, and runs a pretrained SuperNNova classifier. No training: the
models are downloaded from fink-science on first use (run once on a NERSC login node to
fill the cache).

- Without redshift (default): ``elasticc_ia`` (SNIa vs other) or ``SN_vs_other``.
- With ``--redshift-column`` (e.g. the true redshift of simulations): default
  ``elasticc_broad`` (SN, Fast, Long, Periodic, NonPeriodic) or a ``*_vs_other`` model.

The output has one row per ``diaObjectId``: ``p_<class>`` probabilities and
``predicted_class``. ``--fink-exact`` approximates Fink's own processing (zp-27.5 fluxes,
detections only, no time window) for comparison.
"""

import argparse

import numpy as np
from hyrax import Hyrax

import hyrax_snn

DEFAULT_CATALOG = "/global/cfs/cdirs/lsst/groups/TD/SN/EDP2/for_fastdb/subsample_joined.hats"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("catalog", nargs="?", default=DEFAULT_CATALOG, help="Rubin DIA HATS catalog")
    parser.add_argument("--output", default="snn_predictions.parquet", help="Output table (.parquet/.ecsv)")
    parser.add_argument(
        "--model",
        choices=sorted(hyrax_snn.PRETRAINED_MODELS),
        help="Pretrained model (default: elasticc_ia, or elasticc_broad with --redshift-column)",
    )
    parser.add_argument("--redshift-column", help="Per-object redshift column (needed by models with redshift)")
    parser.add_argument("--mwebv-column", help="Per-object Milky Way E(B-V) column (default: 0)")
    parser.add_argument("--fink-exact", action="store_true", help="Approximate Fink's own input processing")
    parser.add_argument("--detection-snr", type=float, help="Keep only observations with flux/fluxerr above this")
    parser.add_argument("--no-time-window", action="store_true", help="Use all observations, not -30..+100 d of the peak")
    parser.add_argument("--fix-clipped-min", action="store_true", help="Restore the unclipped normalization minimum")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--results-dir", default="./results")
    args = parser.parse_args()

    model = args.model or ("elasticc_broad" if args.redshift_column else "elasticc_ia")

    h = Hyrax()
    h.set_config("general.results_dir", args.results_dir)
    h.set_config("data_loader.batch_size", args.batch_size)

    settings = dict(hyrax_snn.RUBIN_DIA_SETTINGS)
    if args.redshift_column:
        settings["redshift_column"] = args.redshift_column
    if args.mwebv_column:
        settings["mwebv_column"] = args.mwebv_column
    if args.fink_exact:
        settings["flux_scale"] = hyrax_snn.FINK_EXACT_FLUX_SCALE
    hyrax_snn.configure(h, args.catalog, pretrained=model, dataset_settings=settings)

    model_settings = dict(hyrax_snn.FINK_EXACT) if args.fink_exact else {}
    if args.detection_snr is not None:
        model_settings["detection_snr"] = args.detection_snr
    if args.no_time_window:
        model_settings["time_window"] = False
    if args.fix_clipped_min:
        model_settings["fix_clipped_norm_min"] = True
    for key, value in model_settings.items():
        h.set_config(f"model.HyraxSNN.{key}", value)

    predictions = hyrax_snn.load_predictions(h.infer())
    class_names = [c for c in predictions.colnames if c not in ("object_id", "predicted_class")]
    for name in class_names:
        predictions.rename_column(name, f"p_{name}")
    predictions.rename_column("object_id", "diaObjectId")

    valid = predictions["predicted_class"] != ""
    print(f"{model}: {len(predictions)} objects with enough good observations; {valid.sum()} classified")
    if valid.any():
        names, counts = np.unique(np.asarray(predictions["predicted_class"])[valid], return_counts=True)
        print("predicted classes:", dict(zip(names.tolist(), counts.tolist())))

    # Hyrax keeps ids as strings; Rubin diaObjectIds are int64.
    predictions["diaObjectId"] = np.asarray(predictions["diaObjectId"], dtype=np.int64)
    predictions.write(args.output, overwrite=True)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
