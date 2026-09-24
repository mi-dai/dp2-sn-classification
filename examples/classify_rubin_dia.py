"""ParSNIP features (and optionally classes) for a Rubin DIA HATS catalog, without redshifts.

    python examples/classify_rubin_dia.py \\
        [/global/cfs/cdirs/lsst/groups/TD/SN/EDP2/for_fastdb/subsample_joined.hats] \\
        [--output parsnip_predictions.parquet] [--classifier classifier.pkl]

Reads forced photometry on difference images (``diaObjectForcedSource``), converts
fluxes from nJy to the PLAsTiCC zeropoint, drops flagged observations, and runs the
pretrained ``plasticc_photoz`` ParSNIP model, which predicts each object's redshift from
its light curve (no redshift input).

The output has one row per ``diaObjectId``: the ParSNIP features, ``predicted_redshift``,
and class probabilities if ``--classifier`` is given. ParSNIP does not ship a classifier;
it must be trained on labeled light curves run through the same model
(see ``hyrax_parsnip.train_classifier``).
"""

import argparse

import numpy as np
from hyrax import Hyrax

import hyrax_parsnip

DEFAULT_CATALOG = "/global/cfs/cdirs/lsst/groups/TD/SN/EDP2/for_fastdb/subsample_joined.hats"

# Rubin fluxes are in nJy (AB zeropoint 31.4); the PLAsTiCC models use zeropoint 27.5.
NJY_TO_ZP27_5 = 10 ** (-0.4 * (31.4 - 27.5))

RUBIN_DIA_SETTINGS = {
    "id_column": "diaObjectId",
    "redshift_column": False,
    "lightcurve_column": "diaObjectForcedSource",
    "time_column": "midpointMjdTai",
    "flux_column": "psfDiffFlux",
    "fluxerr_column": "psfDiffFluxErr",
    "band_column": "band",
    "flux_scale": NJY_TO_ZP27_5,
    "flag_columns": [
        "psfDiffFlux_flag",
        "invalidPsfFlag",
        "pixelFlags_saturatedCenter",
        "pixelFlags_crCenter",
        "pixelFlags_nodata",
        "diff_PixelFlags_nodataCenter",
    ],
}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("catalog", nargs="?", default=DEFAULT_CATALOG, help="Rubin DIA HATS catalog")
    parser.add_argument("--output", default="parsnip_predictions.parquet", help="Output table (.parquet/.ecsv)")
    parser.add_argument("--classifier", help="Saved parsnip.Classifier to apply (see train_classifier)")
    parser.add_argument("--model", default="plasticc_photoz", help="ParSNIP model that predicts redshift")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--results-dir", default="./results")
    args = parser.parse_args()

    h = Hyrax()
    h.set_config("general.results_dir", args.results_dir)
    h.set_config("data_loader.batch_size", args.batch_size)
    hyrax_parsnip.configure(h, args.catalog, pretrained=args.model)
    for key, value in RUBIN_DIA_SETTINGS.items():
        h.set_config(f"data_set.ParsnipHATSDataset.{key}", value)

    predictions = hyrax_parsnip.load_predictions(h.infer())
    predictions.rename_column("object_id", "diaObjectId")
    predictions.remove_column("original_object_id")

    valid = np.isfinite(np.asarray(predictions["s1"]))
    redshift = np.asarray(predictions["predicted_redshift"])[valid]
    print(f"{len(predictions)} objects with enough good observations; {valid.sum()} processed by ParSNIP")
    if valid.any():
        print(f"predicted redshift: median {np.median(redshift):.3f}, 5-95% {np.percentile(redshift, [5, 95])}")

    if args.classifier:
        predictions["object_id"] = predictions["diaObjectId"]
        probabilities = hyrax_parsnip.classify(hyrax_parsnip.load_classifier(args.classifier), predictions)
        predictions.remove_column("object_id")
        class_names = probabilities.colnames[1:]
        for name in class_names:
            predictions[f"p_{name}"] = probabilities[name]
        best = np.stack([np.asarray(probabilities[n]) for n in class_names], axis=1)
        predictions["predicted_class"] = np.where(
            valid, np.asarray(class_names)[np.nan_to_num(best, nan=-1).argmax(axis=1)], ""
        )

    # Hyrax keeps ids as strings; Rubin diaObjectIds are int64.
    predictions["diaObjectId"] = np.asarray(predictions["diaObjectId"], dtype=np.int64)
    predictions.write(args.output, overwrite=True)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
