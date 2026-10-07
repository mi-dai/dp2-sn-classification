"""Train ParSNIP's LightGBM classifier on a labeled catalog (default: PLAsTiCC).

    python examples/parsnip_train_classifier.py [data/plasticc_hats] \\
        [--output examples/models/plasticc_classifier.pkl] [--classes {ia,dp2,all}] [--redshift-column redshift]

Runs a pretrained ParSNIP model over a labeled HATS catalog (e.g. from
``plasticc_to_hats.py``), trains ``parsnip.Classifier`` on the features with K-folding,
prints the K-fold (out-of-sample) performance, and writes (by default in ``examples/models/``):

- ``<output>.pkl``: the classifier, for ``classify_rubin_dia.py --method parsnip --classifier``
- ``<output>.json``: how its features were made (ParSNIP model, whether the model was given
  the redshift, classes, ...). ``classify_rubin_dia.py`` checks it, since the classifier only
  works on features made the same way.
- ``<output>_out_of_sample.parquet``: K-fold out-of-sample probabilities with the labels

Default model ``plasticc_photoz`` with the same constant weak photo-z prior used when
classifying DP2, so no external redshift is used; ``--redshift-column`` trains the
``plasticc`` (redshift-input) variant instead.

Classes: ``--classes ia`` (default) is binary, SNIa vs non-SNIa (every other type, including
SNIa-91bg and SNIax); ``dp2`` predicts SNIa, SNII, SNIbc and "other"; ``all`` keeps every
catalog type.
"""

import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from hyrax import Hyrax

import hyrax_parsnip
from hyrax_lightcurves.labels import SCHEMES, class_labels

DEFAULT_OUTPUT = Path(__file__).resolve().parent / "models" / "plasticc_classifier.pkl"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("catalog", nargs="?", default="data/plasticc_hats", help="Labeled HATS catalog")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Classifier file (.pkl; default examples/models/plasticc_classifier.pkl)")
    parser.add_argument("--model", help="ParSNIP model (default plasticc_photoz, or plasticc with --redshift-column)")
    parser.add_argument("--redshift-column", help="Per-object redshift column to give the model (e.g. redshift)")
    parser.add_argument("--mwebv-column", default="mwebv", help="Milky Way E(B-V) column ('' for none)")
    parser.add_argument("--label-column", default="type", help="Class label column")
    parser.add_argument("--classes", choices=SCHEMES, default="ia", help="Class scheme (default: SNIa vs non-SNIa)")
    parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument(
        "--min-child-weight", type=float, default=10.0,
        help="LightGBM min_child_weight (ParSNIP's 1000 suits millions of augmented rows)",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--results-dir", default=str(DEFAULT_OUTPUT.parents[1] / "results" / "hyrax"), help="Hyrax run directory")
    args = parser.parse_args()

    model = args.model or ("plasticc" if args.redshift_column else "plasticc_photoz")
    output = Path(args.output).resolve()
    if output.suffix != ".pkl":
        parser.error("--output must end in .pkl")
    output.parent.mkdir(parents=True, exist_ok=True)

    h = Hyrax()
    h.set_config("general.results_dir", args.results_dir)
    h.set_config("data_loader.batch_size", args.batch_size)
    hyrax_parsnip.configure(h, args.catalog, pretrained=model)
    h.set_config("data_set.LightCurveHATSDataset.redshift_column", args.redshift_column or False)
    h.set_config("data_set.LightCurveHATSDataset.mwebv_column", args.mwebv_column or False)
    redshift_input = bool(h.config["data_set"]["LightCurveHATSDataset"]["require_redshift"])
    if redshift_input and not args.redshift_column:
        parser.error(f"{model} takes a redshift input: pass --redshift-column")
    if args.redshift_column and not redshift_input:
        parser.error(f"{model} predicts the redshift and ignores --redshift-column; use plasticc or drop it")

    metadata_columns = [args.label_column] + (["sample"] if "sample" in _catalog_columns(args.catalog) else [])
    predictions = hyrax_parsnip.load_predictions(
        h.infer(), metadata=hyrax_parsnip.catalog_metadata(args.catalog, metadata_columns)
    )
    predictions["label"] = class_labels(predictions[args.label_column], args.classes)

    classifier, out_of_sample = hyrax_parsnip.train_classifier(
        predictions, label_column="label", num_folds=args.num_folds, min_child_weight=args.min_child_weight
    )
    classes = [str(c) for c in classifier.class_names]
    classifier.write(str(output))

    # K-fold (out-of-sample) performance on the training catalog.
    oos = out_of_sample.to_pandas()
    labels = predictions.to_pandas()[["object_id", "label"] + metadata_columns[1:]]
    oos = oos.merge(labels, on="object_id")
    oos["predicted"] = np.asarray(classes)[oos[classes].to_numpy().argmax(axis=1)]
    accuracy = float((oos["predicted"] == oos["label"]).mean())
    oos.to_parquet(output.with_name(output.stem + "_out_of_sample.parquet"))

    print(f"\n{model}: {len(oos)} objects with ParSNIP features; K-fold accuracy {accuracy:.3f}")
    confusion = pd.crosstab(oos["label"], oos["predicted"], normalize="index").reindex(index=classes, columns=classes)
    print("K-fold confusion (rows: true, normalized):\n" + confusion.round(3).to_string())
    if "sample" in oos:
        for sample, group in oos.groupby("sample"):
            print(f"  {sample}: accuracy {(group['predicted'] == group['label']).mean():.3f} ({len(group)})")

    settings = h.config["model"]["HyraxParsnip"]
    info = {
        "model": model,
        "redshift_input": redshift_input,
        "redshift_column": args.redshift_column,
        "photoz_prior": None if redshift_input else [float(settings["photoz"]), float(settings["photoz_error"])],
        "classes": classes,
        "class_scheme": args.classes,
        "catalog": str(Path(args.catalog).resolve()),
        "mwebv_column": args.mwebv_column or None,
        "flux_scale": float(h.config["data_set"]["LightCurveHATSDataset"]["flux_scale"]),
        "counts": oos["label"].value_counts().to_dict(),
        "samples": oos["sample"].value_counts().to_dict() if "sample" in oos else None,
        "num_folds": args.num_folds,
        "min_child_weight": args.min_child_weight,
        "kfold_accuracy": accuracy,
        "created": date.today().isoformat(),
    }
    output.with_suffix(".json").write_text(json.dumps(info, indent=2))
    print(f"Wrote {output} and {output.with_suffix('.json').name}")


def _catalog_columns(catalog_path) -> list[str]:
    import lsdb

    return list(lsdb.open_catalog(str(catalog_path)).columns)


if __name__ == "__main__":
    main()
