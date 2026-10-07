"""Classify a Rubin DIA HATS catalog with ParSNIP or SuperNNova.

    python examples/classify_rubin_dia.py --method {parsnip,snn} \\
        [/global/cfs/cdirs/lsst/groups/TD/SN/EDP2/for_fastdb/subsample_joined.hats] \\
        [--model NAME] [--output PREDICTIONS.parquet] \\
        [--redshift-column redshift] [--mwebv-column mwebv]

Reads forced photometry on difference images (``diaObjectForcedSource``), drops flagged
observations, and runs a pretrained model:

- ``--method parsnip``: ParSNIP features, with fluxes converted from nJy to the PLAsTiCC
  zeropoint. Default model ``plasticc_photoz``, which predicts each object's redshift from
  its light curve; with ``--redshift-column`` (e.g. the true redshift of simulations) the
  default is ``plasticc``, which takes the redshift as input. ParSNIP ships no classifier:
  class probabilities need ``--classifier``, a saved ``parsnip.Classifier`` trained on
  labeled light curves run through the same model (``parsnip_train_classifier.py``). Its
  ``.json`` sets the default ``--model`` and is checked: the model and whether it takes the
  redshift must match how the classifier was trained.
- ``--method snn``: class probabilities from a SuperNNova model: a pretrained Fink ELAsTiCC
  model (fluxes kept in nJy) or a model directory trained with ``snn_train.py`` (fluxes
  rescaled to its training zeropoint). Default ``elasticc_ia`` (SNIa vs other, no redshift); with
  ``--redshift-column`` the default is ``elasticc_broad`` (SN, Fast, Long, Periodic,
  NonPeriodic). Models are downloaded on first use (run once on a NERSC login node).
  ``--fink-exact`` approximates Fink's own input processing, for comparison.

The output has one row per ``diaObjectId`` with ``method``, ``model``, ``redshift_input``
(whether the model was given each object's redshift), ``p_<class>``
probabilities and ``predicted_class`` ("" if not classified); for ParSNIP also its features
(and ``predicted_redshift`` for photo-z models). ``evaluate.py`` reads it.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from hyrax import Hyrax

import hyrax_parsnip
import hyrax_snn

DEFAULT_CATALOG = "/global/cfs/cdirs/lsst/groups/TD/SN/EDP2/for_fastdb/subsample_joined.hats"
RESULTS = Path(__file__).resolve().parent / "results"  # default output location (examples/results/)
PARSNIP_MODELS = ["plasticc", "plasticc_photoz", "ps1"]
PARSNIP_OPTIONS = ["classifier"]
SNN_OPTIONS = ["fink_exact", "detection_snr", "no_time_window", "fix_clipped_min"]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("catalog", nargs="?", default=DEFAULT_CATALOG, help="Rubin DIA HATS catalog")
    parser.add_argument("--method", required=True, choices=["parsnip", "snn"], help="Model family")
    parser.add_argument(
        "--model",
        help=f"Pretrained model. parsnip: {', '.join(PARSNIP_MODELS)} or a .pt path (default plasticc_photoz, or "
        f"plasticc with --redshift-column); snn: {', '.join(sorted(hyrax_snn.PRETRAINED_MODELS))} or a model "
        "directory from snn_train.py (default elasticc_ia, or elasticc_broad with --redshift-column)",
    )
    parser.add_argument(
        "--output", help="Output table (.parquet/.ecsv; default examples/results/<method>_predictions.parquet)"
    )
    parser.add_argument("--redshift-column", help="Per-object redshift column to use as model input")
    parser.add_argument("--mwebv-column", help="Per-object Milky Way E(B-V) column (default: no correction)")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--results-dir", default=str(RESULTS / "hyrax"), help="Hyrax run directory")

    parsnip_group = parser.add_argument_group("parsnip only")
    parsnip_group.add_argument("--classifier", help="Saved parsnip.Classifier to apply (see train_classifier)")

    snn_group = parser.add_argument_group("snn only")
    snn_group.add_argument("--fink-exact", action="store_true", help="Approximate Fink's own input processing")
    snn_group.add_argument("--detection-snr", type=float, help="Keep only observations with flux/fluxerr above this")
    snn_group.add_argument(
        "--no-time-window", action="store_true", help="Use all observations, not -30..+100 d of the peak"
    )
    snn_group.add_argument("--fix-clipped-min", action="store_true", help="Restore the unclipped normalization minimum")

    args = parser.parse_args()
    other = SNN_OPTIONS if args.method == "parsnip" else PARSNIP_OPTIONS
    used = [f"--{name.replace('_', '-')}" for name in other if getattr(args, name) not in (None, False)]
    if used:
        parser.error(f"{', '.join(used)} can't be used with --method {args.method}")

    args.classifier_info = None
    if args.classifier:
        args.classifier_info = check_classifier(parser, args)

    if args.model is None:
        if args.method == "parsnip":
            args.model = "plasticc" if args.redshift_column else "plasticc_photoz"
        else:
            args.model = "elasticc_broad" if args.redshift_column else "elasticc_ia"
    elif args.method == "parsnip" and args.model not in PARSNIP_MODELS and not args.model.endswith(".pt"):
        parser.error(f"--model {args.model!r} is not a ParSNIP model ({', '.join(PARSNIP_MODELS)} or a .pt file)")
    elif (
        args.method == "snn"
        and args.model not in hyrax_snn.PRETRAINED_MODELS
        and not (Path(args.model) / "cli_args.json").exists()
    ):
        parser.error(
            f"--model {args.model!r} is not a SuperNNova model ({', '.join(sorted(hyrax_snn.PRETRAINED_MODELS))}, "
            "or a model directory with cli_args.json)"
        )
    args.output = args.output or str(RESULTS / f"{args.method}_predictions.parquet")
    return args


def check_classifier(parser, args) -> dict | None:
    """Read the .json written next to a classifier by parsnip_train_classifier.py and make sure
    the features will be made the same way (ParSNIP model and redshift input)."""
    info_path = Path(args.classifier).with_suffix(".json")
    if not info_path.exists():
        print(f"Warning: no {info_path.name} next to the classifier, so it can't be checked that it was trained "
              "on features from the same ParSNIP model and redshift input.")
        return None
    info = json.loads(info_path.read_text())
    if args.model is None:
        args.model = info["model"]
    elif args.model != info["model"]:
        parser.error(f"--model {args.model} differs from the classifier's model {info['model']} ({info_path.name})")
    if info["redshift_input"] and not args.redshift_column:
        parser.error(f"The classifier was trained with {info['model']} given the redshift: pass --redshift-column")
    if not info["redshift_input"] and args.redshift_column:
        parser.error(f"The classifier was trained without a redshift input ({info['model']}): drop --redshift-column")
    return info


def new_hyrax(args):
    h = Hyrax()
    h.set_config("general.results_dir", args.results_dir)
    h.set_config("data_loader.batch_size", args.batch_size)
    return h


def catalog_settings(base, args):
    settings = dict(base)
    if args.redshift_column:
        settings["redshift_column"] = args.redshift_column
    if args.mwebv_column:
        settings["mwebv_column"] = args.mwebv_column
    return settings


def takes_redshift(h) -> bool:
    """Whether the configured model takes a per-object redshift as input. Both configure()s
    require a redshift for exactly those models (not for photo-z or no-redshift models)."""
    return bool(h.config["data_set"]["ParsnipHATSDataset"]["require_redshift"])


def add_predicted_class(predictions, class_names):
    """`p_<class>` columns from `class_names` columns, and the most probable class."""
    probabilities = np.stack([np.asarray(predictions[name], dtype=np.float64) for name in class_names], axis=1)
    for name in class_names:
        predictions[f"p_{name}"] = predictions[name]
    valid = np.isfinite(probabilities).all(axis=1)
    best = np.argmax(np.nan_to_num(probabilities, nan=-1), axis=1)
    predictions["predicted_class"] = np.where(valid, np.asarray(class_names)[best], "")


def run_parsnip(args):
    """ParSNIP features, plus class probabilities with --classifier."""
    h = new_hyrax(args)
    hyrax_parsnip.configure(h, args.catalog, pretrained=args.model)
    for key, value in catalog_settings(hyrax_parsnip.RUBIN_DIA_SETTINGS, args).items():
        h.set_config(f"data_set.ParsnipHATSDataset.{key}", value)

    predictions = hyrax_parsnip.load_predictions(h.infer())
    valid = np.isfinite(np.asarray(predictions["s1"]))
    print(f"{len(predictions)} objects with enough good observations; {valid.sum()} processed by ParSNIP")
    if valid.any() and "predicted_redshift" in predictions.colnames:
        redshift = np.asarray(predictions["predicted_redshift"])[valid]
        print(f"predicted redshift: median {np.median(redshift):.3f}, 5-95% {np.percentile(redshift, [5, 95])}")

    if args.classifier:
        info = args.classifier_info
        prior = h.config["model"]["HyraxParsnip"]
        if info and info.get("photoz_prior") and info["photoz_prior"] != [float(prior["photoz"]), float(prior["photoz_error"])]:
            print(f"Warning: the classifier was trained with photo-z prior {info['photoz_prior']}, "
                  f"now {[prior['photoz'], prior['photoz_error']]}.")
        probabilities = hyrax_parsnip.classify(hyrax_parsnip.load_classifier(args.classifier), predictions)
        class_names = probabilities.colnames[1:]
        for name in class_names:
            predictions[name] = probabilities[name]
        add_predicted_class(predictions, class_names)
        predictions.remove_columns(class_names)
    else:
        print("No --classifier: writing ParSNIP features only (no class probabilities).")
        predictions["predicted_class"] = np.full(len(predictions), "", dtype="<U1")
    predictions.remove_column("original_object_id")
    return predictions, takes_redshift(h)


def run_snn(args):
    """Class probabilities from a pretrained SuperNNova model."""
    h = new_hyrax(args)
    # Fink models expect nJy; models trained with snn_train.py the zeropoint of their training fluxes.
    settings = catalog_settings(hyrax_snn.rubin_dia_settings(args.model), args)
    if args.fink_exact:
        settings["flux_scale"] = hyrax_snn.FINK_EXACT_FLUX_SCALE
    hyrax_snn.configure(h, args.catalog, pretrained=args.model, dataset_settings=settings)

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
    print(f"{len(predictions)} objects with enough good observations; {(predictions['predicted_class'] != '').sum()} classified")
    return predictions, takes_redshift(h)


def main():
    args = parse_args()
    predictions, redshift_input = run_parsnip(args) if args.method == "parsnip" else run_snn(args)

    predictions.rename_column("object_id", "diaObjectId")
    # Hyrax keeps ids as strings; Rubin diaObjectIds are int64.
    predictions["diaObjectId"] = np.asarray(predictions["diaObjectId"], dtype=np.int64)
    predictions.add_column(np.full(len(predictions), args.method), name="method", index=1)
    predictions.add_column(np.full(len(predictions), args.model), name="model", index=2)
    # Whether the model was given each object's redshift (e.g. the true redshift of simulations).
    predictions.add_column(np.full(len(predictions), redshift_input), name="redshift_input", index=3)
    if args.method == "parsnip":
        predictions["classifier"] = Path(args.classifier).name if args.classifier else ""
    if redshift_input:
        print(f"Note: {args.model} takes the redshift from '{args.redshift_column}' as input.")

    classified = np.asarray(predictions["predicted_class"])
    if (classified != "").any():
        names, counts = np.unique(classified[classified != ""], return_counts=True)
        print(f"{args.method}/{args.model} predicted classes:", dict(zip(names.tolist(), counts.tolist())))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    predictions.write(args.output, overwrite=True)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
