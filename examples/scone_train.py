"""Train SCONE on a labeled catalog (default: PLAsTiCC) with Hyrax.

    python examples/scone_train.py [data/plasticc_hats] [--output examples/models/scone_plasticc] \\
        [--classes {ia,dp2,all}] [--redshift-column redshift] [--epochs 400] [--n-jobs 0]

SCONE publishes no trained weights, so it is trained here, on the data ParSNIP's classifier and
the optional SuperNNova training use. The result is a SCONE model directory (``scone_model.json``,
``model.pt``), used with ``classify_rubin_dia.py --method scone --model DIR``.

- Heatmaps: a GP fit to each light curve, 32 wavelengths x 180 days from 30 days before to 150 days
  after its peak (the S/N^2-weighted time of the detections). They are made once, with the fits in
  parallel (``--n-jobs``, default all CPUs), and saved in ``--cache-dir`` for later runs.
- Network and training as SCONE: its CNN (ported to PyTorch), Adam with a constant learning rate of
  5e-5, 400 epochs as in the SCONE papers; class-weighted cross-entropy instead of SCONE's
  downsampling of every class to the smallest.
- ``--classes ia`` (default): SNIa vs non-SNIa; ``dp2``: SNIa/SNII/SNIbc/other; ``all``.
- No redshift by default; ``--redshift-column`` trains a model that also takes the redshift.
- Fluxes are used as given and their zeropoint is recorded (``--flux-zeropoint``, 27.5 for
  PLAsTiCC), so DP2 nJy fluxes are rescaled to match when classifying.
- ``--resume latest`` (or a checkpoint / Hyrax train run directory) continues an interrupted
  run, e.g. a batch job that hit its time limit, with the same command; ``--epochs`` is the total.
- A fraction of the catalog (``--validate-fraction``) is held out: the weights of the epoch with
  the lowest validation loss are kept, and the script reports the accuracy and per-class recall on it.
"""

import argparse
import json
import shutil
import tomllib
from pathlib import Path

import numpy as np
import torch
from hyrax import Hyrax

import hyrax_scone
from hyrax_lightcurves.labels import SCHEMES
from hyrax_lightcurves.training import find_checkpoint

EXAMPLES = Path(__file__).resolve().parent
DEFAULT_OUTPUT = EXAMPLES / "models" / "scone_plasticc"
DEFAULT_CACHE = EXAMPLES.parent / "data" / "scone_heatmaps"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("catalog", nargs="?", default="data/plasticc_hats", help="Labeled HATS catalog")
    parser.add_argument(
        "--output", default=str(DEFAULT_OUTPUT), help="Model directory (default examples/models/scone_plasticc)"
    )
    parser.add_argument("--classes", choices=SCHEMES, default="ia", help="Class scheme (default: SNIa vs non-SNIa)")
    parser.add_argument("--label-column", default="type")
    parser.add_argument("--redshift-column", help="Train a model that takes this per-object redshift")
    parser.add_argument("--redshift-error", type=float, default=0.0, help="Redshift error given with --redshift-column")
    parser.add_argument("--mwebv-column", default="mwebv", help="Milky Way E(B-V) column ('' for none)")
    parser.add_argument("--flux-zeropoint", type=float, default=27.5, help="Zeropoint of the catalog fluxes")
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=32, help="SCONE's default batch size")
    parser.add_argument("--validate-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=0, help="Processes for the GP fits (0: all CPUs)")
    parser.add_argument(
        "--cache-dir", default=str(DEFAULT_CACHE), help="Heatmap cache (default data/scone_heatmaps)"
    )
    parser.add_argument(
        "--results-dir", default=str(EXAMPLES / "results" / "hyrax"), help="Hyrax run directory"
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing model directory")
    parser.add_argument(
        "--resume",
        metavar="{CHECKPOINT,RUN_DIR,latest}",
        help="Continue an interrupted run from a Hyrax checkpoint ('latest': newest run in --results-dir)",
    )
    args = parser.parse_args()

    output = Path(args.output).resolve()
    checkpoint = None
    if args.resume:
        if args.overwrite:
            parser.error("--resume continues training in the existing model directory; drop --overwrite")
        if not (output / "scone_model.json").exists():
            parser.error(f"--resume needs the model directory of the interrupted run; no {output}/scone_model.json")
        try:
            checkpoint = find_checkpoint(args.resume, args.results_dir)
        except FileNotFoundError as error:
            parser.error(str(error))
        check_resume(parser, checkpoint, output, args)
    elif output.exists():
        if not args.overwrite:
            parser.error(f"{output} exists; pass --overwrite to replace it (or --resume to continue training)")
        shutil.rmtree(output)
    with_z = bool(args.redshift_column)
    dataset_settings = {
        **hyrax_scone.RUBIN_DIA_SETTINGS,  # only for the band map; the columns are reset below
        "id_column": "object_id",
        "lightcurve_column": "lightcurve",
        "time_column": "mjd",
        "flux_column": "flux",
        "fluxerr_column": "fluxerr",
        "band_column": "band",
        "flux_scale": 1.0,
        "flag_columns": [],
        "redshift_column": args.redshift_column or False,
        "mwebv_column": args.mwebv_column or False,
        "label_column": args.label_column,
        "label_scheme": args.classes,
        "scone": {"cache_dir": str(Path(args.cache_dir).resolve()), "n_jobs": args.n_jobs},
    }

    h = Hyrax()
    h.set_config("general.results_dir", args.results_dir)
    h.set_config("data_loader.batch_size", args.batch_size)
    h.set_config("train.epochs", args.epochs)
    h.set_config("split.train", 1 - args.validate_fraction)
    h.set_config("split.validate", args.validate_fraction)
    h.set_config("split.rng_seed", args.seed)

    # 1. Heatmaps of the catalog (made once, then cached), and a new model directory with the classes
    # and class weights (kept as is when resuming).
    dataset = hyrax_scone.training_dataset(h, args.catalog, dataset_settings, with_z=with_z)
    if checkpoint:
        info = hyrax_scone.load_model_info(output)
        h.set_config("train.resume", str(checkpoint))
    else:
        info = hyrax_scone.new_model_dir(
            output, dataset, with_z=with_z, redshift_error=args.redshift_error, flux_zeropoint=args.flux_zeropoint
        )
        n_heatmaps = int(dataset.has_heatmap().sum())
        print(f"New model in {output}: classes {info['class_names']}, {n_heatmaps} of {len(dataset)} objects "
              f"with a heatmap; counts {info['training_counts']}")

    # 2. Train on the train split, validate on the rest.
    hyrax_scone.configure(
        h, args.catalog, model_dir=output, groups=("train", "validate"), dataset_settings=dataset_settings
    )
    model = h.train()
    model.export_scone(output)  # the weights of the epoch with the lowest validation loss
    if model.best_epoch:
        loss, acc = model.validation_history[model.best_epoch - 1]
        n_epochs = len(model.validation_history)
        print(f"Best validation epoch {model.best_epoch}/{n_epochs}: loss {loss:.4f}, accuracy {acc:.3f}")

    # 3. Score the held-out (validation) objects with the trained model.
    run_dir = max(Path(args.results_dir).glob("*-train-*"), key=lambda p: p.stat().st_mtime)
    validate_ids = {dataset.get_object_id(int(i)) for i in np.load(run_dir / "validate_split.npz")["indexes"]}
    h_infer = Hyrax()
    h_infer.set_config("general.results_dir", args.results_dir)
    h_infer.set_config("data_loader.batch_size", 256)
    hyrax_scone.configure(h_infer, args.catalog, model_dir=output, dataset_settings=dataset_settings)
    labels = hyrax_scone.load_predictions(h_infer.infer()).to_pandas()
    labels = labels[labels["object_id"].isin(validate_ids) & (labels["predicted_class"] != "")]
    truth = {dataset.get_object_id(i): info["class_names"][dataset.get_label_index(i)] for i in range(len(dataset))}
    labels["true_class"] = labels["object_id"].map(truth)
    accuracy = float((labels["predicted_class"] == labels["true_class"]).mean())
    recall = {c: float((g["predicted_class"] == c).mean()) for c, g in labels.groupby("true_class")}
    rounded = {k: round(v, 3) for k, v in recall.items()}
    print(f"Validation ({len(labels)} objects): accuracy {accuracy:.3f}; recall {json.dumps(rounded)}")

    info_path = output / "scone_model.json"
    info = json.loads(info_path.read_text())
    info["training"] = {
        "catalog": str(Path(args.catalog).resolve()),
        "class_scheme": args.classes,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": float(h.config["model"]["HyraxSCONE"]["learning_rate"]),
        "validate_fraction": args.validate_fraction,
        "best_epoch": model.best_epoch,
        # null: an epoch whose validation was lost when resuming (Hyrax checkpoints before validating)
        "validation_history": [
            {"loss": loss, "accuracy": acc} if np.isfinite(loss) else None for loss, acc in model.validation_history
        ],
        "validation_objects": len(labels),
        "validation_accuracy": accuracy,
        "validation_recall": recall,
        "hyrax_run": str(run_dir),
        "resumed_from": str(checkpoint) if checkpoint else None,
    }
    info_path.write_text(json.dumps(info, indent=2))
    print(f"Wrote {output}/model.pt; classify with: classify_rubin_dia.py --method scone --model {output}")


def check_resume(parser, checkpoint, output, args):
    """Make sure `checkpoint` continues this run: same model directory, split and batches, and
    epochs left to train (ignite restarts from scratch once the epochs are done)."""
    run_config = tomllib.loads((checkpoint.parent / "runtime_config.toml").read_text())
    expected = {
        "model directory": (run_config["model"]["HyraxSCONE"]["model_dir"], str(output)),
        "--validate-fraction": (run_config["split"].get("validate"), args.validate_fraction),
        "--seed": (run_config["split"].get("rng_seed"), args.seed),
        "--batch-size": (run_config["data_loader"]["batch_size"], args.batch_size),
    }
    for name, (before, now) in expected.items():
        if before != now:
            parser.error(f"--resume: {checkpoint.parent.name} used {name} {before}, now {now}")
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)["trainer"]
    epoch = state["iteration"] // state["epoch_length"]
    if epoch >= args.epochs:
        parser.error(f"--resume: {checkpoint.name} is at epoch {epoch}; pass --epochs > {epoch} to train further")
    print(f"Resuming from {checkpoint} (epoch {epoch} of {args.epochs})")


if __name__ == "__main__":
    main()
