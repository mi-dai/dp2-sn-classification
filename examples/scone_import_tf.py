"""Convert a model trained with SCONE itself (a TensorFlow SavedModel) into a hyrax_scone model directory.

    python examples/scone_import_tf.py data/scone_model_example/trained_model \\
        --output examples/models/scone_example [--classes SNIa SNII ...] [--overwrite]

The SavedModel directory is what SCONE writes as ``trained_model/``: ``saved_model.pb``, ``variables/``
and usually ``filter_mean_wavelength.dat`` (the band wavelengths its heatmaps were made with). The
weights are read without TensorFlow and mapped onto hyrax_scone's PyTorch port of the network.

The result is used like a model trained with ``scone_train.py``:
``classify_rubin_dia.py --method scone --model DIR``. Its heatmaps are made as upstream SCONE made them
(Milky Way extinction added to the flux, divided by the maximum flux, no cap on the uncertainty, the
model's band wavelengths), except for the time window: SCONE centres it on the simulation's true peak,
hyrax_scone on the S/N^2-weighted time of the detections, as for real data.
"""

import argparse
import shutil
from pathlib import Path

from hyrax_scone.convert import import_saved_model


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("saved_model", help="SCONE's trained_model/ directory (saved_model.pb, variables/)")
    parser.add_argument("--output", required=True, help="New model directory, e.g. examples/models/scone_<name>")
    parser.add_argument("--classes", nargs="+", help="Class names of a multi-class model, in SCONE's order")
    parser.add_argument("--redshift-error", type=float, default=0.0, help="Redshift error for models with redshift")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing model directory")
    args = parser.parse_args()

    if not (Path(args.saved_model) / "variables").is_dir():
        parser.error(f"{args.saved_model} has no variables/ directory: the SavedModel's weights are missing")
    output = Path(args.output)
    if output.exists():
        if not args.overwrite:
            parser.error(f"{output} exists; pass --overwrite to replace it")
        shutil.rmtree(output)

    try:
        info = import_saved_model(args.saved_model, output, class_names=args.classes, redshift_error=args.redshift_error)
    except ValueError as error:
        parser.error(str(error))
    heatmaps = info["heatmap_settings"]
    print(f"Converted {args.saved_model} -> {output}")
    print(f"  classes {info['class_names']}; redshift input: {info['with_z']}")
    print(f"  heatmaps {heatmaps['n_wavelengths']} x {heatmaps['n_times']}; band wavelengths "
          f"{heatmaps['band_wavelengths']} ({info['band_wavelengths_from']})")
    print(f"  TensorFlow optimizer state: {info['tensorflow_optimizer']}")
    redshift = " --redshift-column redshift" if info["with_z"] else ""
    print(f"Classify with: classify_rubin_dia.py --method scone --model {output}{redshift}")


if __name__ == "__main__":
    main()
