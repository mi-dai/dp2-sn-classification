"""Train (or fine-tune) ParSNIP with Hyrax on a HATS catalog, then run inference.

    python examples/parsnip_train_then_infer.py /path/to/hats_catalog \\
        [--from-pretrained plasticc] [--epochs 50] [--export model.pt]

Without --from-pretrained, a new ParSNIP model is trained from scratch using the
bands in data_set.LightCurveHATSDataset.band_map (``hyrax_parsnip.PARSNIP_BAND_MAP``).

A Hyrax epoch is one pass over the catalog. ParSNIP's own `fit` treats ~25,000
augmented light curves as an epoch, so small catalogs need proportionally more
Hyrax epochs.
"""

import argparse

from hyrax import Hyrax

import hyrax_parsnip


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("catalog", help="HATS catalog with a nested light-curve column")
    parser.add_argument("--from-pretrained", default=False, help="ParSNIP model to fine-tune")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--export", help="Also save the trained model in native ParSNIP .pt format")
    parser.add_argument("--results-dir", default="./results")
    args = parser.parse_args()

    h = Hyrax()
    h.set_config("general.results_dir", args.results_dir)
    h.set_config("data_loader.batch_size", args.batch_size)
    h.set_config("train.epochs", args.epochs)
    # model_weights_file=False: infer with the weights from the training run below.
    hyrax_parsnip.configure(
        h,
        args.catalog,
        pretrained=args.from_pretrained,
        groups=("train", "infer"),
        model_weights_file=False,
    )

    model = h.train()
    if args.export:
        model.export_parsnip(args.export)
        print(f"Exported ParSNIP model to {args.export}")

    predictions = hyrax_parsnip.load_predictions(h.infer())
    print(predictions["object_id", "s1", "s2", "s3", "color", "luminosity"])


if __name__ == "__main__":
    main()
