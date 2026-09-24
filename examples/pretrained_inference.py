"""HATS catalog -> ParSNIP latents (pretrained model, no training) -> class probabilities.

    python examples/pretrained_inference.py /path/to/hats_catalog \\
        [--model plasticc] [--label-column type] [--classifier classifier.pkl]

- With --label-column, a LightGBM classifier is trained on the labels and saved to
  --classifier.
- With only --classifier, a saved classifier is loaded and applied.
"""

import argparse
from pathlib import Path

from hyrax import Hyrax

import hyrax_parsnip


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("catalog", help="HATS catalog with a nested light-curve column")
    parser.add_argument("--model", default="plasticc", help="Built-in ParSNIP model name or .pt path")
    parser.add_argument("--label-column", help="Catalog column with class labels, to train a classifier")
    parser.add_argument("--classifier", help="Where to save (when training) or load the classifier")
    parser.add_argument("--results-dir", default="./results")
    args = parser.parse_args()

    h = Hyrax()
    h.set_config("general.results_dir", args.results_dir)
    hyrax_parsnip.configure(h, args.catalog, pretrained=args.model)

    predictions = hyrax_parsnip.load_predictions(
        h.infer(),
        metadata=hyrax_parsnip.catalog_metadata(args.catalog, [args.label_column]) if args.label_column else None,
    )
    print(predictions["object_id", "s1", "s2", "s3", "color", "luminosity"])

    if args.label_column:
        classifier, _ = hyrax_parsnip.train_classifier(predictions, label_column=args.label_column)
        if args.classifier:
            classifier.write(str(Path(args.classifier).resolve()))
            print(f"Saved classifier to {args.classifier}")
    elif args.classifier:
        classifier = hyrax_parsnip.load_classifier(args.classifier)
    else:
        return

    print(hyrax_parsnip.classify(classifier, predictions))


if __name__ == "__main__":
    main()
