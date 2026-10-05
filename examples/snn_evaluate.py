"""Evaluate pretrained SuperNNova predictions on a simulated DP2 catalog.

    python examples/snn_evaluate.py CATALOG PREDICTIONS [--output-dir DIR] [--threshold 0.5]

CATALOG is a HATS catalog written by ``simulate_dp2.py`` (truth columns ``type`` and
``redshift``). PREDICTIONS is the table written by ``snn_classify_rubin_dia.py`` for
that catalog. The Fink models have fixed classes; each model's first class (its target:
SNIa, SN, Fast, ...) is scored against the truth. The simulation contains only SN Ia,
II and Ib/c, so for ``elasticc_ia`` this is SN Ia vs other SNe, for ``SN_vs_other`` and
``elasticc_broad`` it is the recall of the SN class, and for the other ``*_vs_other``
models every positive is a false positive. Saves, in DIR:

- ``probability.png``: P(target) per true type
- ``predicted.png``: fraction of each true type assigned to each class
- ``recall.png``: fraction classified as the target vs true redshift and detections
- ``roc.png``: ROC curve and efficiency/purity vs threshold (when the truth has both
  positives and negatives, e.g. elasticc_ia)
"""

import argparse
from pathlib import Path

import lsdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
SEQUENTIAL = "Blues"
TRUTH_COLUMNS = ["diaObjectId", "type", "redshift", "nDiaSources"]
SN_TYPES = {"SNIa", "SNII", "SNIbc"}

plt.rcParams.update(
    {
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#e5e5e5",
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "legend.frameon": False,
        "savefig.dpi": 150,
    }
)


def is_target(types: pd.Series, target: str) -> np.ndarray:
    """True type -> whether it belongs to the model's target class."""
    if target == "SNIa":
        return (types == "SNIa").to_numpy()
    if target == "SN":
        return types.isin(SN_TYPES).to_numpy()
    return np.zeros(len(types), dtype=bool)  # Fast, Long, Periodic, NonPeriodic: no SN belongs


def load(catalog_path, predictions_path):
    truth = lsdb.open_catalog(str(catalog_path), columns=TRUTH_COLUMNS).compute().reset_index(drop=True)
    predictions = pd.read_parquet(predictions_path)
    classes = [c[2:] for c in predictions.columns if c.startswith("p_")]
    table = predictions.merge(truth, on="diaObjectId", how="left")
    if table["type"].isna().any():
        raise ValueError("Some predictions have no truth: is PREDICTIONS from this CATALOG?")
    table = table[table["predicted_class"] != ""].reset_index(drop=True)
    return table, classes


def plot_probability(table, target, types, path):
    fig, ax = plt.subplots(figsize=(5, 3.4))
    bins = np.linspace(0, 1, 26)
    for color, label in zip(PALETTE, types):
        sel = table["type"] == label
        ax.hist(table.loc[sel, f"p_{target}"], bins=bins, histtype="step", lw=2, color=color, label=f"{label} ({sel.sum()})")
    ax.set(xlabel=f"P({target})", ylabel="objects", title=f"P({target}) by true type")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_predicted(table, classes, types, path):
    counts = pd.crosstab(table["type"], table["predicted_class"]).reindex(index=types, columns=classes, fill_value=0)
    frac = counts.to_numpy() / np.maximum(counts.to_numpy().sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(1.1 * len(classes) + 2.2, 0.6 * len(types) + 1.8))
    ax.imshow(frac, cmap=SEQUENTIAL, vmin=0, vmax=1, aspect="auto")
    ax.grid(False)
    for i in range(len(types)):
        for j in range(len(classes)):
            ax.text(
                j, i, f"{frac[i, j]:.2f}\n({counts.iat[i, j]})", ha="center", va="center", fontsize=8,
                color="white" if frac[i, j] > 0.55 else "#1a1a1a",
            )
    ax.set_xticks(range(len(classes)), classes)
    ax.set_yticks(range(len(types)), types)
    ax.set(xlabel="predicted class", ylabel="true type", title="Predicted class per true type (rows normalized)")
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def binned_fraction(x, flag, bins, min_count=10):
    idx = np.digitize(x, bins)
    keep = [i for i in range(1, len(bins)) if (idx == i).sum() >= min_count]
    centers = np.array([(bins[i - 1] + bins[i]) / 2 for i in keep])
    frac = np.array([flag[idx == i].mean() for i in keep])
    n = np.array([(idx == i).sum() for i in keep])
    return centers, frac, np.sqrt(frac * (1 - frac) / n)


def plot_recall(table, target, types, threshold, path):
    selected = (table[f"p_{target}"] >= threshold).to_numpy()
    z_bins = np.linspace(0, table["redshift"].max(), 9)
    det_bins = np.unique(np.quantile(table["nDiaSources"], np.linspace(0, 1, 9)))
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for ax, column, bins in [(axes[0], "redshift", z_bins), (axes[1], "nDiaSources", det_bins)]:
        for color, label in zip(PALETTE, types):
            sel = (table["type"] == label).to_numpy()
            x, frac, err = binned_fraction(table[column].to_numpy()[sel], selected[sel], bins)
            ax.errorbar(x, frac, err, color=color, lw=2, marker="o", ms=5, capsize=0, label=label)
    axes[0].set(xlabel="true redshift", ylabel=f"fraction with P({target}) ≥ {threshold}", ylim=(0, 1.02))
    axes[1].set(xlabel="detections (nDiaSources)")
    axes[0].legend(loc="upper right")
    fig.suptitle(f"Classified as {target}, per true type", fontsize=10)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_roc(table, target, truth, path):
    from sklearn.metrics import roc_auc_score, roc_curve

    score = table[f"p_{target}"].to_numpy()
    fpr, tpr, _ = roc_curve(truth, score)
    auc = roc_auc_score(truth, score)
    thresholds = np.linspace(0, 1, 101)
    efficiency = np.array([(score[truth] >= t).mean() for t in thresholds])
    with np.errstate(invalid="ignore"):
        purity = np.array([truth[score >= t].mean() if (score >= t).any() else np.nan for t in thresholds])

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    axes[0].plot(fpr, tpr, color=PALETTE[0], lw=2)
    axes[0].plot([0, 1], [0, 1], color="#999999", ls=":", lw=1)
    axes[0].set(xlabel="false positive rate", ylabel="true positive rate", title=f"{target} vs rest: AUC {auc:.3f}")
    axes[1].plot(thresholds, efficiency, color=PALETTE[0], lw=2, label="efficiency")
    axes[1].plot(thresholds, purity, color=PALETTE[1], lw=2, label="purity")
    axes[1].set(xlabel=f"threshold on P({target})", ylim=(0, 1.02), title="Efficiency and purity")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return auc


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("catalog", help="Simulated HATS catalog (from simulate_dp2.py)")
    parser.add_argument("predictions", help="Predictions table (from snn_classify_rubin_dia.py)")
    parser.add_argument("--output-dir", default="snn_evaluation")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold on P(target) for recall/purity")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    table, classes = load(args.catalog, args.predictions)
    target = classes[0]
    types = sorted(table["type"].unique())
    truth = is_target(table["type"], target)
    selected = (table[f"p_{target}"] >= args.threshold).to_numpy()

    plot_probability(table, target, types, out / "probability.png")
    plot_predicted(table, classes, types, out / "predicted.png")
    plot_recall(table, target, types, args.threshold, out / "recall.png")

    print(f"{len(table)} classified objects; classes {classes}; target {target}")
    for label in types:
        sel = (table["type"] == label).to_numpy()
        print(f"  {label}: {selected[sel].mean():.3f} with P({target}) >= {args.threshold}, median P {table[f'p_{target}'][sel].median():.3f}")
    if truth.any() and not truth.all():
        auc = plot_roc(table, target, truth, out / "roc.png")
        efficiency = selected[truth].mean()
        purity = truth[selected].mean() if selected.any() else float("nan")
        print(f"{target} vs rest: AUC {auc:.3f}; at {args.threshold}: efficiency {efficiency:.3f}, purity {purity:.3f}")
    table.to_parquet(out / "evaluation.parquet")
    print(f"Wrote figures to {out}")


if __name__ == "__main__":
    main()
