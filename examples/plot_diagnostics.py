"""Diagnostic plots for ParSNIP run on a simulated DP2 catalog.

    python examples/plot_diagnostics.py CATALOG PREDICTIONS [--output-dir DIR]

CATALOG is a HATS catalog written by ``simulate_dp2.py`` (it has the truth columns
``type``, ``redshift`` and ``t0``). PREDICTIONS is the table written by
``classify_rubin_dia.py`` for that catalog. Trains a LightGBM classifier on the
predictions with K-folding and saves, in DIR:

- ``sample.png``: true redshift and number of detections per class
- ``lightcurves.png``: typical light curves of each class, with the out-of-sample prediction
- ``redshift.png``: ParSNIP predicted vs true redshift, and the residual vs true redshift
  (only for photo-z models, whose predictions include ``predicted_redshift``)
- ``latent.png``: the ParSNIP latent space (s1, s2, s3) colored by class
- ``confusion.png``: out-of-sample confusion matrix
- ``accuracy.png``: out-of-sample accuracy vs light-curve S/N and vs true redshift
- ``out_of_sample.parquet``: out-of-sample class probabilities, with the truth
"""

import argparse
from pathlib import Path

import lsdb
import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.table import Table

import hyrax_parsnip

BANDS = ["u", "g", "r", "i", "z", "y"]
# Fixed categorical order (blue, orange, aqua, yellow, magenta, green); classes keep
# their color in every figure.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
BAND_COLORS = dict(zip(BANDS, PALETTE))
SEQUENTIAL = "Blues"
TRUTH_COLUMNS = ["diaObjectId", "type", "redshift", "t0", "nDiaSources"]

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


def load(catalog_path, predictions_path, num_folds, min_child_weight):
    """Truth + ParSNIP features + out-of-sample probabilities, one row per object."""
    frame = lsdb.open_catalog(str(catalog_path), columns=[*TRUTH_COLUMNS, "diaObjectForcedSource"]).compute()
    frame = frame.set_index("diaObjectId", drop=False)

    predictions = Table.from_pandas(pd.read_parquet(predictions_path))
    ids = np.asarray(predictions["diaObjectId"]).astype(str)
    predictions["object_id"] = ids
    predictions["original_object_id"] = ids
    predictions["type"] = frame.loc[np.asarray(predictions["diaObjectId"]), "type"].to_numpy().astype(str)

    classifier, out_of_sample = hyrax_parsnip.train_classifier(
        predictions, label_column="type", num_folds=num_folds, min_child_weight=min_child_weight
    )
    class_names = [str(c) for c in classifier.class_names]
    oos = out_of_sample.to_pandas()[["object_id", *class_names]]
    oos["diaObjectId"] = oos.pop("object_id").astype(np.int64)

    table = predictions.to_pandas().drop(columns=["object_id", "original_object_id", "type"])
    table = table.merge(frame[TRUTH_COLUMNS].reset_index(drop=True), on="diaObjectId")
    table = table.merge(oos, on="diaObjectId", how="left")
    probs = table[class_names].to_numpy()
    table["predicted_type"] = np.where(
        np.isfinite(probs).all(axis=1), np.asarray(class_names)[np.nan_to_num(probs, nan=-1).argmax(axis=1)], None
    )
    table["p_max"] = np.nanmax(probs, axis=1)
    return frame, table, class_names


def class_colors(class_names):
    return dict(zip(class_names, PALETTE))


def plot_sample(table, class_names, path):
    colors = class_colors(class_names)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    z_bins = np.linspace(0, table["redshift"].max() * 1.02, 31)
    det_bins = np.arange(0, min(table["nDiaSources"].quantile(0.99), 80) + 2, 2)
    for label in class_names:
        sel = table[table["type"] == label]
        kwargs = dict(histtype="step", lw=2, color=colors[label], label=f"{label} ({len(sel)})")
        axes[0].hist(sel["redshift"], bins=z_bins, **kwargs)
        axes[1].hist(sel["nDiaSources"].clip(upper=det_bins[-1] - 1), bins=det_bins, **kwargs)
    axes[0].set(xlabel="true redshift", ylabel="objects", title="Redshift")
    axes[1].set(xlabel="detections (nDiaSources, S/N ≥ 5)", ylabel="objects", title="Detections per object")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_lightcurves(frame, table, class_names, path, per_class=4, seed=0):
    """Random objects of each class with a median-ish number of detections."""
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(len(class_names), per_class, figsize=(3.4 * per_class, 2.5 * len(class_names)))
    for row_axes, label in zip(axes, class_names):
        sel = table[(table["type"] == label) & table["predicted_type"].notna()]
        lo, hi = sel["nDiaSources"].quantile([0.25, 0.75])
        typical = sel[sel["nDiaSources"].between(lo, hi)]
        picks = typical.iloc[rng.choice(len(typical), size=min(per_class, len(typical)), replace=False)]
        for ax, (_, obj) in zip(row_axes, picks.iterrows()):
            lc = frame.loc[obj["diaObjectId"], "diaObjectForcedSource"]
            lc = lc[(lc["midpointMjdTai"] > obj["t0"] - 30) & (lc["midpointMjdTai"] < obj["t0"] + 100)]
            for band in BANDS:
                b = lc[lc["band"] == band]
                if len(b):
                    ax.errorbar(
                        b["midpointMjdTai"] - obj["t0"], b["psfDiffFlux"], b["psfDiffFluxErr"],
                        fmt="o", ms=3, lw=0.8, color=BAND_COLORS[band], label=band,
                    )
            ok = "✓" if obj["predicted_type"] == label else "✗"
            ax.set_title(
                f"{label}, z={obj['redshift']:.2f} → {obj['predicted_type']} "
                f"(p={obj['p_max']:.2f}) {ok}",
                fontsize=8,
            )
            ax.axhline(0, color="#999999", lw=0.6)
            ax.set_xlim(-30, 100)
        row_axes[0].set_ylabel("psfDiffFlux (nJy)")
    for ax in axes[-1]:
        ax.set_xlabel("days from true t0")
    handles = [plt.Line2D([], [], marker="o", ls="", color=BAND_COLORS[b], label=b) for b in BANDS]
    fig.legend(handles=handles, loc="upper center", ncol=len(BANDS), bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path)
    plt.close(fig)


def plot_redshift(table, class_names, path):
    colors = class_colors(class_names)
    t = table[np.isfinite(table["predicted_redshift"])]
    dz = (t["predicted_redshift"] - t["redshift"]) / (1 + t["redshift"])
    zmax = max(t["redshift"].max(), t["predicted_redshift"].max()) * 1.05

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for label in class_names:
        sel = t["type"] == label
        d = dz[sel]
        nmad = 1.4826 * np.median(np.abs(d - np.median(d)))
        name = f"{label}: bias {np.median(d):+.3f}, NMAD {nmad:.3f}"
        axes[0].scatter(t["redshift"][sel], t["predicted_redshift"][sel], s=5, alpha=0.4, color=colors[label], label=name)
        axes[1].scatter(t["redshift"][sel], d[sel], s=5, alpha=0.4, color=colors[label])
        # Running median of the residual, to show the trend under the scatter.
        bins = np.linspace(t["redshift"][sel].min(), t["redshift"][sel].max(), 9)
        idx = np.digitize(t["redshift"][sel], bins)
        centers = [(bins[i - 1] + bins[i]) / 2 for i in range(1, len(bins)) if (idx == i).sum() >= 10]
        medians = [np.median(d[idx == i]) for i in range(1, len(bins)) if (idx == i).sum() >= 10]
        axes[1].plot(centers, medians, color=colors[label], lw=2)
    axes[0].plot([0, zmax], [0, zmax], color="#555555", ls="--", lw=1)
    axes[0].set(xlim=(0, zmax), ylim=(0, zmax), xlabel="true redshift", ylabel="predicted redshift")
    nmad_all = 1.4826 * np.median(np.abs(dz - np.median(dz)))
    axes[0].set_title(f"plasticc_photoz: {len(t)} objects, NMAD {nmad_all:.3f}")
    axes[0].legend(loc="upper left", markerscale=3)
    axes[1].axhline(0, color="#555555", ls="--", lw=1)
    axes[1].set(xlabel="true redshift", ylabel="Δz / (1 + z)", title="Residual (lines: running median)")
    axes[1].set_ylim(np.percentile(dz, [0.5, 99.5]) * 1.2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def finite_range(values, percentiles=(0.5, 99.5)):
    values = np.asarray(values, dtype=np.float64)
    return np.percentile(values[np.isfinite(values)], percentiles)


def plot_latent(table, class_names, path):
    colors = class_colors(class_names)
    t = table[table["predicted_type"].notna()]
    pairs = [("s1", "s2"), ("s1", "s3"), ("s2", "s3"), ("color", "luminosity")]
    fig, axes = plt.subplots(1, len(pairs), figsize=(4 * len(pairs), 3.8))
    for ax, (x, y) in zip(axes, pairs):
        for label in class_names:
            sel = t["type"] == label
            ax.scatter(t[x][sel], t[y][sel], s=4, alpha=0.35, color=colors[label], label=label)
        ax.set(xlabel=x, ylabel=y)
        ax.set_xlim(finite_range(t[x]))
        ax.set_ylim(finite_range(t[y]))
    axes[-1].invert_yaxis()  # brighter up
    axes[0].legend(markerscale=4)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_confusion(table, class_names, path):
    t = table[table["predicted_type"].notna()]
    counts = pd.crosstab(t["type"], t["predicted_type"]).reindex(index=class_names, columns=class_names, fill_value=0)
    frac = counts.to_numpy() / counts.to_numpy().sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(4.6, 4))
    ax.imshow(frac, cmap=SEQUENTIAL, vmin=0, vmax=1)
    ax.grid(False)
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(
                j, i, f"{frac[i, j]:.2f}\n({counts.iat[i, j]})", ha="center", va="center", fontsize=9,
                color="white" if frac[i, j] > 0.55 else "#1a1a1a",
            )
    ax.set_xticks(range(len(class_names)), class_names)
    ax.set_yticks(range(len(class_names)), class_names)
    ax.set(xlabel="predicted", ylabel="true")
    accuracy = (t["type"] == t["predicted_type"]).mean()
    ax.set_title(f"Out-of-sample: accuracy {accuracy:.3f} ({len(t)} objects)\nrows normalized")
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def binned_accuracy(x, correct, bins):
    idx = np.digitize(x, bins)
    keep = [i for i in range(1, len(bins)) if (idx == i).sum() >= 10]
    centers = np.array([np.sqrt(bins[i - 1] * bins[i]) if bins[0] > 0 else (bins[i - 1] + bins[i]) / 2 for i in keep])
    acc = np.array([correct[idx == i].mean() for i in keep])
    n = np.array([(idx == i).sum() for i in keep])
    return centers, acc, np.sqrt(acc * (1 - acc) / n)


def plot_accuracy(table, class_names, path):
    """Per-class recall (fraction of each true class classified correctly)."""
    colors = class_colors(class_names)
    t = table[table["predicted_type"].notna()]
    correct = (t["type"] == t["predicted_type"]).to_numpy()
    s2n_bins = np.geomspace(max(t["total_s2n"].quantile(0.01), 1), t["total_s2n"].quantile(0.99), 9)
    z_bins = np.linspace(t["redshift"].min(), t["redshift"].max(), 9)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for ax, column, bins in [(axes[0], "total_s2n", s2n_bins), (axes[1], "redshift", z_bins)]:
        for label in class_names:
            sel = (t["type"] == label).to_numpy()
            x, acc, err = binned_accuracy(t[column].to_numpy()[sel], correct[sel], bins)
            ax.errorbar(x, acc, err, color=colors[label], lw=2, marker="o", ms=5, capsize=0, label=label)
        ax.axhline(1 / len(class_names), color="#999999", ls=":", lw=1)
    axes[0].set_xscale("log")
    axes[0].xaxis.set_major_locator(matplotlib.ticker.FixedLocator([10, 20, 50, 100, 200, 500, 1000]))
    axes[0].xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    axes[0].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    axes[0].set(xlabel="light-curve total S/N", ylabel="recall (fraction correct)", ylim=(0, 1.02))
    axes[1].set(xlabel="true redshift")
    axes[0].legend(loc="lower right")
    fig.suptitle("Out-of-sample recall per class (dotted: chance)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("catalog", help="Simulated HATS catalog (from simulate_dp2.py)")
    parser.add_argument("predictions", help="Predictions table (from classify_rubin_dia.py)")
    parser.add_argument("--output-dir", default="diagnostics")
    parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument("--min-child-weight", type=float, default=10.0)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    frame, table, class_names = load(args.catalog, args.predictions, args.num_folds, args.min_child_weight)

    plot_sample(table, class_names, out / "sample.png")
    plot_lightcurves(frame, table, class_names, out / "lightcurves.png")
    if "predicted_redshift" in table:
        plot_redshift(table, class_names, out / "redshift.png")
    plot_latent(table, class_names, out / "latent.png")
    plot_confusion(table, class_names, out / "confusion.png")
    plot_accuracy(table, class_names, out / "accuracy.png")
    table.to_parquet(out / "out_of_sample.parquet")

    accuracy = (table["type"] == table["predicted_type"])[table["predicted_type"].notna()].mean()
    print(f"Out-of-sample accuracy {accuracy:.3f}; wrote plots to {out}")


if __name__ == "__main__":
    main()
