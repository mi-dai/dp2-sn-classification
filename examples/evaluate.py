"""Evaluate ParSNIP or SuperNNova predictions on a simulated DP2 catalog.

    python examples/evaluate.py CATALOG PREDICTIONS [--output-dir DIR] [--threshold 0.5] \\
        [--kfold-lightgbm [--binary-class SNIa]]

CATALOG is a HATS catalog written by ``simulate_dp2.py`` (truth columns ``type``,
``redshift``, ``t0``, ``nDiaSources``). PREDICTIONS is the table written by
``classify_rubin_dia.py`` for that catalog, with either method.

The class probabilities in PREDICTIONS are scored as they are: the simulation is used for
validation only. ParSNIP predictions need a classifier for that (``--classifier`` in
``classify_rubin_dia.py``). ``--kfold-lightgbm`` instead trains ParSNIP's LightGBM
classifier **on this catalog** with K-folding and scores the out-of-sample probabilities
(with ``--binary-class LABEL``, LABEL vs non-LABEL); this is a quick check, not a validation.

If the model was given each object's redshift (``classify_rubin_dia.py
--redshift-column``), a warning is printed and added to the confusion/predicted figure:
ParSNIP's classifier has no redshift feature, but with ``plasticc`` the redshift enters
through ``luminosity`` and the rest-frame latents.

Scoring depends on the classes:
- matched (the classes are the true types, possibly plus extra ones, e.g. a classifier
  trained on PLAsTiCC with --classes dp2, SNIa/SNII/SNIbc/other): confusion matrix, accuracy, recall per type;
- target (classes that aren't true types, e.g. SNIa vs non-SNIa, SuperNNova's SNIa/other or
  broad classes): the target is SNIa if it is a class, otherwise the first class, scored as
  SNIa → type SNIa, SN → any SN, and any other class (Fast, Long, ...) has no members in the
  simulation.

`evaluate()` makes the same figures from Python (e.g. the demo notebook): join an in-memory
predictions table with `prepare(predictions, load_truth(catalog)[1])` first.

Saves, in DIR (default ``examples/results/eval_<PREDICTIONS name>/``):
- ``sample.png``: true redshift and number of detections per type
- ``lightcurves.png``: typical light curves of each type, with the predicted class
- ``confusion.png``: SN Ia vs non-Ia confusion matrix (true SNIa vs any other type, predicted
  P(SNIa) ≥ threshold) for models with an SNIa class; otherwise predicted class per true type
  (``confusion.png`` when the classes are the true types, ``predicted.png`` when not)
- ``recall.png``: fraction correct (matched) or with P(target) ≥ threshold (target), vs
  light-curve S/N and true redshift, per type
- ``probability.png``: P(target) per true type (target mode)
- ``roc.png``: ROC and efficiency/purity of the target class, when the simulation has both
  members and non-members (e.g. SNIa)
- ``redshift.png``: predicted vs true redshift (ParSNIP photo-z models)
- ``latent.png``: ParSNIP latent space (when the predictions have ParSNIP features)
- ``evaluation.parquet``: predictions joined with the truth
"""

import argparse
from pathlib import Path

import lsdb
import matplotlib
import matplotlib.ticker
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.table import Table

import hyrax_parsnip

BANDS = ["u", "g", "r", "i", "z", "y"]
# Fixed categorical order (blue, orange, aqua, yellow, magenta, green); types keep
# their color in every figure.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
BAND_COLORS = dict(zip(BANDS, PALETTE))
SEQUENTIAL = "Blues"
TRUTH_COLUMNS = ["diaObjectId", "type", "redshift", "t0", "nDiaSources"]
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


# ── loading ──────────────────────────────────────────────────────────────────


def load_truth(catalog_path):
    """Truth per object plus its light-curve S/N, and the catalog frame (for light curves)."""
    frame = lsdb.open_catalog(str(catalog_path), columns=[*TRUTH_COLUMNS, "diaObjectForcedSource"]).compute()
    frame = frame.set_index("diaObjectId", drop=False)

    flat = frame["diaObjectForcedSource"].nest.to_flat()
    good = np.ones(len(flat), dtype=bool)
    for column in hyrax_parsnip.RUBIN_DIA_SETTINGS["flag_columns"]:
        if column in flat.columns:
            good &= ~flat[column].fillna(False).to_numpy(dtype=bool)
    snr2 = (flat["psfDiffFlux"].to_numpy(np.float64) / flat["psfDiffFluxErr"].to_numpy(np.float64)) ** 2
    snr = pd.Series(np.where(good & np.isfinite(snr2), snr2, 0.0), index=flat.index).groupby(level=0).sum() ** 0.5

    truth = frame[TRUTH_COLUMNS].reset_index(drop=True)
    truth["snr"] = snr.reindex(truth["diaObjectId"]).to_numpy()
    return frame, truth


def kfold_probabilities(predictions, truth, num_folds, min_child_weight):
    """Out-of-sample probabilities from ParSNIP's LightGBM classifier trained on this catalog
    (on the truth types, already relabeled for --binary-class)."""
    if "s1" not in predictions:
        raise SystemExit("--kfold-lightgbm needs ParSNIP predictions (classify_rubin_dia.py --method parsnip).")
    table = Table.from_pandas(predictions.drop(columns=[c for c in predictions if c.startswith("p_")]))
    ids = np.asarray(table["diaObjectId"]).astype(str)
    table["object_id"] = ids
    table["original_object_id"] = ids
    table["type"] = truth.set_index("diaObjectId").loc[np.asarray(table["diaObjectId"]), "type"].to_numpy().astype(str)

    classifier, out_of_sample = hyrax_parsnip.train_classifier(
        table, label_column="type", num_folds=num_folds, min_child_weight=min_child_weight
    )
    class_names = [str(c) for c in classifier.class_names]
    oos = out_of_sample.to_pandas()[["object_id", *class_names]]
    oos["diaObjectId"] = oos.pop("object_id").astype(np.int64)

    out = predictions.drop(columns=[c for c in predictions if c.startswith("p_")] + ["predicted_class"])
    out = out.merge(oos.rename(columns={c: f"p_{c}" for c in class_names}), on="diaObjectId", how="left")
    probs = out[[f"p_{c}" for c in class_names]].to_numpy()
    valid = np.isfinite(probs).all(axis=1)
    out["predicted_class"] = np.where(valid, np.asarray(class_names)[np.nan_to_num(probs, nan=-1).argmax(axis=1)], "")
    return out


def load(catalog_path, predictions_path, args):
    frame, truth = load_truth(catalog_path)
    predictions = pd.read_parquet(predictions_path)
    if args.binary_class:
        types = truth["type"].astype(str)
        if args.binary_class not in set(types):
            raise SystemExit(f"--binary-class {args.binary_class!r} is not one of the types {sorted(set(types))}")
        truth["type"] = types.where(types == args.binary_class, f"non-{args.binary_class}")

    if args.kfold_lightgbm:
        predictions = kfold_probabilities(predictions, truth, args.num_folds, args.min_child_weight)
    elif not (predictions["predicted_class"] != "").any():
        raise SystemExit(
            "PREDICTIONS has no class probabilities. Run classify_rubin_dia.py --method parsnip with --classifier, "
            "use --method snn, or pass --kfold-lightgbm (trains on this catalog)."
        )
    table, classes = prepare(predictions, truth)
    return frame, table, classes


def prepare(predictions: pd.DataFrame, truth: pd.DataFrame):
    """Join a predictions table (`diaObjectId`, `p_<class>`, `predicted_class`, ...) with the truth.

    Returns the joined table and the class names (from the `p_` columns).
    """
    classes = [c[2:] for c in predictions if c.startswith("p_")]
    truth_columns = [c for c in truth if c == "diaObjectId" or c not in predictions]
    table = predictions.merge(truth[truth_columns], on="diaObjectId", how="left")
    if table["type"].isna().any():
        raise SystemExit("Some predictions have no truth: is PREDICTIONS from this CATALOG?")
    probs = table[[f"p_{c}" for c in classes]].to_numpy()
    table["p_max"] = np.nanmax(np.where(np.isfinite(probs), probs, -np.inf), axis=1)
    return table, classes


def used_redshift(table) -> bool:
    """Whether the predictions came from a model given each object's redshift.

    Uses classify_rubin_dia.py's `redshift_input` column; older tables: ParSNIP models
    without `predicted_redshift`, and SuperNNova models other than the no-redshift ones.
    """
    if "redshift_input" in table:
        return bool(table["redshift_input"].any())
    if table.get("method", pd.Series(["parsnip"])).iloc[0] == "snn":
        return table["model"].iloc[0] not in ("elasticc_ia", "SN_vs_other")
    return "predicted_redshift" not in table


def target_members(types: pd.Series, target: str) -> np.ndarray:
    """Whether each true type belongs to the target class of a target-mode model."""
    if target == "SNIa":
        return (types == "SNIa").to_numpy()
    if target == "SN":
        return types.isin(SN_TYPES).to_numpy()
    return np.zeros(len(types), dtype=bool)


# ── plots ────────────────────────────────────────────────────────────────────


def type_colors(types):
    return dict(zip(types, PALETTE))


def plot_sample(table, types, path):
    colors = type_colors(types)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    z_bins = np.linspace(0, table["redshift"].max() * 1.02, 31)
    det_bins = np.arange(0, min(table["nDiaSources"].quantile(0.99), 80) + 2, 2)
    for label in types:
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


def plot_lightcurves(frame, table, types, matched, path, per_class=4, seed=0):
    """Random classified objects of each type with a median-ish number of detections."""
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(len(types), per_class, figsize=(3.4 * per_class, 2.5 * len(types)), squeeze=False)
    for row_axes, label in zip(axes, types):
        sel = table[(table["type"] == label) & (table["predicted_class"] != "")]
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
            mark = (" ✓" if obj["predicted_class"] == label else " ✗") if matched else ""
            ax.set_title(
                f"{label}, z={obj['redshift']:.2f} → {obj['predicted_class']} (p={obj['p_max']:.2f}){mark}", fontsize=8
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


def draw_matrix(counts: pd.DataFrame, title, path, xlabel="predicted", ylabel="true"):
    """Counts with rows normalized (fraction and count in each cell)."""
    frac = counts.to_numpy() / np.maximum(counts.to_numpy().sum(axis=1, keepdims=True), 1)
    n_rows, n_cols = counts.shape
    fig, ax = plt.subplots(figsize=(max(4.6, 1.1 * n_cols + 2.2), max(4, 0.6 * n_rows + 1.8)))
    ax.imshow(frac, cmap=SEQUENTIAL, vmin=0, vmax=1, aspect="auto")
    ax.grid(False)
    for i in range(n_rows):
        for j in range(n_cols):
            ax.text(
                j, i, f"{frac[i, j]:.2f}\n({counts.iat[i, j]})", ha="center", va="center", fontsize=9,
                color="white" if frac[i, j] > 0.55 else "#1a1a1a",
            )
    ax.set_xticks(range(n_cols), counts.columns)
    ax.set_yticks(range(n_rows), counts.index)
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_ia_confusion(table, threshold, path, note=""):
    """SN Ia vs non-Ia confusion matrix: true SNIa vs any other type, predicted P(SNIa) >= threshold."""
    t = table[table["predicted_class"] != ""]
    labels = ["SNIa", "non-SNIa"]
    true = np.where(t["type"] == "SNIa", "SNIa", "non-SNIa")
    predicted = np.where(t["p_SNIa"] >= threshold, "SNIa", "non-SNIa")
    counts = pd.crosstab(pd.Series(true, name="true"), pd.Series(predicted, name="predicted"))
    counts = counts.reindex(index=labels, columns=labels, fill_value=0)
    accuracy = (true == predicted).mean()
    draw_matrix(
        counts, f"SN Ia vs non-Ia: accuracy {accuracy:.3f} ({len(t)} objects){note}\nrows normalized", path,
        xlabel=f"predicted (P(SNIa) ≥ {threshold})",
    )
    return accuracy


def plot_class_matrix(table, types, classes, matched, path, note=""):
    """Rows: true type; columns: predicted class; rows normalized."""
    t = table[table["predicted_class"] != ""]
    counts = pd.crosstab(t["type"], t["predicted_class"]).reindex(index=types, columns=classes, fill_value=0)
    if matched:
        accuracy = (t["type"] == t["predicted_class"]).mean()
        title = f"Accuracy {accuracy:.3f} ({len(t)} objects){note}\nrows normalized"
    else:
        title = f"Predicted class per true type ({len(t)} objects){note}\nrows normalized"
    draw_matrix(counts, title, path)


def binned_fraction(x, flag, bins, min_count=10):
    idx = np.digitize(x, bins)
    keep = [i for i in range(1, len(bins)) if (idx == i).sum() >= min_count]
    centers = np.array([np.sqrt(bins[i - 1] * bins[i]) if bins[0] > 0 else (bins[i - 1] + bins[i]) / 2 for i in keep])
    frac = np.array([flag[idx == i].mean() for i in keep])
    n = np.array([(idx == i).sum() for i in keep])
    return centers, frac, np.sqrt(frac * (1 - frac) / n)


def plot_recall(table, types, flag, ylabel, title, path, chance=None):
    """Fraction of each true type with `flag` set, vs light-curve S/N and true redshift."""
    colors = type_colors(types)
    t = table[table["predicted_class"] != ""]
    flag = flag[(table["predicted_class"] != "").to_numpy()]
    snr = t["snr"].to_numpy()
    s2n_bins = np.geomspace(max(np.nanquantile(snr, 0.01), 1), np.nanquantile(snr, 0.99), 9)
    z_bins = np.linspace(t["redshift"].min(), t["redshift"].max(), 9)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for ax, column, bins in [(axes[0], "snr", s2n_bins), (axes[1], "redshift", z_bins)]:
        for label in types:
            sel = (t["type"] == label).to_numpy()
            x, frac, err = binned_fraction(t[column].to_numpy()[sel], flag[sel], bins)
            ax.errorbar(x, frac, err, color=colors[label], lw=2, marker="o", ms=5, capsize=0, label=label)
        if chance is not None:
            ax.axhline(chance, color="#999999", ls=":", lw=1)
    axes[0].set_xscale("log")
    axes[0].xaxis.set_major_locator(matplotlib.ticker.FixedLocator([10, 20, 50, 100, 200, 500, 1000, 2000]))
    axes[0].xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    axes[0].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    axes[0].set(xlabel="light-curve S/N (forced photometry)", ylabel=ylabel, ylim=(0, 1.02))
    axes[1].set(xlabel="true redshift")
    axes[0].legend(loc="lower right")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_probability(table, target, types, path):
    colors = type_colors(types)
    t = table[table["predicted_class"] != ""]
    fig, ax = plt.subplots(figsize=(5, 3.4))
    bins = np.linspace(0, 1, 26)
    for label in types:
        sel = t["type"] == label
        ax.hist(t.loc[sel, f"p_{target}"], bins=bins, histtype="step", lw=2, color=colors[label], label=f"{label} ({sel.sum()})")
    ax.set(xlabel=f"P({target})", ylabel="objects", title=f"P({target}) by true type")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_roc(table, target, members, path):
    from sklearn.metrics import roc_auc_score, roc_curve

    score = table[f"p_{target}"].to_numpy()
    fpr, tpr, _ = roc_curve(members, score)
    auc = roc_auc_score(members, score)
    thresholds = np.linspace(0, 1, 101)
    efficiency = np.array([(score[members] >= t).mean() for t in thresholds])
    purity = np.array([members[score >= t].mean() if (score >= t).any() else np.nan for t in thresholds])

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


def plot_redshift(table, types, path):
    colors = type_colors(types)
    t = table[np.isfinite(table["predicted_redshift"])]
    dz = (t["predicted_redshift"] - t["redshift"]) / (1 + t["redshift"])
    zmax = max(t["redshift"].max(), t["predicted_redshift"].max()) * 1.05

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for label in types:
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
    model = table["model"].iloc[0] if "model" in table else "ParSNIP"
    axes[0].set_title(f"{model}: {len(t)} objects, NMAD {nmad_all:.3f}")
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


def plot_latent(table, types, path):
    colors = type_colors(types)
    t = table[np.isfinite(table["s1"])]
    pairs = [("s1", "s2"), ("s1", "s3"), ("s2", "s3"), ("color", "luminosity")]
    fig, axes = plt.subplots(1, len(pairs), figsize=(4 * len(pairs), 3.8))
    for ax, (x, y) in zip(axes, pairs):
        for label in types:
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


# ── main ─────────────────────────────────────────────────────────────────────


def choose_target(classes, binary_class=None):
    """The class scored as the target: --binary-class, else SNIa if it is a class, else the first class."""
    if binary_class:
        return binary_class
    return "SNIa" if "SNIa" in classes else classes[0]


def evaluate(frame, table, classes, out_dir, threshold=0.5, binary_class=None):
    """Make the diagnostic figures for one predictions table (joined with the truth by `prepare`) and print a summary.

    Returns ``{"figures": [paths in display order], "metrics": {...}}``.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    figures, metrics = [], {}
    types = sorted(table["type"].unique())
    # Matched: the classes are the true types, possibly plus extra ones (e.g. a classifier's "other").
    matched = set(types) <= set(classes) or set(classes) <= set(types)
    classified = (table["predicted_class"] != "").to_numpy()
    method = table["method"].iloc[0] if "method" in table else "parsnip"
    model = table["model"].iloc[0] if "model" in table else "?"
    print(f"{method}/{model}: {classified.sum()} of {len(table)} objects classified; classes {classes}")
    with_redshift = used_redshift(table)
    metrics["redshift_input"] = with_redshift
    warning = (
        f"WARNING: {model} was given each object's redshift as input (classify_rubin_dia.py --redshift-column or --photoz-column); "
        "these results include redshift information."
    )
    note = "\nmodel given the redshift" if with_redshift else ""
    if with_redshift:
        print(warning)

    def save(plot, name, *args, **kwargs):
        path = out / name
        result = plot(*args, path, **kwargs)
        figures.append(path)
        return result

    def save_matrix(matched):
        """SN Ia vs non-Ia confusion matrix when SNIa is a class, otherwise predicted class per true type."""
        if "SNIa" in classes:
            metrics["ia_accuracy"] = float(save(plot_ia_confusion, "confusion.png", table, threshold, note=note))
        else:
            save(plot_class_matrix, "confusion.png" if matched else "predicted.png", table, types, classes, matched, note=note)

    save(plot_sample, "sample.png", table, types)
    save(plot_lightcurves, "lightcurves.png", frame, table, types, matched)
    if matched:
        save_matrix(True)
        correct = (table["type"] == table["predicted_class"]).to_numpy()
        save(
            plot_recall, "recall.png", table, types, correct, "recall (fraction correct)",
            "Recall per true type (dotted: chance)", chance=1 / len(classes),
        )
        metrics["accuracy"] = float(correct[classified].mean())
        print(f"accuracy {metrics['accuracy']:.3f}")
        for label in types:
            sel = classified & (table["type"] == label).to_numpy()
            metrics[f"recall_{label}"] = float(correct[sel].mean())
            print(f"  {label}: recall {correct[sel].mean():.3f} ({sel.sum()})")
        target = binary_class if binary_class else None
    else:
        target = choose_target(classes)
        save_matrix(False)
        save(plot_probability, "probability.png", table, target, types)
        selected = (table[f"p_{target}"] >= threshold).to_numpy()
        save(
            plot_recall, "recall.png", table, types, selected, f"fraction with P({target}) ≥ {threshold}",
            f"Classified as {target}, per true type",
        )
        for label in types:
            sel = classified & (table["type"] == label).to_numpy()
            metrics[f"selected_{label}"] = float(selected[sel].mean())
            print(f"  {label}: {selected[sel].mean():.3f} with P({target}) >= {threshold}")

    if target is not None:
        t = table[classified]
        members = target_members(t["type"], target) if not matched else (t["type"] == target).to_numpy()
        if members.any() and not members.all():
            auc = save(plot_roc, "roc.png", t, target, members)
            selected = (t[f"p_{target}"] >= threshold).to_numpy()
            purity = members[selected].mean() if selected.any() else float("nan")
            metrics.update(
                target=target, auc=float(auc), efficiency=float(selected[members].mean()), purity=float(purity)
            )
            print(f"{target} vs rest: AUC {auc:.3f}; at {threshold}: efficiency {selected[members].mean():.3f}, purity {purity:.3f}")

    if "predicted_redshift" in table and np.isfinite(table["predicted_redshift"]).any():
        save(plot_redshift, "redshift.png", table, types)
    if {"s1", "s2", "s3", "color", "luminosity"} <= set(table.columns):
        save(plot_latent, "latent.png", table, types)
    table.to_parquet(out / "evaluation.parquet")
    print(f"Wrote figures to {out}")
    if with_redshift:
        print(warning)
    return {"figures": figures, "metrics": metrics}


def main():
    # Figures are only written to files. Set here, not at import, so notebooks can use the plot helpers.
    matplotlib.use("Agg")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("catalog", help="Simulated HATS catalog (from simulate_dp2.py)")
    parser.add_argument("predictions", help="Predictions table (from classify_rubin_dia.py)")
    parser.add_argument("--output-dir", help="Figures (default examples/results/eval_<PREDICTIONS name>/)")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold on P(target) (target mode)")
    kfold = parser.add_argument_group("ParSNIP K-fold LightGBM (trains on CATALOG)")
    kfold.add_argument("--kfold-lightgbm", action="store_true", help="Train ParSNIP's classifier on CATALOG with K-folding")
    kfold.add_argument("--num-folds", type=int, default=5)
    kfold.add_argument("--min-child-weight", type=float, default=10.0)
    kfold.add_argument("--binary-class", metavar="LABEL", help="Binary LABEL vs non-LABEL classifier (e.g. SNIa)")
    args = parser.parse_args()
    if args.binary_class and not args.kfold_lightgbm:
        parser.error("--binary-class is only used with --kfold-lightgbm")

    output_dir = args.output_dir or Path(__file__).resolve().parent / "results" / f"eval_{Path(args.predictions).stem}"
    frame, table, classes = load(args.catalog, args.predictions, args)
    evaluate(frame, table, classes, output_dir, args.threshold, args.binary_class)


if __name__ == "__main__":
    main()
