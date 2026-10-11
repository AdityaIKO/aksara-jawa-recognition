"""Read the experiment log and turn it into tables, statistics and figures."""
import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

INK, MUTED, ACCENT, ACCENT2, GRID = "#1f2933", "#7b8794", "#0b6e99", "#c8553d", "#e4e7eb"


def style():
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight", "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": MUTED,
        "axes.labelcolor": INK, "text.color": INK, "xtick.color": INK, "ytick.color": INK,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
        "axes.titleweight": "bold", "axes.titlesize": 11, "legend.frameon": False,
    })


def load_runs(root="."):
    rows = [json.loads(l) for l in open(os.path.join(root, "results", "runs.jsonl"))]
    return pd.DataFrame(rows).drop_duplicates(["stage", "name", "split", "seed"], keep="last").reset_index(drop=True)


def summarise(df, stage, order=None):
    """Mean and standard deviation over splits for every configuration of one stage (accuracy in %)."""
    d = df[df.stage == stage]
    g = d.groupby("name", sort=False)
    out = pd.DataFrame({
        "val acc": g.val_acc.mean() * 100, "val sd": g.val_acc.std() * 100,
        "test acc": g.test_acc.mean() * 100, "test sd": g.test_acc.std() * 100,
        "test F1": g.test_f1.mean() * 100, "epochs": g.best_epoch.mean(),
        "sec/run": g.seconds.mean(), "params": g.params.first(), "runs": g.size(),
    })
    return out.loc[order] if order else out


def load_preds(root, stage, name):
    """Stack the saved test-set probabilities of a stage. Returns dict split -> {seed: (idx, proba)}."""
    out = {}
    for f in sorted(glob.glob(os.path.join(root, "results", "preds", f"{stage}_{name}_*_s*.npz"))):
        split, seed = os.path.basename(f)[len(f"{stage}_{name}_"):-4].rsplit("_s", 1)
        z = np.load(f)
        out.setdefault(split, {})[int(seed)] = (z["idx"], z["proba"])
    return out


def cluster_bootstrap(correct, groups, n_boot=5000, seed=0):
    """95% confidence interval for accuracy when images come in correlated clusters.

    The ~29 images of one drawing are near-duplicates, so the drawings (not the images) are
    resampled with replacement."""
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(groups, return_inverse=True)
    num = np.bincount(inv, weights=correct.astype(float))
    den = np.bincount(inv).astype(float)
    pick = rng.integers(0, len(uniq), size=(n_boot, len(uniq)))
    accs = num[pick].sum(1) / den[pick].sum(1)
    return float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5))


def wilson(k, n, z=1.96):
    """Naive binomial (Wilson) interval that wrongly treats every image as independent."""
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


def bar_with_points(ax, labels, values_per_label, color=ACCENT, highlight=None, fmt="{:.1f}"):
    """Horizontal bars of the mean with one dot per fold: shows the spread, not only the mean."""
    means = [np.mean(v) for v in values_per_label]
    y = np.arange(len(labels))[::-1]
    cols = [ACCENT2 if (highlight and l in highlight) else color for l in labels]
    ax.barh(y, means, color=cols, height=0.62, alpha=0.9)
    for yi, v in zip(y, values_per_label):
        ax.scatter(v, np.full(len(v), yi), s=20, color=INK, zorder=3, edgecolors="white", linewidths=0.7)
    for yi, m, v in zip(y, means, values_per_label):
        ax.annotate(fmt.format(m), (max(m, np.max(v)), yi), xytext=(6, 0), textcoords="offset points",
                    va="center", ha="left", fontsize=9, fontweight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.grid(axis="y", visible=False)
    return ax
