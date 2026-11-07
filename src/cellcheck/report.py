"""Figures and a results table, drawn only from saved result files."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import pandas as pd
from matplotlib.ticker import MaxNLocator

INK = "#111827"
MUTED = "#6B7280"
GRID = "#E5E7EB"
FLOOR = "#9A3412"
COLORS = {
    "no_change": "#C9CED6",
    "additive": "#9AA3AF",
    "scaled_additive": "#6B7280",
    "random": "#A9BED6",
    "shuffled": "#6F93BD",
    "model": "#1F4E8C",
}


def _style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": MUTED,
        "ytick.color": INK,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def _labels(model: str) -> dict:
    pretty = model.capitalize()
    return {
        "no_change": "No change",
        "additive": "Additive",
        "scaled_additive": "Scaled additive",
        "random": "Random embeddings",
        "shuffled": f"Shuffled {pretty} embeddings",
        model: f"{pretty} embeddings",
    }


def _load(results_dir: Path):
    summary = pd.read_csv(results_dir / "summary.csv", index_col="method")
    meta = json.loads((results_dir / "results.json").read_text())
    return summary, meta


def _color(method: str, model: str) -> str:
    return COLORS["model"] if method == model else COLORS.get(method, MUTED)


def fig_rmse(summary, meta, out: Path):
    model, floor = meta["embedding"], meta["noise_floor_rmse"]
    lab = _labels(model)
    order = ["no_change", "additive", "scaled_additive", "random", "shuffled", model]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.4), layout="constrained", sharey=True)

    y = range(len(order))
    vals = [summary.loc[m, "mean_rmse"] for m in order]
    errs = [summary.loc[m, "std_across_splits"] for m in order]
    a1.barh(list(y), vals, xerr=errs, color=[_color(m, model) for m in order],
            error_kw={"ecolor": INK, "lw": 1, "capsize": 3})
    a1.axvline(floor, color=FLOOR, ls="--", lw=1.2)
    a1.text(floor + max(vals) * 0.015, -0.62, "estimated noise floor", color=FLOOR, fontsize=8.5, va="center")
    a1.set_yticks(list(y), [lab[m] for m in order])
    a1.invert_yaxis()
    a1.set_xlim(0, max(vals) * 1.22)
    a1.set_xlabel("Prediction error (RMSE), lower is better")
    a1.set_title("All methods", loc="left", fontsize=11, color=INK)
    for yi, v in zip(y, vals):
        a1.text(v + max(vals) * 0.05, yi, f"{v:.4f}", va="center", fontsize=8.5, color=INK)

    for yi, m in enumerate(order):
        if m == "no_change":
            continue
        a2.errorbar(summary.loc[m, "mean_rmse"], yi, xerr=summary.loc[m, "std_across_splits"],
                    fmt="o", color=_color(m, model), ecolor=_color(m, model), capsize=3, markersize=7)
    a2.axvline(floor, color=FLOOR, ls="--", lw=1.2)
    a2.tick_params(axis="y", length=0)
    zoom_m = [m for m in order if m != "no_change"]
    lo = min([floor] + [summary.loc[m, "mean_rmse"] - summary.loc[m, "std_across_splits"] for m in zoom_m])
    hi = max(summary.loc[m, "mean_rmse"] + summary.loc[m, "std_across_splits"] for m in zoom_m)
    pad = (hi - lo) * 0.08
    a2.set_xlim(lo - pad, hi + pad)
    a2.xaxis.set_major_locator(MaxNLocator(6))
    a2.text(floor + (hi - lo) * 0.02, -0.62, "estimated noise floor", color=FLOOR, fontsize=8.5, va="center")
    a2.set_xlabel("Prediction error (RMSE), lower is better")
    a2.set_title("Zoom, without the no-change reference", loc="left", fontsize=11, color=INK)
    a2.grid(axis="x", color=GRID, lw=0.8)
    a2.set_axisbelow(True)
    fig.supxlabel(f"Mean over {meta['n_splits']} random splits; bars show one standard deviation across splits. "
                  "Dashed line: estimated noise floor.", fontsize=8.5, color=MUTED)
    fig.savefig(out / "rmse_by_method.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_gap(summary, meta, out: Path):
    model = meta["embedding"]
    lab = _labels(model)
    order = ["scaled_additive", "random", "shuffled", model]
    vals = [summary.loc[m, "gap_to_floor_closed_pct"] for m in order]
    fig, ax = plt.subplots(figsize=(8.6, 3.4), layout="constrained")
    y = list(range(len(order)))
    ax.barh(y, vals, color=[_color(m, model) for m in order])
    ax.set_xlim(0, 100)
    ax.axvline(0, color=INK, lw=1)
    ax.axvline(100, color=FLOOR, ls="--", lw=1.2)
    ax.text(99.5, -0.62, "noise floor", color=FLOOR, ha="right", fontsize=8.5)
    ax.set_yticks(y, [lab[m] for m in order])
    ax.invert_yaxis()
    ax.set_xlabel("Gap between additive (0) and the noise floor (100) that is closed, in percent")
    for yi, v in zip(y, vals):
        ax.text(v + 1.2, yi, f"{v:.1f}", va="center", fontsize=9, color=INK)
    ax.set_title("Most of the distance to the estimated noise floor is still open", loc="left", fontsize=11, color=INK)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    fig.savefig(out / "gap_to_noise_floor.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_paired(meta, out: Path):
    model = meta["embedding"]
    pretty = model.capitalize()
    items = [
        (f"{pretty} embeddings vs shuffled embeddings", meta["paired"]["model_vs_shuffled"]),
        (f"{pretty} embeddings vs scaled additive", meta["paired"]["model_vs_scaled_additive"]),
    ]
    fig, ax = plt.subplots(figsize=(9.5, 2.9), layout="constrained")
    lows = [p["range_95_low"] for _, p in items] + [0.0]
    highs = [p["range_95_high"] for _, p in items]
    span = max(highs) - min(lows)
    for i, (label, p) in enumerate(items):
        ax.plot([p["range_95_low"], p["range_95_high"]], [i, i], color=COLORS["model"], lw=2.2)
        ax.plot(p["mean_improvement"], i, "o", color=COLORS["model"], markersize=8)
        share = 100 * p["better_in_fraction_of_splits"]
        ax.text(p["range_95_high"] + 0.03 * span, i, f"better in {share:.0f}% of splits",
                va="center", fontsize=9, color=INK)
    ax.axvline(0, color=INK, lw=1)
    ax.set_yticks(range(len(items)), [l for l, _ in items])
    ax.set_ylim(len(items) - 0.4, -0.6)
    ax.set_xlim(min(lows) - 0.08 * span, max(highs) + 0.55 * span)
    ax.set_xlabel("Reduction in RMSE per split (positive favors the embeddings)")
    ax.set_title("Paired differences across splits (dot: mean, line: 95% of splits)", loc="left", fontsize=11, color=INK)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    fig.supxlabel("Beating scaled additive reflects the linear model fitted on training doubles; "
                  "random embeddings also beat it (see rmse_by_method.png).", fontsize=8.5, color=MUTED)
    fig.savefig(out / "paired_differences.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def _box(ax, x, y, w, h, text, fc="#F3F4F6", ec=GRID, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.025",
                                fc=fc, ec=ec, lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9.2, color=INK,
            fontweight="bold" if bold else "normal", wrap=True)


def _arrow(ax, x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=12,
                                 color=MUTED, lw=1.2))


def fig_overview(out: Path):
    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    w, h = 0.27, 0.17
    xs = [0.02, 0.365, 0.71]
    _box(ax, xs[0], 0.74, w, h, "Cells from a perturbation\nexperiment (public data)")
    _box(ax, xs[1], 0.74, w, h, "One average expression\nprofile per perturbation")
    _box(ax, xs[2], 0.74, w, h, "Train: all single perturbations\n+ half of the doubles\nTest: the other half")
    _box(ax, xs[0], 0.43, w, h, "Gene vectors from the\nmodel under test")
    _box(ax, xs[1], 0.43, w, h, "A perturbation is the sum\nof its gene vectors")
    _box(ax, xs[2], 0.43, w, h, "Ridge regression predicts the\nchange in expression")
    for i in range(2):
        _arrow(ax, xs[i] + w + 0.005, 0.825, xs[i + 1] - 0.005, 0.825)
        _arrow(ax, xs[i] + w + 0.005, 0.515, xs[i + 1] - 0.005, 0.515)
    _arrow(ax, xs[2] + w / 2, 0.735, xs[2] + w / 2, 0.605)
    _arrow(ax, xs[2] + w / 2, 0.425, xs[2] + w / 2, 0.295)
    _box(ax, 0.02, 0.10, 0.96, 0.18,
         "Score on held-out doubles (RMSE), repeated over many random splits, against:\n"
         "no change   |   additive   |   scaled additive   |   shuffled embeddings   |   random embeddings"
         "   |   noise floor", fc="#EEF2FF", ec="#C7D2FE", bold=False)
    ax.text(0.02, 0.02, "No model is trained or fine-tuned. The only fitted component is the ridge regression.",
            fontsize=9, color=MUTED)
    fig.savefig(out / "overview.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def results_table(summary, meta) -> str:
    model = meta["embedding"]
    lab = _labels(model)
    order = ["no_change", "additive", "scaled_additive", "random", "shuffled", model]
    lines = ["| Method | Mean RMSE | Std across splits | Gap to noise floor closed (%) |",
             "|---|---|---|---|"]
    for m in order:
        r = summary.loc[m]
        lines.append(f"| {lab[m]} | {r['mean_rmse']:.4f} | {r['std_across_splits']:.4f} | "
                     f"{r['gap_to_floor_closed_pct']:.1f} |")
    lines.append("")
    lines.append(f"Estimated noise floor: {meta['noise_floor_rmse']:.4f}. "
                 f"Splits: {meta['n_splits']}.")
    return "\n".join(lines)


def make_report(results_dir, out_dir) -> list:
    _style()
    results_dir, out_dir = Path(results_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary, meta = _load(results_dir)
    fig_rmse(summary, meta, out_dir)
    fig_gap(summary, meta, out_dir)
    fig_paired(meta, out_dir)
    fig_overview(out_dir)
    (results_dir / "RESULTS.md").write_text(results_table(summary, meta) + "\n")
    return sorted(p.name for p in out_dir.glob("*.png")) + [str(results_dir / "RESULTS.md")]
