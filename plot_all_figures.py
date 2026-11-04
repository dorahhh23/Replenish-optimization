#!/usr/bin/env python3
"""Export Experiment B figures as PNG files without HTML or model training.

Place this file beside the existing results/ directory, then run:
    python plot_all_figures.py
Dependencies: numpy, pandas, matplotlib. LightGBM is NOT needed for plotting.
"""
from pathlib import Path
import argparse
import json
import textwrap

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter

METHODS = ["Empirical", "Recent13", "DampedTrendResidual", "LightGBM", "Poisson", "NegativeBinomial"]
LABELS = {
    "Empirical": "Historical empirical", "Recent13": "Recent 13-week empirical",
    "DampedTrendResidual": "Damped trend + residuals", "LightGBM": "Grouped LightGBM",
    "Poisson": "Poisson", "NegativeBinomial": "Negative binomial",
}
NORMS = {
    "training_sd": "Training standard deviation",
    "nonzero_median": "Median absolute non-zero training sales",
    "interval_width": "Predicted interval width",
}
LEVELS = [70, 80, 90]
GROUPS = ["Smooth", "Erratic", "Intermittent", "Lumpy", "Zero"]
GROUP_COLORS = ["#26758F", "#C2762D", "#449B85", "#8864AD", "#8D99A3"]
LEVEL_COLORS = ["#98BCCB", "#488BA5", "#174F6C"]
DENSITY_CMAP = plt.get_cmap("magma_r").copy()
DENSITY_CMAP.set_bad("white")
PANELS = {
    "All_SKUs": "All SKUs, including Zero-group",
    "Nonzero_groups": "All SKUs, excluding Zero-group",
    "Zero": "Zero group", "Smooth": "Smooth group", "Erratic": "Erratic group",
    "Intermittent": "Intermittent group", "Lumpy": "Lumpy group",
}
CASE_LABELS = {
    "main_W42": "Main B: test weeks 2026-W25 to W42",
    "sensitivity_W43": "Sensitivity B: test weeks 2026-W25 to provisional W43",
}
PERIOD_LABELS = {
    "main_W42": "training: 2025-W41 to 2026-W24, testing: 2026-W25 to 2026-W42",
    "sensitivity_W43": "training: 2025-W41 to 2026-W24, testing: 2026-W25 to 2026-W43",
}


def find_results(explicit):
    if explicit is not None:
        root = explicit.expanduser().resolve()
    else:
        home = Path(__file__).resolve().parent
        candidates = [home / "results", home / "B_experiment_zero60" / "results", home]
        root = next((p for p in candidates if (p / "main_W42" / "metrics_long.csv").is_file()), None)
        if root is None:
            raise FileNotFoundError(
                "Cannot find results/main_W42/metrics_long.csv. Put this script beside "
                "the existing results folder, or use --results PATH_TO_RESULTS. "
                "If results have not been generated, first run run_b_experiment.py."
            )
    if not root.is_dir():
        raise FileNotFoundError(f"Results directory does not exist: {root}")
    return root


def save_png(fig, path, dpi):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.png")
    fig.savefig(temporary, format="png", dpi=dpi, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    temporary.replace(path)
    print(f"Saved: {path}", flush=True)


def entry(path, category, case="", level="", norm="", panel=""):
    return dict(path=path.as_posix(), category=category, case=case,
                interval_level=level, normalization=norm, table_id=panel)


def plot_outside_points(points, weeks, destination, case, norms, levels, dpi):
    required = {"sku_id", "method", "demand_group", "iso_week", "interval_level"}
    required.update(f"signed_normalized_{n}" for n in norms)
    absent = required - set(points.columns)
    if absent:
        raise ValueError(f"Outside-point data are missing columns: {sorted(absent)}")
    unknown = set(points.method) - set(METHODS)
    unknown_groups = set(points.demand_group) - set(GROUPS)
    if unknown or unknown_groups:
        raise ValueError(f"Unrecognized methods/groups: {unknown} / {unknown_groups}")
    points = points.copy()
    points["week_index"] = points.iso_week.map({w: i for i, w in enumerate(weeks)})
    if points.week_index.isna().any():
        raise ValueError("Outside-point weeks do not match the scored test weeks.")
    hashed = pd.util.hash_pandas_object(
        points.sku_id.astype(str) + "|" + points.iso_week, index=False
    ).to_numpy(np.uint64)
    points["plot_x"] = points.week_index + (hashed % 10000).astype(float) / 10000 * .42 - .21
    files, audit = [], []
    for level in levels:
        data = points[points.interval_level.eq(level)]
        for norm in norms:
            key = f"signed_normalized_{norm}"
            values = data[key].to_numpy(float)
            if not np.isfinite(values).all() or np.any(values == 0):
                raise ValueError("Outside-point distances must be finite and non-zero.")
            maxabs = float(np.abs(values).max()) if len(values) else 1.
            fig, axes = plt.subplots(2, 3, figsize=(17, 10), sharex=True, sharey=True)
            plotted = 0
            for ax, method in zip(axes.ravel(), METHODS):
                method_points = data[data.method.eq(method)]
                for group, color in zip(GROUPS, GROUP_COLORS):
                    g = method_points[method_points.demand_group.eq(group)]
                    ax.scatter(g.plot_x, g[key], s=4, alpha=.23, color=color,
                               edgecolors="none", rasterized=True)
                    plotted += len(g)
                ax.set_yscale("symlog", linthresh=.1)
                low=min(0.,float(values.min())) if len(values) else 0.
                high=max(0.,float(values.max())) if len(values) else 0.
                ax.set_ylim(low-max(.02,abs(low)*.15),high+max(.02,high*.15))
                ax.axhline(0, color="#55636C", linewidth=.7)
                ax.set_title(f"{LABELS[method]}\nOutside points: {len(method_points):,}",
                             loc="left", fontsize=11)
                ax.spines[["top", "right"]].set_visible(False)
                ax.grid(axis="y", alpha=.14)
                ticks = sorted(set(list(range(0, len(weeks), 4)) + [len(weeks) - 1]))
                ax.set_xticks(ticks, [weeks[i] for i in ticks], rotation=35, fontsize=8)
                ax.set_xlim(-.6, len(weeks) - .4)
            if plotted != len(data):
                raise AssertionError("Not every outside point was plotted.")
            fig.suptitle(f"Every outside point | {level}% interval | {NORMS[norm]}\n"
                         f"{PERIOD_LABELS.get(case, case)}", fontsize=14, y=.995)
            fig.supxlabel("Test week; deterministic horizontal jitter separates overlaps", y=.066)
            fig.supylabel("Signed normalized outside distance (symlog scale)")
            fig.legend(handles=[Patch(color=c, label=g) for c, g in zip(GROUP_COLORS, GROUPS)],
                       loc="lower center", ncol=6, frameon=False)
            fig.tight_layout(rect=(.02, .10, 1, .94))
            path = destination / f"scatter_{level}_{norm}.png"
            save_png(fig, path, dpi)
            files.append(entry(path, "outside_scatter", case, level, norm))

            # Symmetric log1p bins retain small misses and the entire long tail.
            zmax = float(np.log1p(maxabs))
            xedges = np.linspace(-zmax, zmax, 101)
            yedges = np.arange(len(weeks) + 1) - .5
            histograms, counted = [], 0
            for method in METHODS:
                g = data[data.method.eq(method)]
                v = g[key].to_numpy(float)
                transformed = np.sign(v) * np.log1p(np.abs(v))
                hist, _, _ = np.histogram2d(g.week_index, transformed, bins=[yedges, xedges])
                histograms.append(hist)
                counted += int(hist.sum())
            if counted != len(data):
                raise AssertionError("Some outside points fell outside the heatmap bins.")
            maximum = max(2, max(float(h.max()) for h in histograms))
            fig, axes = plt.subplots(2, 3, figsize=(17, 10), sharex=True, sharey=True)
            im = None
            for ax, method, hist in zip(axes.ravel(), METHODS, histograms):
                im = ax.imshow(np.ma.masked_equal(hist, 0), origin="lower", aspect="auto",
                               extent=[-zmax, zmax, -.5, len(weeks) - .5], cmap=DENSITY_CMAP,
                               norm=LogNorm(vmin=1, vmax=maximum))
                ax.axvline(0, color="#D97732", linewidth=.8, linestyle=(0, (3, 3)), zorder=4)
                ticks = np.linspace(-zmax, zmax, 5)
                ax.set_xticks(ticks, [f"{np.sign(x) * np.expm1(abs(x)):.2g}" for x in ticks], fontsize=9)
                yticks = sorted(set(list(range(0, len(weeks), 3)) + [len(weeks) - 1]))
                ax.set_yticks(yticks, [weeks[i] for i in yticks], fontsize=8)
                ax.set_title(f"{LABELS[method]}\nOutside points: {int(hist.sum()):,}",
                             loc="left", fontsize=11)
            fig.subplots_adjust(left=.075, right=.89, bottom=.12, top=.87, hspace=.32, wspace=.30)
            cax = fig.add_axes([.915, .19, .015, .60])
            fig.colorbar(im, cax=cax, label="Outside-point count (log color scale)")
            fig.suptitle(f"Outside-point density | {level}% interval | {NORMS[norm]}\n"
                         f"{PERIOD_LABELS.get(case, case)}", fontsize=14, y=.98)
            fig.supxlabel("Signed normalized outside distance (symmetric log1p bins)", y=.035)
            fig.supylabel("Test week")
            path = destination / f"heatmap_{level}_{norm}.png"
            save_png(fig, path, dpi)
            files.append(entry(path, "outside_heatmap", case, level, norm))
            audit.append(dict(case=case, interval_level=level, normalization=norm,
                              source_outside_rows=len(data), scatter_points=plotted,
                              heatmap_count=counted, no_downsampling=True, no_tail_clipping=True))
    return files, audit


def table_images(metrics, destination, case, norms, levels, dpi):
    files = []
    for panel, title in PANELS.items():
        selected = metrics[metrics.table_id.eq(panel)]
        methods = [m for m in METHODS if selected.method.eq(m).any()]
        if not methods:
            continue
        indexed = selected.set_index(["method", "interval_level"], verify_integrity=True)
        for norm in norms:
            headers = ["Model", "SKUs", "Observed\npoints"]
            for level in levels:
                headers += [f"{level}%\nCoverage", f"{level}%\nMean width",
                            f"{level}%\nMean normalized\noutside distance"]
            rows = []
            for method in methods:
                first = indexed.loc[(method, levels[0])]
                row = [LABELS[method], f"{int(first.skus):,}", f"{int(first.observations):,}"]
                for level in levels:
                    a = indexed.loc[(method, level)]
                    row += [f"{a.coverage:.1%}", f"{a.mean_width:,.3f}",
                            f"{a[f'mean_normalized_{norm}']:.4f}"]
                rows.append(row)
            fig, ax = plt.subplots(figsize=(22, 4.7))
            ax.axis("off")
            widths = [.15, .05, .065] + [.06, .075, .11] * len(levels)
            widths = np.asarray(widths) / sum(widths)
            table = ax.table(cellText=rows, colLabels=headers, colWidths=widths,
                             cellLoc="right", bbox=[0, .14, 1, .64])
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            for (r, c), cell in table.get_celld().items():
                cell.set_edgecolor("#DCE5E9")
                cell.set_linewidth(.6)
                if r == 0:
                    cell.set_facecolor("#E9F1F5")
                    cell.set_text_props(weight="bold", color="#17475D")
                    cell.set_height(cell.get_height() * 1.5)
                elif r % 2 == 0:
                    cell.set_facecolor("#F6F9FB")
                if c == 0:
                    cell.set_text_props(ha="left")
            ax.text(0, .97, title, fontsize=17, weight="bold", color="#17475D", transform=ax.transAxes)
            ax.text(0, .88, f"{CASE_LABELS.get(case, case)} | Normalization: {NORMS[norm]}",
                    fontsize=11, transform=ax.transAxes)
            note = "Width is in sales units. Mean normalized outside distance includes covered observations as zero. Divisors are floored at 1 unit."
            ax.text(0, .035, note, fontsize=10, color="#526975", transform=ax.transAxes)
            path = destination / f"table_{panel}_{norm}.png"
            save_png(fig, path, dpi)
            files.append(entry(path, "table", case, norm=norm, panel=panel))
    return files


def overview_images(metrics, destination, case, norms, levels, dpi):
    files = []
    panels = ["All_SKUs", "Nonzero_groups"]
    specs = [("coverage", "Realized interval coverage"), ("mean_width", "Mean interval width (sales units)"),
             (None, "Mean normalized outside distance")]
    for norm in norms:
        fig, axes = plt.subplots(len(panels), 3, figsize=(22, 10))
        for row, panel in enumerate(panels):
            selected = metrics[metrics.table_id.eq(panel)]
            methods = [m for m in METHODS if selected.method.eq(m).any()]
            indexed = selected.set_index(["method", "interval_level"], verify_integrity=True)
            y = np.arange(len(methods))
            for col, (key, title) in enumerate(specs):
                ax = axes[row, col]
                if not len(methods):
                    ax.set_visible(False)
                    continue
                actual_key = key or f"mean_normalized_{norm}"
                for k, level in enumerate(levels):
                    v = [indexed.loc[(m, level), actual_key] for m in methods]
                    offset = (k - (len(levels) - 1) / 2) * .23
                    ax.barh(y + offset, v, height=.21, color=LEVEL_COLORS[k], label=f"{level}% interval")
                ax.set_yticks(y, [LABELS[m] for m in methods], fontsize=9)
                ax.invert_yaxis()
                ax.set_title(title, loc="left", fontsize=11)
                ax.spines[["top", "right"]].set_visible(False)
                ax.grid(axis="x", alpha=.15)
                ax.set_axisbelow(True)
                if key == "coverage":
                    ax.set_xlim(0, 1)
                    ax.xaxis.set_major_formatter(PercentFormatter(1))
                    for k, level in enumerate(levels):
                        ax.axvline(level / 100, color=LEVEL_COLORS[k], linestyle="--", linewidth=.8)
                    ax.text(0, 1.10, textwrap.fill(PANELS[panel], 52), transform=ax.transAxes,
                            fontsize=12, weight="bold", color="#17475D")
                if key is None:
                    ax.set_xlabel("Covered observations contribute zero")
        fig.suptitle(f"B model comparison | {NORMS[norm]} normalization\n{CASE_LABELS.get(case, case)}",
                     fontsize=17, y=.995)
        fig.legend(handles=[Patch(color=LEVEL_COLORS[k], label=f"{v}% interval") for k, v in enumerate(levels)],
                   loc="lower center", ncol=len(levels), frameon=False)
        fig.tight_layout(rect=(0, .035, 1, .95), h_pad=6, w_pad=2)
        path = destination / f"overall_comparison_{norm}.png"
        save_png(fig, path, dpi)
        files.append(entry(path, "overall_comparison", case, norm=norm))
    return files


def group_sizes(results, destination, dpi):
    path = results / "training_profiles.csv"
    if not path.is_file():
        return []
    data = pd.read_csv(path)
    counts = data.demand_group.value_counts().reindex(GROUPS, fill_value=0)
    fig, ax = plt.subplots(figsize=(11, 5))
    bars = ax.barh(GROUPS, counts, color=GROUP_COLORS)
    ax.bar_label(bars, labels=[f"{v:,}" for v in counts], padding=5)
    ax.set_xlim(0, max(1, counts.max()) * 1.15)
    ax.invert_yaxis()
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlabel("SKUs with complete training histories")
    ax.set_title("Training demand groups | Zero-sales share threshold: 60%", loc="left", fontsize=14)
    ax.text(0, -.20, "Groups and zero shares use training weeks only. At least 22 of 36 zero weeks meet the Zero rule.",
            transform=ax.transAxes, fontsize=9)
    fig.tight_layout()
    out = destination / "training_group_sizes.png"
    save_png(fig, out, dpi)
    return [entry(out, "training_group_sizes")]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, help="Existing results directory (default: beside this script)")
    parser.add_argument("--output", type=Path, help="PNG output directory (default: figures beside this script)")
    parser.add_argument("--case", choices=["all", "main_W42", "sensitivity_W43"], default="all")
    parser.add_argument("--normalizations", nargs="+", choices=list(NORMS), default=list(NORMS))
    parser.add_argument("--levels", nargs="+", type=int, choices=LEVELS, default=LEVELS)
    parser.add_argument("--dpi", type=int, default=160)
    args = parser.parse_args()
    if args.dpi < 72:
        parser.error("--dpi must be at least 72")
    results = find_results(args.results)
    output = args.output or Path(__file__).resolve().parent / "figures"
    output = output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    cases = [c for c in CASE_LABELS if (results / c / "metrics_long.csv").is_file()]
    if args.case != "all":
        if args.case not in cases:
            raise FileNotFoundError(f"Missing case: {results / args.case}")
        cases = [args.case]
    if not cases:
        raise FileNotFoundError(f"No supported Experiment B results in {results}")
    files, audits = [], []
    for case in cases:
        root = results / case
        metrics = pd.read_csv(root / "metrics_long.csv")
        points = pd.read_csv(root / "all_outside_points.csv.gz", dtype={"sku_id": str})
        weekly = root / "weekly_group_metrics.csv"
        if weekly.is_file():
            weeks = sorted(pd.read_csv(weekly, usecols=["iso_week"]).iso_week.unique())
        else:
            weeks = sorted(points.iso_week.unique())
        if not weeks:
            raise ValueError(f"No test-week information for {case}")
        target = output / case
        plotted, audit = plot_outside_points(points, weeks, target / "outside_points", case,
                                              args.normalizations, args.levels, args.dpi)
        files += plotted
        audits += audit
        files += table_images(metrics, target / "tables", case, args.normalizations, args.levels, args.dpi)
        files += overview_images(metrics, target / "comparisons", case, args.normalizations, args.levels, args.dpi)
    files += group_sizes(results, output, args.dpi)
    for row in files:
        row["path"] = Path(row["path"]).relative_to(output).as_posix()
    pd.DataFrame(files).to_csv(output / "figure_index.csv", index=False, encoding="utf-8-sig")
    (output / "plot_audit.json").write_text(json.dumps(audits, indent=2), encoding="utf-8")
    guide = """# Experiment B: PNG figure guide

All figures open directly in an image viewer. No HTML, browser, model fitting,
SciPy or LightGBM is needed to generate or view them.

- training_group_sizes.png: training group counts.
- main_W42/: primary test period, 2026-W25 to W42.
- sensitivity_W43/: adds the provisional W43; inspect separately from primary results.
- */comparisons/: overall coverage, width and mean normalized outside distance.
- */tables/: all overall and group tables as PNG images, with 70/80/90% intervals.
- */outside_points/scatter_*: every outside point; positive means above the upper
  bound, negative means below the lower bound. The y-axis uses a symmetric log scale.
- */outside_points/heatmap_*: week-by-distance counts of the same points. White
  means no points; color is logarithmic. Signed distances use log1p bins.
- figure_index.csv: filenames and corresponding case, interval and normalization.
- plot_audit.json: source, scatter and heatmap counts must agree exactly.

Start with main_W42/comparisons/overall_comparison_training_sd.png and
main_W42/outside_points/heatmap_90_training_sd.png.

Normalization divisors are floored at 1 sales unit: training sample SD,
median absolute non-zero training sales, or forecast interval width. Historical
scales use training data only. Table averages include covered points as zero.
Widths remain in sales units. The conditional mean over outside points is a
different metric and remains available in the original metrics_long.csv.

No outside point is sampled away or tail-clipped. One point is one SKU-week-
method-interval record; the same observed sale can be outside several forecasts.
Plots use every available prediction and retain returns at their original
negative values. Only five demand groups are used. Main comparison tables
use the four methods supporting all SKUs; supplemental count-eligible tables
compare all six methods on a common applicable population.
The Zero rule sets all models' intervals to [0,0] for training zero share >=60%.
Future non-zero Zero-group sales are retained and scored.
"""
    (output / "FIGURE_GUIDE.md").write_text(guide, encoding="utf-8")
    print(f"\nDone: {len(files)} PNG files. Open {output}", flush=True)
    print("All outside-point counts reconcile. No HTML or LightGBM was used.", flush=True)


if __name__ == "__main__":
    main()
