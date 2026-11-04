#!/usr/bin/env python3
"""Plot every outside point, retaining signed values and using actual y limits.

Run beside results/: python plot_signed_scatter.py
No training, HTML, SciPy or LightGBM is required.
"""
from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

METHODS = ["Empirical", "Recent13", "DampedTrendResidual", "LightGBM", "Poisson", "NegativeBinomial"]
LABELS = {"Empirical": "Historical empirical", "Recent13": "Recent 13-week empirical",
          "DampedTrendResidual": "Damped trend + residuals", "LightGBM": "Grouped LightGBM",
          "Poisson": "Poisson", "NegativeBinomial": "Negative binomial"}
GROUPS = ["Smooth", "Erratic", "Intermittent", "Lumpy", "Zero"]
COLORS = ["#26758F", "#C2762D", "#449B85", "#8864AD", "#8D99A3"]
NORMS = {"training_sd": "Training standard deviation",
         "nonzero_median": "Median absolute non-zero training sales",
         "interval_width": "Predicted interval width"}
PERIODS = {"main_W42": "training: 2025-W41 to 2026-W24, testing: 2026-W25 to 2026-W42",
           "sensitivity_W43": "training: 2025-W41 to 2026-W24, testing: 2026-W25 to 2026-W43"}


def plot_case(points, weeks, destination, case, levels, norms, dpi):
    destination.mkdir(parents=True, exist_ok=True)
    if set(points.method) - set(METHODS) or set(points.demand_group) - set(GROUPS):
        raise ValueError("Unrecognized methods or groups in outside-point data.")
    # Display jitter affects only the horizontal coordinate. Signed y-values
    # are read directly from the saved result columns without alteration.
    lookup = {w: i for i, w in enumerate(weeks)}
    index = points.iso_week.map(lookup)
    if index.isna().any():
        raise ValueError("Some outside points have unknown test weeks.")
    hashed = pd.util.hash_pandas_object(points.sku_id.astype(str) + "|" + points.iso_week,
                                       index=False).to_numpy(np.uint64)
    x = index.to_numpy(float) + (hashed % 10000).astype(float) / 10000 * .42 - .21
    audits = []
    for level in levels:
        select = points.interval_level.eq(level).to_numpy()
        d = points.loc[select]
        if d.empty:
            raise ValueError(f"No outside points for interval {level} in {case}.")
        display_x = x[select]
        for norm in norms:
            key = f"signed_normalized_{norm}"
            y = d[key].to_numpy(float)
            if not np.isfinite(y).all() or (y == 0).any():
                raise ValueError("Outside-point signed distances must be finite and non-zero.")
            minimum, maximum = float(y.min()), float(y.max())
            # A shared range across methods, with independent positive and
            # negative limits: no symmetric empty extension and no clipping.
            low = min(0., minimum) - max(.02, abs(min(0., minimum)) * .15)
            high = max(0., maximum) + max(.02, max(0., maximum) * .15)
            assert low < minimum and high > maximum
            fig, axes = plt.subplots(2, 3, figsize=(17, 10), sharex=True, sharey=True)
            plotted = 0
            counts = {}
            for ax, method in zip(axes.ravel(), METHODS):
                method_mask = d.method.eq(method).to_numpy()
                counts[method] = int(method_mask.sum())
                for group, color in zip(GROUPS, COLORS):
                    mask = method_mask & d.demand_group.eq(group).to_numpy()
                    ax.scatter(display_x[mask], y[mask], s=4, alpha=.23,
                               color=color, edgecolors="none", rasterized=True)
                    plotted += int(mask.sum())
                ax.set_yscale("symlog", linthresh=.1)
                ax.set_ylim(low, high)
                ax.axhline(0, color="#55636C", linewidth=.7)
                ax.set_title(f"{LABELS[method]}\nOutside points: {counts[method]:,}", loc="left", fontsize=11)
                ax.spines[["top", "right"]].set_visible(False)
                ax.grid(axis="y", alpha=.14)
                ticks = sorted(set(list(range(0, len(weeks), 4)) + [len(weeks) - 1]))
                ax.set_xticks(ticks, [weeks[i] for i in ticks], rotation=35, fontsize=8)
                ax.set_xlim(-.6, len(weeks) - .4)
            assert plotted == len(d)
            fig.suptitle(f"Every outside point | {level}% interval | {NORMS[norm]}\n{PERIODS[case]}",
                         fontsize=14, y=.995)
            fig.supxlabel("Test week; deterministic horizontal jitter separates overlaps", y=.066)
            fig.supylabel("Signed normalized outside distance (symlog scale)")
            fig.legend(handles=[Patch(color=c, label=g) for c, g in zip(COLORS, GROUPS)],
                       loc="lower center", ncol=6, frameon=False)
            fig.tight_layout(rect=(.02, .10, 1, .94))
            path = destination / f"scatter_{level}_{norm}_asymmetric.png"
            temporary = path.with_suffix(".tmp.png")
            fig.savefig(temporary, format="png", dpi=dpi, facecolor="white", bbox_inches="tight")
            plt.close(fig)
            temporary.replace(path)
            audits.append(dict(case=case, interval_level=level, normalization=norm, source_rows=len(d),
                plotted_rows=plotted, negative_distance_points=int((y < 0).sum()),
                positive_distance_points=int((y > 0).sum()), negative_actual_records=int(d.actual_sales.lt(0).sum()),
                actual_minimum_signed_distance=minimum, actual_maximum_signed_distance=maximum,
                y_axis_lower=low, y_axis_upper=high, model_point_counts=counts,
                signed_values_unchanged=True, no_downsampling=True, no_tail_clipping=True))
            print("Saved:", path, flush=True)
    return audits


def main():
    home = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", type=Path, default=home / "results")
    p.add_argument("--output", type=Path, default=home / "figures_signed_scatter")
    p.add_argument("--case", choices=["all", *PERIODS], default="all")
    p.add_argument("--levels", type=int, nargs="+", choices=[70, 80, 90], default=[70, 80, 90])
    p.add_argument("--normalizations", nargs="+", choices=list(NORMS), default=list(NORMS))
    p.add_argument("--dpi", type=int, default=160)
    args = p.parse_args()
    if args.dpi < 72:
        p.error("--dpi must be at least 72")
    cases = [c for c in PERIODS if (args.results / c / "all_outside_points.csv.gz").is_file()]
    if args.case != "all":
        if args.case not in cases:
            raise FileNotFoundError(f"Missing outside-point data for {args.case}")
        cases = [args.case]
    if not cases:
        raise FileNotFoundError("Put this script beside results/, or set --results PATH.")
    audits = []
    for case in cases:
        root = args.results / case
        points = pd.read_csv(root / "all_outside_points.csv.gz", dtype={"sku_id": str})
        weeks = sorted(pd.read_csv(root / "weekly_group_metrics.csv", usecols=["iso_week"]).iso_week.unique())
        audits += plot_case(points, weeks, args.output / case, case, args.levels, args.normalizations, args.dpi)
    (args.output / "scatter_audit.json").write_text(json.dumps(audits, indent=2), encoding="utf-8")
    print(f"Done: {len(audits)} scatter plots; all signed values retained.", flush=True)


if __name__ == "__main__":
    main()
