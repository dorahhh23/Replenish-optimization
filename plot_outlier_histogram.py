#!/usr/bin/env python3
"""Histogram of every outside-interval point, with signed distances retained.

This is a count histogram in symmetric log1p bins, not a probability density
in linear sales units. No points are sampled, winsorized or clipped. Saved
predictions are reused; LightGBM is not required to draw the histogram.
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

from plot_signed_scatter import METHODS, LABELS, GROUPS, COLORS, NORMS, PERIODS


def transform(values, scale=.1):
    values = np.asarray(values, dtype=float)
    return np.sign(values) * np.log1p(np.abs(values) / scale)


def inverse_transform(values, scale=.1):
    values = np.asarray(values, dtype=float)
    return np.sign(values) * np.expm1(np.abs(values)) * scale


def bin_edges(values, bins, scale=.1):
    """Use shared bins across models, with an explicit boundary at zero."""
    if bins < 10:
        raise ValueError("Use at least 10 bins.")
    z = transform(values, scale)
    # Move outer edges outward in display space so even the most extreme
    # point lies strictly inside the full range. Zero separates the tails.
    low = min(0., float(z.min()))
    high = max(0., float(z.max()))
    low = low * 1.015 if low < 0 else 0.
    high = high * 1.015 if high > 0 else 0.
    if low < 0 < high:
        left_bins = max(1, min(bins - 1, round(bins * -low / (high - low))))
        return np.r_[np.linspace(low, 0., left_bins + 1),
                     np.linspace(0., high, bins - left_bins + 1)[1:]]
    if high > 0:
        return np.linspace(0., high, bins + 1)
    if low < 0:
        return np.linspace(low, 0., bins + 1)
    raise ValueError("Outside distances cannot all be zero.")


def tick_values(edges, scale):
    candidates = np.array([-100000., -10000., -1000., -100., -10., -1.,
                           0., 1., 10., 100., 1000., 10000., 100000.])
    ticks = transform(candidates, scale)
    mask = (ticks >= edges[0]) & (ticks <= edges[-1])
    return ticks[mask], [f"{v:g}" for v in candidates[mask]]


def make_histogram(data, output, case, level, normalization, bins=72, dpi=160):
    key = f"signed_normalized_{normalization}"
    required = {"method", "demand_group", "sku_id", "actual_sales", key}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Missing outside-point columns: {sorted(missing)}")
    if set(data.method) - set(METHODS) or set(data.demand_group) - set(GROUPS):
        raise ValueError("Outside-point data contain unknown methods or groups.")
    values = data[key].to_numpy(float)
    if not len(values) or not np.isfinite(values).all() or (values == 0).any():
        raise ValueError("Outside distances must be finite, non-zero and non-empty.")
    scale = .1
    edges = bin_edges(values, bins, scale)
    raw_edges = inverse_transform(edges, scale)
    z = transform(values, scale)
    ticks, tick_labels = tick_values(edges, scale)
    fig, axes = plt.subplots(2, 3, figsize=(17, 10), sharex=True, sharey=True)
    records, summaries, method_counts = [], [], {}
    counted_total = 0
    for ax, method in zip(axes.ravel(), METHODS):
        method_mask = data.method.eq(method).to_numpy()
        local = values[method_mask]
        total = int(method_mask.sum())
        method_counts[method] = total
        bottom = np.zeros(len(edges) - 1, dtype=int)
        for group, color in zip(GROUPS, COLORS):
            mask = method_mask & data.demand_group.eq(group).to_numpy()
            counts, _ = np.histogram(z[mask], bins=edges)
            if int(counts.sum()) != int(mask.sum()):
                raise AssertionError("Histogram dropped an outside point.")
            counted_total += int(counts.sum())
            ax.bar(edges[:-1], counts, width=np.diff(edges), align="edge",
                   bottom=bottom, color=color, edgecolor="none", linewidth=0)
            bottom += counts
            for i, count in enumerate(counts):
                records.append(dict(case=case, interval_level=level,
                    normalization=normalization, method=method, demand_group=group,
                    bin_index=i, transformed_left=edges[i], transformed_right=edges[i + 1],
                    signed_distance_left=raw_edges[i], signed_distance_right=raw_edges[i + 1],
                    count=int(count)))
        ax.axvline(0, color="#D97732", linewidth=.8, linestyle=(0, (3, 3)), zorder=5)
        below, above = int((local < 0).sum()), int((local > 0).sum())
        ax.set_title(f"{LABELS[method]}\nOutside points: {total:,} | Below: {below:,} | Above: {above:,}",
                     loc="left", fontsize=10.5, pad=10)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.16)
        ax.set_axisbelow(True)
        ax.set_xlim(edges[0], edges[-1])
        ax.set_xticks(ticks, tick_labels, fontsize=9)
        ax.tick_params(axis="y", labelsize=9)
        ax.ticklabel_format(axis="y", style="plain")
        summary = dict(case=case, interval_level=level, normalization=normalization,
                       method=method, outside_points=total,
                       below_lower_bound=below, above_upper_bound=above,
                       below_share=below / total if total else np.nan,
                       negative_actual_outside_points=int(data.loc[method_mask, "actual_sales"].lt(0).sum()))
        for name, number in [("signed_p05", .05), ("signed_median", .5), ("signed_p95", .95)]:
            summary[name] = float(np.quantile(local, number)) if total else np.nan
        for name, number in [("absolute_median", .5), ("absolute_p90", .9), ("absolute_p95", .95)]:
            summary[name] = float(np.quantile(np.abs(local), number)) if total else np.nan
        summary["minimum_signed_distance"] = float(local.min()) if total else np.nan
        summary["maximum_signed_distance"] = float(local.max()) if total else np.nan
        summaries.append(summary)
    if counted_total != len(data):
        raise AssertionError("Source and histogram counts do not match.")
    fig.suptitle(f"Outside-point distribution | {level}% interval | {NORMS[normalization]}\n{PERIODS[case]}",
                 fontsize=14, y=.995)
    fig.supylabel("Outside points per transformed bin", x=.008)
    fig.supxlabel("Signed normalized outside distance (symmetric log1p scale)", y=.097)
    fig.text(.5, .064,
             "Left of 0: below interval | Right of 0: above interval | All outside points retained | Zero threshold: 60%",
             ha="center", fontsize=9, color="#4D5962")
    fig.text(.5, .041, "Each panel uses available forecasts; model sample sizes may differ. Colors show the five demand groups.",
             ha="center", fontsize=8.5, color="#4D5962")
    fig.legend(handles=[Patch(color=c, label=g) for c, g in zip(COLORS, GROUPS)],
               loc="lower center", ncol=5, frameon=False, fontsize=10)
    fig.tight_layout(rect=(.025, .13, 1, .93), h_pad=2.5)
    output.mkdir(parents=True, exist_ok=True)
    stem = output / f"histogram_{level}_{normalization}"
    image = stem.with_suffix(".png")
    temporary = image.with_suffix(".tmp.png")
    fig.savefig(temporary, format="png", dpi=dpi, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    temporary.replace(image)
    pd.DataFrame(records).to_csv(stem.with_name(stem.name + "_counts.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(summaries).to_csv(stem.with_name(stem.name + "_summary.csv"), index=False, encoding="utf-8-sig")
    audit = dict(case=case, interval_level=level, normalization=normalization,
                 source_rows=len(data), histogram_count=counted_total, model_point_counts=method_counts,
                 bins=len(edges) - 1, display_transform="sign(x) * log1p(abs(x) / 0.1)",
                 minimum_signed_distance=float(values.min()), maximum_signed_distance=float(values.max()),
                 shared_bin_edges=True, zero_is_bin_boundary=True, no_downsampling=True,
                 no_tail_clipping=True, signed_values_unchanged=True,
                 negative_actual_outside_points=int(data.actual_sales.lt(0).sum()))
    stem.with_suffix(".audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(f"Saved: {image} ({counted_total:,} outside points)", flush=True)
    return audit


def main():
    home = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, default=home / "results")
    parser.add_argument("--output", type=Path, default=home / "figures_histogram")
    parser.add_argument("--case", choices=["all", *PERIODS], default="all")
    parser.add_argument("--levels", type=int, nargs="+", choices=[70, 80, 90], default=[80])
    parser.add_argument("--normalizations", nargs="+", choices=list(NORMS), default=["nonzero_median"])
    parser.add_argument("--bins", type=int, default=72)
    parser.add_argument("--dpi", type=int, default=160)
    args = parser.parse_args()
    if args.bins < 10 or args.dpi < 72:
        parser.error("--bins must be at least 10 and --dpi at least 72")
    cases = [case for case in PERIODS if (args.results / case / "all_outside_points.csv.gz").is_file()]
    if args.case != "all":
        if args.case not in cases:
            parser.error(f"Missing outside-point data for {args.case}. Run the experiment first.")
        cases = [args.case]
    if not cases:
        parser.error("No results found. Keep this script beside results/, or use --results PATH.")
    for case in cases:
        points = pd.read_csv(args.results / case / "all_outside_points.csv.gz", dtype={"sku_id": str})
        if points.duplicated(["sku_id", "method", "iso_week", "interval_level"]).any():
            raise ValueError("Duplicate outside-point records found.")
        for level in args.levels:
            data = points.loc[points.interval_level.eq(level)]
            for normalization in args.normalizations:
                make_histogram(data, args.output / case, case, level, normalization, args.bins, args.dpi)


if __name__ == "__main__":
    main()
