#!/usr/bin/env python3
"""Keep signed sales; export horizontal interval-width bars, coverage dots and deviation tables.

Run beside the existing results/ folder: python make_combined_charts.py
No model retraining, HTML, SciPy or LightGBM is required.
"""
from pathlib import Path
import argparse
import hashlib
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter

LEVELS = {70: ("q15", "q85"), 80: ("q10", "q90"), 90: ("q05", "q95")}
ALL_METHODS = ["Empirical", "Recent13", "DampedTrendResidual", "LightGBM", "Poisson", "NegativeBinomial"]
SIGNED_METHODS = ALL_METHODS[:4]
LABELS = {"Empirical": "Historical empirical", "Recent13": "Recent 13-week empirical",
          "DampedTrendResidual": "Damped trend + residuals", "LightGBM": "Grouped LightGBM",
          "Poisson": "Poisson", "NegativeBinomial": "Negative binomial"}
SHORT = {"Empirical": "Empirical", "Recent13": "Recent 13-week", "DampedTrendResidual": "Damped trend",
         "LightGBM": "LightGBM", "Poisson": "Poisson", "NegativeBinomial": "Neg. binomial"}
GROUPS = ["Zero", "Smooth", "Erratic", "Intermittent", "Lumpy"]
TITLES = {"All_SKUs": "All SKUs, including Zero-group",
          "Nonzero_groups": "All SKUs, excluding Zero-group",
          **{g: f"{g} group" for g in GROUPS}}
CASES = {"main_W42": "Main B: 2026-W25 to W42",
         "sensitivity_W43": "Sensitivity B: 2026-W25 to provisional W43"}
PERIODS = {"main_W42": "training: 2025-W41 to 2026-W24, testing: 2026-W25 to 2026-W42",
           "sensitivity_W43": "training: 2025-W41 to 2026-W24, testing: 2026-W25 to 2026-W43"}
BAR = "#397E9C"
DOT = "#C05B22"
INTERVAL_COLORS = ["#98BCCB", "#488BA5", "#174F6C"]


def write_csv(frame, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def save_png(fig, path, dpi):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.png")
    fig.savefig(tmp, format="png", dpi=dpi, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    tmp.replace(path)
    print("Saved:", path, flush=True)


def summarize(predictions, profiles, case):
    # Training completeness is the only SKU eligibility rule here. Negative
    # training or test values never remove a SKU or a scored observation.
    scored = predictions[predictions.actual_sales.notna()].copy()
    wanted = set(profiles.sku_id)
    if set(scored[scored.method.eq("LightGBM")].sku_id) != wanted:
        raise ValueError("Saved predictions do not cover every complete-training SKU.")
    if scored.duplicated(["sku_id", "iso_week", "method"]).any():
        raise ValueError("Duplicate SKU-week-method predictions.")
    grouped = {g: profiles.loc[profiles.demand_group.eq(g), "sku_id"] for g in GROUPS}
    panels = [("All_SKUs", profiles.sku_id, SIGNED_METHODS),
              ("Nonzero_groups", profiles.loc[profiles.demand_group.ne("Zero"), "sku_id"], SIGNED_METHODS)]
    panels += [(g, grouped[g], SIGNED_METHODS) for g in GROUPS]
    rows, audits = [], []
    for panel, ids, methods in panels:
        ids = set(ids)
        if not ids:
            continue
        reference = scored[scored.method.eq("LightGBM") & scored.sku_id.isin(ids)]
        reference = reference.set_index(["sku_id", "iso_week"]).actual_sales.sort_index()
        for method in methods:
            d = scored[scored.method.eq(method) & scored.sku_id.isin(ids)]
            actuals = d.set_index(["sku_id", "iso_week"]).actual_sales.sort_index()
            if not reference.equals(actuals):
                raise ValueError(f"{panel}/{method}: models do not use identical signed actuals.")
            q = d[list(dict.fromkeys(c for pair in LEVELS.values() for c in pair))].to_numpy(float)
            if not np.isfinite(q).all():
                raise ValueError(f"Non-finite forecast quantiles: {panel}/{method}")
            divisor = np.maximum(d.nonzero_abs_median.to_numpy(float), 1.)
            if not np.isfinite(divisor).all():
                raise ValueError("Invalid training median scales.")
            if not np.allclose(divisor, d.nonzero_median_divisor):
                raise ValueError("Stored normalization disagrees with the training non-zero median.")
            y = d.actual_sales.to_numpy(float)
            for level, (lo, hi) in LEVELS.items():
                lower, upper = d[lo].to_numpy(float), d[hi].to_numpy(float)
                if np.any(lower > upper):
                    raise ValueError("Crossed interval bounds.")
                distance = np.maximum.reduce([lower - y, y - upper, np.zeros(len(y))])
                normalized = distance / divisor
                outside = distance > 0
                rows.append(dict(case=case, table_id=panel, method=method,
                    skus=d.sku_id.nunique(), observations=len(d), negative_actuals=int((y < 0).sum()),
                    interval_level=level, coverage=float((~outside).mean()), mean_width=float((upper - lower).mean()),
                    outside_points=int(outside.sum()), mean_normalized_outside_distance=float(normalized.mean()),
                    outside_only_mean_normalized_distance=float(normalized[outside].mean()) if outside.any() else np.nan))
            audits.append(dict(case=case, table_id=panel, method=method, skus=d.sku_id.nunique(),
                observations=len(d), negative_actuals_retained=int((y < 0).sum()), same_actuals_as_reference=True))
    return pd.DataFrame(rows), audits


def make_tables(metrics, output):
    width_tables, deviation_tables = [], []
    for panel, part in metrics.groupby("table_id", sort=False):
        wr, dr = [], []
        for method in ALL_METHODS:
            d = part[part.method.eq(method)]
            if d.empty:
                continue
            first = d.iloc[0]
            base = {"Model": LABELS[method], "SKUs": int(first.skus), "Observed points": int(first.observations)}
            w, v = dict(base), dict(base)
            for level in LEVELS:
                row = d[d.interval_level.eq(level)].iloc[0]
                w[f"{level}% Coverage"] = row.coverage
                w[f"{level}% Mean Width"] = row.mean_width
                v[f"{level}% Mean Normalized Outside Distance"] = row.mean_normalized_outside_distance
            widths, deviations = w, v
            wr.append(widths)
            dr.append(deviations)
        wframe, dframe = pd.DataFrame(wr), pd.DataFrame(dr)
        write_csv(wframe, output / "coverage_width_tables" / f"{panel}.csv")
        write_csv(dframe, output / "deviation_tables" / f"{panel}.csv")
        for frame, collection in [(wframe, width_tables), (dframe, deviation_tables)]:
            f = frame.copy()
            f.insert(0, "Table", panel)
            f.insert(0, "Case", part.case.iloc[0])
            collection.append(f)
    return pd.concat(width_tables, ignore_index=True), pd.concat(deviation_tables, ignore_index=True)


def plot_panel(ax, metrics, panel):
    part = metrics[metrics.table_id.eq(panel)]
    d = part.set_index(["method", "interval_level"], verify_integrity=True)
    methods = [m for m in ALL_METHODS if m in part.method.unique()]
    y = np.arange(len(methods))
    maximum = max(1., float(part.mean_width.max()) * 1.12)
    for k, (level, color) in enumerate(zip(LEVELS, INTERVAL_COLORS)):
        positions = y + (k - 1) * .23
        ax.barh(positions, [d.loc[(m, level), "mean_width"] for m in methods],
                height=.20, color=color, zorder=2)
    ax.set_xlim(0, maximum)
    ax.set_ylim(len(methods) - .45, -.55)
    ax.set_yticks(y, [LABELS[m] for m in methods], fontsize=11)
    ax.set_xlabel("Mean interval width (sales units) — bars / bottom axis", color=BAR, fontsize=11)
    ax.tick_params(axis="x", colors=BAR)
    ax.tick_params(axis="y", length=0)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="x", alpha=.14, zorder=0)
    top = ax.twiny()
    top.set_xlim(0, 1.08)
    top.set_xticks(np.arange(0, 1.01, .2))
    top.xaxis.set_major_formatter(PercentFormatter(1))
    top.set_xlabel("Observed interval coverage — dots / top axis", color=DOT, fontsize=11, labelpad=9)
    top.tick_params(axis="x", colors=DOT)
    top.spines[["right", "left", "bottom"]].set_visible(False)
    top.spines["top"].set_color("#D8D8D8")
    for k, (level, color) in enumerate(zip(LEVELS, INTERVAL_COLORS)):
        positions = y + (k - 1) * .23
        values = [d.loc[(m, level), "coverage"] for m in methods]
        top.axvline(level / 100, color=color, linestyle="--", linewidth=.8, alpha=.4, zorder=1)
        top.scatter(values, positions, s=40, color=DOT, edgecolor="white", linewidth=.7, zorder=5)
        for v, position in zip(values, positions):
            top.annotate(f"{v:.1%}", (v, position), xytext=(7, 0), textcoords="offset points",
                         ha="left", va="center", fontsize=9, color=DOT,
                         bbox=dict(facecolor="white", edgecolor="none", alpha=.85, pad=.6))
    first = part.iloc[0]
    ax.text(0, 1.34, TITLES[panel], transform=ax.transAxes,
            fontsize=13, weight="bold", color="#17475D")
    ax.text(0, 1.24, f"{int(first.skus):,} SKUs | {int(first.observations):,} observed points per method | Returns retained as negative sales",
            transform=ax.transAxes, fontsize=9, color="#566B75")


def comparison_image(metrics, panels, path, case, dpi):
    fig, axes = plt.subplots(len(panels), 1, figsize=(13, 5.9 * len(panels)), squeeze=False)
    for row, panel in enumerate(panels):
        plot_panel(axes[row, 0], metrics, panel)
    fig.suptitle(f"Interval width and coverage (Zero threshold: 60%)\n{PERIODS[case]}", fontsize=14, y=.985)
    handles = [Patch(color=c, label=f"{level}% interval") for level, c in zip(LEVELS, INTERVAL_COLORS)]
    handles += [Line2D([], [], marker="o", linestyle="None", color=DOT, label="Dot: observed coverage (top axis)")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=10,
               bbox_to_anchor=(.55, .015))
    fig.subplots_adjust(left=.25, right=.96, bottom=.24 if len(panels) == 1 else .11,
                        top=.68 if len(panels) == 1 else .81, hspace=.98)
    save_png(fig, path, dpi)


def deviation_image(metrics, panel, path, case, dpi):
    part = metrics[metrics.table_id.eq(panel)]
    rows = []
    for method in ALL_METHODS:
        d = part[part.method.eq(method)].set_index("interval_level")
        if d.empty:
            continue
        first = d.iloc[0]
        rows.append([LABELS[method], f"{int(first.skus):,}", f"{int(first.observations):,}"] +
                    [f"{d.loc[level, 'mean_normalized_outside_distance']:.4f}" for level in LEVELS])
    fig, ax = plt.subplots(figsize=(13, 4.2))
    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=["Model", "SKUs", "Observed\npoints", "70% interval", "80% interval", "90% interval"],
                     colWidths=[.29, .10, .15, .15, .15, .16], cellLoc="right", bbox=[0, .21, 1, .55])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#DCE5E9")
        cell.set_linewidth(.6)
        if r == 0:
            cell.set_facecolor("#E9F1F5")
            cell.set_text_props(weight="bold", color="#17475D")
        elif r % 2 == 0:
            cell.set_facecolor("#F5F8FA")
        if c == 0:
            cell.set_text_props(ha="left")
    ax.text(0, .97, f"Mean normalized outside distance | {TITLES[panel]}", transform=ax.transAxes,
            fontsize=13, color="#17475D", weight="bold")
    ax.text(0, .86, CASES[case], transform=ax.transAxes, fontsize=11)
    ax.text(0, .09, "Per point: outside distance / max(median absolute non-zero training sales, 1 unit).",
            transform=ax.transAxes, fontsize=10)
    ax.text(0, .01, "Mean uses all scored observations; covered points contribute zero. Lower is better.",
            transform=ax.transAxes, fontsize=10)
    save_png(fig, path, dpi)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    home = Path(__file__).resolve().parent
    p.add_argument("--results", type=Path, default=home / "results")
    p.add_argument("--output", type=Path, default=home / "revised_charts")
    p.add_argument("--case", choices=["all", *CASES], default="all")
    p.add_argument("--dpi", type=int, default=170)
    args = p.parse_args()
    if args.dpi < 72:
        p.error("--dpi must be at least 72")
    root, output = args.results.resolve(), args.output.resolve()
    profiles = pd.read_csv(root / "training_profiles.csv", dtype={"sku_id": str})
    if profiles.sku_id.duplicated().any():
        raise ValueError("Duplicate training profiles.")
    cases = [c for c in CASES if (root / c / "predictions.csv.gz").is_file()]
    if args.case != "all":
        if args.case not in cases:
            raise FileNotFoundError(f"Missing predictions for {args.case}")
        cases = [args.case]
    if not cases:
        raise FileNotFoundError(f"No prediction files found under {root}")
    all_widths, all_distances, all_metrics, audits, hashes = [], [], [], [], {}
    for case in cases:
        source = root / case / "predictions.csv.gz"
        hashes[case] = hashlib.sha256(source.read_bytes()).hexdigest()
        pred = pd.read_csv(source, dtype={"sku_id": str})
        metrics, audit = summarize(pred, profiles, case)
        target = output / case
        w, d = make_tables(metrics, target)
        all_widths.append(w)
        all_distances.append(d)
        all_metrics.append(metrics)
        audits += audit
        write_csv(metrics, target / "metrics_long.csv")
        comparison_image(metrics, ["All_SKUs", "Nonzero_groups"], target / "width_coverage_overall.png", case, args.dpi)
        for group in GROUPS:
            if metrics.table_id.eq(group).any():
                comparison_image(metrics, [group], target / f"width_coverage_{group}.png", case, args.dpi)
        for panel in metrics.table_id.unique():
            deviation_image(metrics, panel, target / "deviation_tables" / f"{panel}.png", case, args.dpi)
    write_csv(pd.concat(all_widths, ignore_index=True), output / "B_coverage_width.csv")
    write_csv(pd.concat(all_distances, ignore_index=True), output / "B_deviation_median.csv")
    write_csv(pd.concat(all_metrics, ignore_index=True), output / "B_metrics_long.csv")
    write_csv(pd.DataFrame(audits), output / "signed_data_audit.csv")
    settings = dict(negative_sales_policy="Negative sales are returns; retain original values without adjustment",
        zero_threshold=.60,
        overall_methods=SIGNED_METHODS, count_models="Supplemental CountEligible result tables compare six methods on applicable common populations",
        grouping="Five demand groups only; returns do not create a separate group; Zero>=60%",
        denominator="max(median(abs(nonzero training sales)) per SKU, 1 unit)",
        mean="All scored points; covered points contribute zero", intervals=list(LEVELS),
        complete_training_skus=len(profiles), input_prediction_sha256=hashes)
    (output / "settings.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
    print(f"Done: {len(profiles):,} SKUs retained; outputs in {output}", flush=True)


if __name__ == "__main__":
    main()
