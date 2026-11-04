# Zero threshold experiment: Zero SKUs excluded

Code updated on November 4, 2026. This revision has **not been executed** by
the assistant, as requested. Run the commands below to generate new results.
Existing validation reports and figures under `results/` describe the earlier
Zero-prediction policy and do not validate this revision.

## Run on Windows

Open PowerShell in this folder:

```powershell
cd "C:\Users\yizhe.hu\Documents\ChatGPT\毕设\Bexperiment\test_zero_threshold"
& 'C:\Users\Public\anaconda3\python.exe' .\run_experiment.py --thresholds 50 60 70 80
& 'C:\Users\Public\anaconda3\python.exe' .\validate_results.py
```

Or double-click `run_windows.bat` for the full experiment, including figures.
The scripts use the existing environment and local `_vendor` LightGBM 4.6.0.
Paths support spaces and Chinese. Set `ZERO_PYTHON` to select another Python
executable for the Windows launchers.

`config.json` controls the thresholds, input, output, model rounds, threads,
seed, W43 sensitivity, and PNG resolution. The default output is
`results_zero_excluded/`. Relative input/output paths resolve against the config
file. `--output results_another_run` writes to a different subfolder here.
Use a new folder when changing the set of thresholds to keep run outputs separate.

## Classification and evaluation policy

- Training: 2025-W41 through 2026-W24 (36 weeks). Main testing: 2026-W25
  through W42 (18 weeks). W43 sensitivity includes W25 through W43 using the
  same fitted models; W43 may be incomplete.
- A SKU is Zero when the proportion of training weeks with sales **equal to
  zero** is at least the threshold. At 50%, 60%, 70%, 80%, the minimum zero
  week counts are 18, 22, 26, 29. Only training data determine membership.
- Zero SKUs are exported to `zerolist.csv` and excluded **before** fitting and
  prediction. They receive no point forecasts or intervals. They do not enter
  any evaluation denominator, coverage, width, deviation, or outlier counts.
- Every threshold models and evaluates its **own non-Zero SKU population**.
  There are no fixed cross-threshold cohorts or all-SKU evaluation tables.
  Expected non-Zero counts for the supplied data are 988, 1,206, 1,485, 1,754;
  these are expectations from the previous classification, not a new run result.
- The evaluated groups are Smooth, Erratic, Intermittent, and Lumpy. The original
  positive-demand ADI 1.32 and CV-squared 0.49 rules are retained.
- Negative sales and large orders in the eligible non-Zero sample keep their
  original values. Missing values are never zero-filled. Incomplete training
  histories are listed separately; missing test actuals are unscored.
- Empirical, Recent13, DampedTrendResidual, and grouped LightGBM use every
  eligible non-Zero SKU. All four methods use the same observations **within
  each threshold**. Each threshold freshly fits its own grouped LightGBM models
  with the original settings, chronological validation, and final training refit.
- Poisson and NegativeBinomial are supplemental: all six methods are compared
  on the current threshold's non-Zero SKUs with nonnegative integer training
  histories. This population can change across thresholds. Test-period returns
  are retained. Count eligibility never removes returns from primary evaluation.
- 70%, 80%, and 90% coverage values are proportions. The **primary width metric**
  is `mean((upper-lower) / max(median(abs(nonzero training sales)), 1))`.
  The **secondary width metric** is
  `mean((upper-lower) / max(sample standard deviation of training sales, 1))`.
  The sample standard deviation uses all 36 original signed training values,
  including zeros and returns, with `ddof=1`. Both denominators use training data
  only and are shared across methods and interval levels for the same SKU.
- Normalize each scored SKU-week first, then average across all valid test points,
  using the same observation weighting as coverage and outside distance. Do not
  divide an aggregate mean width by an aggregate scale. A normalized full width
  of 2 represents two scale units, not a plus/minus two-standard-deviation interval.
  The denominator floor is one sales unit. Tables count scored SKUs whose raw
  median or standard deviation is below this floor. Empty cohorts have missing
  metrics rather than zero scores.
- Per-point outside distance is `max(lower-actual, actual-upper, 0)`.
  Divide by the SKU's `max(median(abs(nonzero training sales)), 1)` and average
  all valid test points, including zeros contributed by covered points. No WIS.

## New output layout

All files below are created under `results_zero_excluded/` when you run:

- `threshold_summary.csv`: Zero list size, modeled SKU count, count eligibility,
  and retained negative training counts at each threshold.
- `group_counts.csv`: counts of the four modeled groups.
- `zerolist_all_thresholds.csv`: combined threshold-specific Zero lists.
- `overall_comparison.csv`: overall evaluation on each threshold's own non-Zero sample.
- `comparison_all_tables.csv`, `metrics_long.csv`: overall, four group, and
  supplemental count-eligible comparisons, with sample sizes. Main tables contain
  `Mean Median-Normalized Width` (primary) and `Mean SD-Normalized Width`
  (secondary) at each interval level, alongside coverage and outside distance.
- `absolute_width_reference.csv`: mean absolute width in sales units, for reference.
  The `mean_width` column in `metrics_long.csv` also retains this raw value for
  audit purposes; the two `mean_normalized_width_*` columns drive the main charts.
- `main_W42/` and `sensitivity_W43/`: tables, separate deviation tables, and
  width-bar/coverage-dot and deviation PNGs comparing thresholds. For each panel,
  `*_width_coverage.png` shows the primary median-normalized width,
  `*_width_training_sd_coverage.png` shows the secondary SD-normalized width,
  and `*_deviation.png` shows median-normalized outside distance separately.
  The width chart axes and titles explicitly identify their denominator.
- `zero_50/`, `zero_60/`, `zero_70/`, `zero_80/`: each contains `zerolist.csv`,
  all training classifications in `training_profiles.csv`, the actual modeled
  population in `model_training_profiles.csv`, fresh models, fitting logs, settings,
  and non-Zero-only `predictions.csv.gz` for the maximum requested test horizon.
- Each threshold's case subfolder contains `metrics_long.csv`,
  `all_outside_points.csv.gz`, and `histograms/`. Histograms have only four group
  colors and separately show the four primary methods and six applicable methods.
  They retain signed tails and all outside points, with counts and audit files.
- `validation.json` and `sample_audit.csv`: checks performed during the run.
  `independent_validation.json`: produced only by `validate_results.py`, which
  independently recomputes metrics, checks Zero exclusion and raw signed actuals,
  reloads model files, verifies both width formulas against raw training scales,
  checks scale-floor counts, and recounts the histograms.

## Plot saved results without training

`plot_results.py` contains the width/coverage and deviation plotting code.
`plot_outliers.py` contains the outlier histogram code. Neither imports LightGBM.
Run the experiment first; previous Zero-prediction results are rejected.
Saved metrics from before the normalized-width change must also be regenerated
before plotting. The plotter reports missing columns instead of silently using
absolute widths with normalized labels. Outlier histogram distances continue to
use their existing median normalization.

```powershell
# Comparison figures only (all thresholds and both testing cases)
& 'C:\Users\Public\anaconda3\python.exe' .\plot_results.py
# Selected comparison panels
& 'C:\Users\Public\anaconda3\python.exe' .\plot_results.py --case main_W42 --panels Current_nonzero Smooth Lumpy
# All outlier histograms
& 'C:\Users\Public\anaconda3\python.exe' .\plot_outliers.py
# Main testing, 60% Zero threshold, 80% prediction interval
& 'C:\Users\Public\anaconda3\python.exe' .\plot_outliers.py --thresholds 60 --case main_W42 --levels 80
```

Double-click `plot_windows.bat` for comparison figures or
`plot_outliers_windows.bat` for histograms. `run_experiment.py --plots-only`
regenerates both types. Plotters accept `--results`, `--config`, and `--dpi`.

The original B experiment and earlier threshold results under `results/` are
preserved. `legacy_zero_prediction_code/` contains the previous threshold source
files for reference. Current entry points use the exclusion policy above.
