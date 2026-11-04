# Zero threshold comparison

Run from this folder in Windows PowerShell:

```powershell
& 'C:\Users\Public\anaconda3\python.exe' .\run_experiment.py
& 'C:\Users\Public\anaconda3\python.exe' .\run_experiment.py --thresholds 50 60 70 80
```

Or double-click `run_windows.bat`. Set the `ZERO_PYTHON` environment variable
to select another Python executable. The launcher quotes paths and changes to
its own directory. Python paths and input paths may contain spaces or Chinese.
The existing Anaconda environment supplies NumPy, pandas, SciPy and Matplotlib;
LightGBM 4.6.0 is installed locally in `_vendor`. No global packages were changed.

`config.json` sets thresholds, input, output, rounds, threads, seed, W43 inclusion
and PNG resolution. Relative input/output paths resolve against the config file.
Every normal run freshly fits all models at every requested threshold. Output is
restricted to a subfolder here. A rerun replaces this experiment's output files;
use `--output results_new` for a separate run. Original B results are preserved.
`--plots-only` redraws saved metrics without fitting models.

## Standalone plotting

`plot_results.py` contains the plotting code for this threshold experiment.
It reads saved `metrics_long.csv` and needs only NumPy, pandas and Matplotlib;
it does not import LightGBM or the training runner. `engine/make_combined_charts.py`
provides the original bar/dot styling. The training runner also calls this
shared plotting module, so both entry points produce the same figures.

Double-click `plot_windows.bat` to redraw all figures, or run:

```powershell
# All thresholds, cases, and panels
& 'C:\Users\Public\anaconda3\python.exe' .\plot_results.py
# Only the Zero demand group in the main test
& 'C:\Users\Public\anaconda3\python.exe' .\plot_results.py --case main_W42 --panels Zero
# Overall and fixed-sample comparisons
& 'C:\Users\Public\anaconda3\python.exe' .\plot_results.py --panels All_SKUs Fixed_60_nonzero
```

Optional arguments: `--dpi 200`, `--results results_new`, and `--config config.json`.
PNG output goes to the selected results folder's `main_W42/figures/` and/or
`sensitivity_W43/figures/`. Saved models, predictions, and metrics are unchanged.

## Protocol

- Training: 2025-W41 through 2026-W24, exactly 36 weeks. Main testing:
  2026-W25 through W42, 18 weeks. W43 sensitivity adds W43 to the same
  forecast horizon using the same fitted models; W43 may be incomplete.
- Zero means training zero share **at least** the threshold: 50%, 60%, 70%,
  80% require 18, 22, 26, 29 zero weeks, respectively. No test values determine
  classification. Other groups preserve the original positive-demand ADI 1.32
  and CV-squared 0.49 rules: Smooth, Erratic, Intermittent, Lumpy.
- Returns retain their original signed training and test values. Large orders
  are retained. Incomplete training histories are excluded once, consistently
  across thresholds; missing test values are unscored, never zero-filled.
- Zero forecasts have all seven quantiles equal to zero, including point
  forecast q50 and all three intervals [0,0]. Future actuals still count.
- Model code is snapshotted in `engine/` from the existing B implementation.
  The only engine change is Unicode-safe model-file writing on Windows.
  Empirical, Recent13, DampedTrendResidual and grouped LightGBM retain their
  original settings, seed, chronological validation, quantile correction and
  training-only final refit. Each threshold fits its own LightGBM models;
  no previous forecasts or model files are loaded. Model files and fitting
  logs are saved separately for each threshold.
- Coverage is a proportion; width is in sales units. Outside distance is
  `max(lower - actual, actual - upper, 0)`. Divide each point by its SKU's
  `max(median(abs(nonzero training sales)), 1)`, then average **all valid test
  points**, with covered points contributing zero. No WIS is reported.

## Samples and outputs

`results/overall_comparison.csv`: all four thresholds, all complete-training
SKUs, identical observed SKU-week records for the four signed-sales methods.
`comparison_all_tables.csv` and `metrics_long.csv`: every case and cohort.
`group_counts.csv`: all five groups at each threshold, including zero counts.
`cohort_membership.csv`: per-SKU group at each threshold and fixed-sample flags.

Within each case, `tables/`, `coverage_width_tables/`, `deviation_tables/`,
and `figures/` contain the following comparisons:

- `All_SKUs`: fixed primary sample, including all returns.
- `Current_nonzero`: excludes the current Zero group; **sample varies**.
- `Fixed_60_nonzero`: SKUs classified non-Zero at 60%, fixed across thresholds.
  At 50% some of these SKUs may receive the Zero forecast rule; they remain scored.
- Five demand groups: membership can vary with the threshold.
- `Common_count`: six methods on the same fixed sample of nonnegative integer
  training histories, independently of Zero assignment.
- `Common_count_fixed_60_nonzero`: intersection of the preceding fixed samples.

Poisson and NegativeBinomial do not remove returns from the main analysis.
Even in the count-eligible sample, negative test actuals remain scored.
Eligibility is determined by training values only. The original NegativeBinomial
implementation falls back to Poisson when training variance does not exceed mean.

The width/coverage PNGs preserve B's horizontal blue interval-width bars and
orange coverage dots on separate axes. Each image compares all thresholds.
Mean normalized outside distance is plotted separately. Tables, axes, titles,
and legends are in English. Empty cohorts are labeled and reported with zero
observations and missing metrics, not artificial zero scores.

`results/zero_50/` (and 60/70/80) contains training profiles, seven quantile
models per non-Zero group, training logs, settings and `predictions.csv.gz`.
This CSV compression is a storage format for predictions, not a ZIP delivery.
The prediction file contains the maximum test horizon; use `iso_week` to select
the main W42 case or sensitivity through W43. Per-case metrics are also saved.

`validation.json` and `sample_audit.csv` record checks of group boundaries,
sample equality, raw actuals including missing/negative values, Zero forecasts,
normalization, interval nesting, and original-file hashes. `run_manifest.json`
records the exact environment, settings, code hashes and fitting logs.
`validate_results.py` independently rereads predictions, recomputes all metrics,
and checks saved models and source values. Run it after an experiment:

```powershell
& 'C:\Users\Public\anaconda3\python.exe' .\validate_results.py
```
