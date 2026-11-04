# Code update status — November 4, 2026

The Zero-exclusion revision has been saved but not executed or runtime-tested,
at the user's request. No new experiment results are claimed.

Run `run_experiment.py` and then `validate_results.py` using the commands in README.md.
New output defaults to `results_zero_excluded/`; existing `results/`, `run.log`,
`validation.log`, and `plot_validation.json` belong to the earlier policy.

Modified: run_experiment.py, engine/run_b_experiment.py, plot_results.py,
validate_results.py, config.json, run_windows.bat, plot_windows.bat, README.md.
Added: plot_outliers.py, plot_outliers_windows.bat, CODE_UPDATE.md.
Previous code is backed up in legacy_zero_prediction_code/.

## Normalized width update

The current code uses mean nonzero-absolute-median-normalized width as the primary
metric and mean training-sample-SD-normalized width as the secondary metric. Each
SKU-week is normalized before averaging over valid test points. Both training
scales are floored at one sales unit. Main tables and charts use these metrics;
absolute width is retained in a reference CSV. Scale-floor counts and independent
verification of both formulas have been added.

Files updated for this change: run_experiment.py, plot_results.py,
engine/make_combined_charts.py, validate_results.py, README.md, CODE_UPDATE.md.
No experiments, plots, or runtime tests were executed for this update.
