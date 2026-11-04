# Experiment B: five demand groups, Zero threshold 60%

## Start on your own computer

This is a portable project folder, with editable Python source, merged input
data, fitted models, saved predictions, tables and PNG figures. It has no
fixed machine-specific paths. Extract the ZIP first; do not run scripts
inside the ZIP. You can place the extracted `Bexperiment` folder under
`files/CHATGPT/` or in another location of your choice. See `START_HERE.md`
for Windows, macOS and Linux commands.

Use Python 3.10 or later; Python 3.11 is suitable. The drawing-only workflow
does not need LightGBM or model fitting. The full workflow installs LightGBM
4.6.0 and retrains the existing Experiment B model specification.

```bash
python -m pip install -r requirements_plotting.txt
python run_local.py histogram
```

The quick command draws the main W42 80% interval histogram, normalized by
the median absolute non-zero training sale. To redraw all figures from saved
results, run `python run_local.py figures`. To retrain and rebuild everything,
install `requirements.txt` and run `python run_local.py full`.

This replaces the previous additional returns group with exactly five groups:
Smooth, Erratic, Intermittent, Lumpy and Zero. Returns remain negative in both
training data and test actuals. No SKU is removed or reclassified into a
separate group because it has a return.

## Open the results

- `revised_charts/main_W42/width_coverage_overall.png`: width bars and coverage
  dots in the requested horizontal layout, including and excluding Zero.
- `revised_charts/B_deviation_median.csv`: separate overall and group tables
  of mean median-normalized outside distance.
- `figures/main_W42/outside_points/heatmap_80_nonzero_median.png`: the density
  heatmap, with the purple-orange-yellow palette (higher counts darker), a
  thin orange dashed zero marker, and training/testing dates in its title.
- `figures_signed_scatter/main_W42/scatter_80_nonzero_median_asymmetric.png`:
  the scatter plot with every signed point and data-based asymmetric y limits.
- `results/training_profiles.csv`: every SKU's five-group classification and
  unmodified training characteristics.
- `results/group_reassignment.csv`: the 119 reassigned SKU identifiers and
  their current five-group labels.
- `figures_histogram/main_W42/histogram_80_nonzero_median.png`: the new
  six-model histogram of signed, median-normalized outside distances.
- Beside each histogram, `_counts.csv` contains bin counts by model and
  demand group; `_summary.csv` gives below/above counts and distance quantiles.

The main case trains on 2025-W41 to 2026-W24 (36 weeks) and tests on 2026-W25
to W42 (18 weeks). W43 sensitivity adds the provisional final week using the
same fitted models. Complete training histories are required; missing test
actuals are unscored. Missing values are never replaced with zero.

## Classification and model fitting

At least 60% exactly-zero training weeks assigns the Zero group, before the
other classification rules. With 36 training weeks this means at least 22
zero weeks. The other four groups use ADI 1.32 and positive-demand CV squared
0.49. Positive demand characterizes demand occurrence/size; the original
negative returns remain in training and evaluation.

Current SKU counts: Zero 5,258; Lumpy 604; Erratic 484; Smooth 95;
Intermittent 23. All 6,464 complete-training SKUs are retained. Of the 119
previously separated SKUs, 55 now belong to Lumpy, 50 to Erratic, 13 to Smooth
and 1 to Intermittent.

Grouped LightGBM has been retrained within these four non-Zero groups; this
is a model change, not merely a legend edit. Seven quantiles directly define
70%, 80% and 90% intervals. Internal chronological validation and final
refitting use training data only. Quantile crossings are repaired by sorting.
For SKUs with a negative training value, negative forecast quantiles are
allowed. For nonnegative training histories, the previous nonnegative
forecast constraint is retained. The common Zero rule remains point forecast
zero and intervals [0,0]; future positive and negative actuals remain scored.

Main overall and group comparison charts/tables use the four methods that
support all SKUs: full historical empirical, recent 13-week empirical,
damped trend plus residuals, and grouped LightGBM. Every compared method has
the same observed SKU-week records in a panel.

Poisson and negative binomial require nonnegative integer training histories,
apart from the common Zero override. Supplemental `CountEligible*` tables
compare all six methods on common applicable populations. The six-model
scatter/heatmap panels retain every available forecast and show model-specific
point counts. This support restriction does not remove returns or SKUs from
the primary four-method analysis.

## Deviation and plots

Outside distance is `max(lower - actual, actual - upper, 0)`. Its signed form
is negative below the lower bound and positive above the upper bound.
Negative plot coordinates need not mean negative sales.

The main deviation table divides each point's distance by its SKU's median
absolute non-zero training sale, floored at 1 unit, then averages all scored
points. Covered points contribute zero. Extra training-SD and interval-width
normalizations are supplied in the full result metrics. Interval widths
remain in sales units. CSV coverage values are proportions.

All points are retained without sampling or tail clipping. Scatter y-axis
limits use the actual minima/maxima across the six methods, with a margin;
the original symlog transformation is retained. Heatmaps count those same
points in week-by-signed-distance bins, with logarithmic count colors. White
cells are empty. All chart labels and data tables are in English.

## Read the histogram

An outside point is an observed SKU-week whose actual sale is outside that
model's prediction interval. It is not automatically an error in the data.
Covered observations are absent from the histogram; they still contribute
zero to the mean deviation tables described above.

Each panel is one model. The horizontal coordinate is the signed outside
distance divided by `max(median(abs(nonzero training sales)), 1 unit)`.
Left of the dashed zero line means below the interval, and right means
above the interval. A negative coordinate does not necessarily mean a
negative actual sale. Negative sales (returns) remain unchanged.

The bar height counts outside points in a bin; colors stack the five demand
groups. Shared bin edges and y limits allow the six panels to be inspected
together. The horizontal display uses `sign(x) * log1p(abs(x) / 0.1)` to show
small deviations and the complete long tails. This changes axis spacing,
not the underlying signed distances. Counts are per transformed bin, not
probability density per linear sales unit. Limits use actual minima/maxima;
the negative span is not mirrored to match the larger positive tail.

Models can have different numbers of usable predictions, as explained
above. The histogram summarizes each model's available outside records,
not a shared-model accuracy ranking. Counts, below/above totals and tails
are checked against the source records with no sampling or clipping.

Change the histogram without editing the source:

```bash
python plot_outlier_histogram.py --case main_W42 --levels 80 --normalizations nonzero_median --bins 60
python plot_outlier_histogram.py --case all --levels 70 80 90 --normalizations training_sd nonzero_median interval_width
```

`plot_outlier_histogram.py` controls histogram style, binning and exports;
`plot_signed_scatter.py` controls scatter plots; `make_combined_charts.py`
controls width/coverage charts and deviation tables; `run_b_experiment.py`
and `lightgbm_features.py` control model fitting. The training/testing week
ranges are specific to the supplied data. Replacing the input with a new
date range also requires updating the split and figure period labels.

## Reproduce

Extract this complete replacement into a fresh folder, so previous generated
group files do not mix with the current outputs.

```bash
python -m pip install -r requirements.txt
python run_b_experiment.py --include-week43-sensitivity
python rebuild_figures.py
python validate_five_groups.py
```

To redraw saved results only, run `python rebuild_figures.py`; fitting
LightGBM again is unnecessary. `plot_signed_scatter.py` accepts `--case`,
`--levels` and `--normalizations` for a narrower scatter export.
`rebuild_figures.py` now also generates all 18 histogram variants: two
test cases, three interval levels and three normalizations. The complete
project contains 98 PNG figures, with independent histogram count checks.
