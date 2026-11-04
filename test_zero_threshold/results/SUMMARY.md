# Zero threshold experiment results

All four thresholds were freshly fitted on the local computer. Training: 2025-W41 through 2026-W24. Main testing: 2026-W25 through W42. W43 sensitivity is separate because W43 may be incomplete.

## Group counts

| Zero threshold | Minimum zero weeks | Smooth | Erratic | Intermittent | Lumpy | Zero |
|---|---:|---:|---:|---:|---:|---:|
| 50% | 18 | 95 | 484 | 11 | 398 | 5,476 |
| 60% | 22 | 95 | 484 | 23 | 604 | 5,258 |
| 70% | 26 | 95 | 484 | 57 | 849 | 4,979 |
| 80% | 29 | 95 | 484 | 119 | 1,056 | 4,710 |

## LightGBM: main test, 80% prediction interval

The Zero threshold and prediction interval level are different parameters.

| Sample | Zero threshold | SKUs | Observations | Coverage | Mean width | Mean normalized outside distance |
|---|---|---:|---:|---:|---:|---:|
| All_SKUs | 50% | 6,464 | 115,992 | 90.14% | 19.8751 | 0.589368 |
| All_SKUs | 60% | 6,464 | 115,992 | 91.26% | 21.4395 | 0.566438 |
| All_SKUs | 70% | 6,464 | 115,992 | 92.47% | 22.7562 | 0.542443 |
| All_SKUs | 80% | 6,464 | 115,992 | 93.23% | 23.5178 | 0.528677 |
| Fixed_60_nonzero | 50% | 1,206 | 21,663 | 84.56% | 106.4188 | 0.384821 |
| Fixed_60_nonzero | 60% | 1,206 | 21,663 | 90.57% | 114.7952 | 0.262045 |
| Fixed_60_nonzero | 70% | 1,206 | 21,663 | 90.83% | 115.9599 | 0.257415 |
| Fixed_60_nonzero | 80% | 1,206 | 21,663 | 90.56% | 114.4053 | 0.264173 |

On all SKUs, increasing the Zero threshold improves LightGBM coverage and reduces mean normalized outside distance, while increasing average width. On the fixed 60%-non-Zero sample, 70% has the lowest mean normalized outside distance for the 80% prediction interval; 80% does not improve it. These observations do not establish a single best threshold for every model, interval level, or sample.

## Audit and interpretation

- All primary comparisons use the same 6,464 SKUs and 115,992 observed main-test points. The 732 incomplete-training SKUs are excluded consistently. There are 360 missing main-test values among the eligible SKUs; these are not filled or scored.
- All 190 negative training values (182 SKUs) and 127 negative main-test values remain unchanged. Large orders are retained.
- The fixed 60%-non-Zero sample has 1,206 SKUs. The strict training-only count-eligible cohort has 6,282 SKUs. Count methods are supplemental and do not remove returns from primary analysis.
- Current_nonzero and the five group tables have threshold-dependent membership. Their changes must not be interpreted as fixed-sample model improvements.
- Covered points contribute zero to mean normalized outside distance. The denominator is the SKU training median absolute nonzero sale, floored at one. No WIS column is included.
- Original project files are unchanged (328 file hashes checked).

See comparison_all_tables.csv for all models, interval levels and samples; sample_audit.csv and validation.json for run checks; independent_validation.json for saved-result and model verification.
