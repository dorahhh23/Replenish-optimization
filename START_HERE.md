# Run Experiment B locally

Extract the ZIP to a normal folder, for example `files/CHATGPT/Bexperiment`.
Keep the scripts beside `input/` and `results/`. All code and chart/table
labels are English. This project needs Python 3.10 or later; Python 3.11
is suitable. Python itself is not bundled.

## Windows: double-click entry points

1. Run `00_install_plotting.bat` once to create a private `.venv` and install
   the drawing dependencies. Internet access is needed for installation.
2. Run `01_histogram.bat` to draw the main 80% interval histogram.
3. Run `02_all_figures.bat` to redraw all 98 images from saved predictions.
4. Run `03_train_and_run_all.bat` to install the training dependencies,
   fit the models again, regenerate all results and validate the outputs.

The full training command takes longer than drawing saved results. Installation
or execution errors remain visible in the command window. These scripts use
the project `.venv`, without changing another project's environment.

## macOS / Linux

Open a terminal in the extracted folder, then run:

```bash
bash setup_local.sh
bash run_local.sh histogram
bash run_local.sh figures
```

For a complete refit instead of drawing saved results:

```bash
.venv/bin/python -m pip install -r requirements.txt
bash run_local.sh full
```

If LightGBM reports a missing system/OpenMP library, install the appropriate
runtime for your operating system, then rerun the full command. This does
not affect the drawing-only workflow.

## Any operating system: use an existing Python environment

```bash
python -m pip install -r requirements_plotting.txt
python run_local.py histogram
python run_local.py histogram --case all --bins 60
python run_local.py figures
```

For training and full validation:

```bash
python -m pip install -r requirements.txt
python run_local.py full
```

After changing source code, rerun the relevant command. Model changes require
`full`; style/binning changes only require `histogram` or `figures`.

## Main output locations

- `figures_histogram/main_W42/histogram_80_nonzero_median.png`: new histogram.
- `figures_histogram/main_W42/histogram_80_nonzero_median_counts.csv`: exact
  histogram counts for every bin, model and demand group.
- `figures_histogram/main_W42/histogram_80_nonzero_median_summary.csv`: totals,
  below/above counts, distance quantiles and retained negative actuals.
- `revised_charts/main_W42/width_coverage_overall.png`: combined comparison.
- `figures_signed_scatter/main_W42/`: every outside point, without clipping.
- `results/`: predictions, tables, profiles and fitted LightGBM models.
- `validation.json`: independent checks after a full run.

The histogram counts points outside the prediction interval. The signed
coordinate is negative below the lower bound and positive above the upper
bound, divided by each SKU's absolute non-zero training median (minimum
divisor 1). All five demand groups remain; returns retain their original
negative sales values. See `README.md` for methods and interpretation.
