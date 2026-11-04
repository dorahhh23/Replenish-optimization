#!/usr/bin/env python3
"""Portable local runner. Run 'python run_local.py --help' for the task menu."""
from pathlib import Path
import argparse
import importlib
import subprocess
import sys

HOME = Path(__file__).resolve().parent


def run(script, *args):
    command = [sys.executable, str(HOME / script), *map(str, args)]
    print(f"Running {script} {' '.join(map(str, args))}", flush=True)
    subprocess.run(command, cwd=HOME, check=True)


def check_dependencies(full=False):
    dependencies = ["numpy", "pandas", "matplotlib"]
    if full:
        dependencies += ["scipy", "lightgbm", "PIL"]
    missing = []
    for module in dependencies:
        try:
            importlib.import_module(module)
        except (ImportError, OSError) as error:
            missing.append(f"{module}: {error}")
    if missing:
        requirement = "requirements.txt" if full else "requirements_plotting.txt"
        raise RuntimeError("Dependencies are unavailable:\n  " + "\n  ".join(missing)
                           + f"\nInstall them with: python -m pip install -r {requirement}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", nargs="?", default="histogram",
                        choices=["histogram", "figures", "full", "validate"],
                        help="histogram: quick plot; figures: all images; full: train and draw; validate: full checks")
    parser.add_argument("--case", choices=["main_W42", "sensitivity_W43", "all"], default="main_W42",
                        help="Histogram task only; other tasks process both cases")
    parser.add_argument("--bins", type=int, default=72, help="Number of transformed bins for the histogram task")
    args = parser.parse_args()
    if sys.version_info < (3, 10):
        parser.error("Use Python 3.10 or later (Python 3.11 is suitable).")
    try:
        check_dependencies(full=args.task in {"full", "validate"})
        if args.task == "histogram":
            run("plot_outlier_histogram.py", "--case", args.case, "--levels", "80",
                "--normalizations", "nonzero_median", "--bins", args.bins)
        elif args.task == "figures":
            run("rebuild_figures.py")
        elif args.task == "full":
            run("run_b_experiment.py", "--include-week43-sensitivity")
            run("rebuild_figures.py")
            run("validate_five_groups.py")
        else:
            run("validate_five_groups.py")
    except (RuntimeError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"\nERROR: {error}\n")
    print(f"Done. Project folder: {HOME}", flush=True)


if __name__ == "__main__":
    main()
