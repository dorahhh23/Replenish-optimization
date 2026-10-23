import argparse
from pathlib import Path
import sys
from .config import load_config
from .pipeline import run_pipeline


def main(argv=None):
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Read-only replenishment demand analysis over the last 12 complete months")
    parser.add_argument("--input", type=Path, default=root.parent / "Replenishment Forecastint Report All Field (1).xlsx")
    parser.add_argument("--output", type=Path, default=root / "outputs")
    parser.add_argument("--config", type=Path, default=root / "config.json")
    parser.add_argument("--reference-date", default=None, help="YYYY-MM-DD; defaults to the system run date")
    parser.add_argument("--inspect-only", action="store_true", help="Inspect source schema and pause before formal analysis")
    args = parser.parse_args(argv)
    try:
        result = run_pipeline(args.input, args.output, load_config(args.config), args.reference_date, args.inspect_only)
    except Exception as error:
        print(f"Analysis paused: {error}", file=sys.stderr)
        return 2
    print(f"{result['status']}: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
