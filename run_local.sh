#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
if [ ! -x .venv/bin/python ]; then
  echo "Run bash setup_local.sh first."
  exit 1
fi
exec .venv/bin/python run_local.py "$@"
