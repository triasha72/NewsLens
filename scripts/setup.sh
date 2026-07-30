#!/usr/bin/env bash
set -euo pipefail

python_command="${PYTHON_BIN:-python3.12}"

if ! command -v "${python_command}" >/dev/null 2>&1; then
  echo "Python 3.12 was not found. Install it or set PYTHON_BIN explicitly." >&2
  exit 1
fi

"${python_command}" -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m pytest
python -m ruff check .

echo "NewsLens setup complete."
echo "Activate it later with: source .venv/bin/activate"
