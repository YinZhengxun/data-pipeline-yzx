#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

if command -v python >/dev/null 2>&1; then
  PYTHON_BIN=python
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN=python3
else
  echo "python/python3 not found in PATH" >&2
  exit 1
fi

"$PYTHON_BIN" "$SCRIPT_DIR/pipeline_v3_unified.py" "$@"
