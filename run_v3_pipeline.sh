#!/usr/bin/env bash
# V3 Pipeline Launcher
# Usage: bash run_v3_pipeline.sh --episode_path /path/to/video.mp4 --work_dir output_v3
#
# All pipeline arguments can be passed directly. For example:
#   bash run_v3_pipeline.sh --episode_path video.mp4 --work_dir output_v3 --chunk_sec 600 --overlap_sec 15

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Auto-detect python (prefer python3)
if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN=python3
elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN=python
else
    echo "Error: python/python3 not found in PATH" >&2
    exit 1
fi

echo "[run_v3_pipeline] Python: $PYTHON_BIN"
echo "[run_v3_pipeline] Working directory: $SCRIPT_DIR"
echo "[run_v3_pipeline] Launching pipeline..."

"$PYTHON_BIN" "$SCRIPT_DIR/pipeline_v3_unified.py" "$@"
