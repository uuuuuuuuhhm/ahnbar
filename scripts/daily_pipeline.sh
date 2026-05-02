#!/usr/bin/env bash
# Run the daily data + scoring pipeline from the repo root.
# Usage:
#   chmod +x scripts/daily_pipeline.sh
#   ./scripts/daily_pipeline.sh
#   ./scripts/daily_pipeline.sh --skip-fetch --predict
#
# Prefer a venv interpreter (see scripts/LAUNCHD.md for launchd examples).

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
exec python3 jobs/daily.py "$@"
