#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# Keep the user's currently activated Python, as required by the server handoff.
# Offline environment variables are applied by Python before any SPURS imports.
exec python -u "$PROJECT_DIR/scripts/run_spurs.py" "$@"
