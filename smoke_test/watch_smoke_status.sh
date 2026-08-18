#!/usr/bin/env bash
set -euo pipefail

# Continuously watch smoke status output during long runs.
#
# Usage:
#   bash smoke_test/watch_smoke_status.sh
#
# Optional env vars:
#   REPO_DIR=/content/GemmaTest
#   STATUS_FILE=/content/GemmaTest/smoke_status.txt
#   INTERVAL_SEC=5

REPO_DIR="${REPO_DIR:-/content/GemmaTest}"
STATUS_FILE="${STATUS_FILE:-$REPO_DIR/smoke_status.txt}"
INTERVAL_SEC="${INTERVAL_SEC:-5}"

if ! [[ "$INTERVAL_SEC" =~ ^[0-9]+$ ]] || [[ "$INTERVAL_SEC" -lt 1 ]]; then
  echo "ERROR: INTERVAL_SEC must be a positive integer."
  exit 1
fi

echo "Watching status file: $STATUS_FILE"
echo "Press Ctrl+C to stop."

while true; do
  echo ""
  echo "===== $(date -u +%Y-%m-%dT%H:%M:%SZ) ====="
  if [[ -f "$STATUS_FILE" ]]; then
    cat "$STATUS_FILE"
  else
    echo "Status file not created yet."
  fi
  sleep "$INTERVAL_SEC"
done
