#!/usr/bin/env bash
# VPN WINDOW #2 — pull results (a few MB) in ONE ssh connection and install the head.   bash gb10/fetch.sh
set -euo pipefail
cd "$(dirname "$0")/.."
source gb10/config.env
DEST=data/gb10_results
mkdir -p "$DEST"
ssh -p "$GB10_PORT" -o ConnectTimeout=8 "$GB10_HOST" \
  "cd ~/$GB10_DIR && tar czf - out job.log work/status.json 2>/dev/null" | tar xzf - -C "$DEST"
echo "fetched into $DEST"
if [ -f "$DEST/out/DONE" ]; then
  cp "$DEST/out/dino_head.npz" api/models/dino_head.npz
  echo "INSTALLED api/models/dino_head.npz — restart the API. Metrics:"
  cat "$DEST/out/metrics.json"
else
  echo "Not finished yet (no DONE marker). Status:"; cat "$DEST/work/status.json" 2>/dev/null; tail -n 15 "$DEST/job.log"
fi
