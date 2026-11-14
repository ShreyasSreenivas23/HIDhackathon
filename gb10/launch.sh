#!/usr/bin/env bash
# VPN WINDOW #1 — upload bundle + start training in tmux, in ONE ssh connection (~5-15 s).
#   bash gb10/launch.sh            (from the repo root, Git Bash)
#   RESTART=1 bash gb10/launch.sh  (kill a running job and start over)
set -euo pipefail
cd "$(dirname "$0")/.."
source gb10/config.env
[ -n "$GB10_HOST" ] || { echo "Set GB10_HOST in gb10/config.env"; exit 1; }
[ -d data/gb10_bundle ] || { echo "Run first: api/.venv/Scripts/python gb10/prepare_bundle.py"; exit 1; }
SSH=(ssh -p "$GB10_PORT" -o ConnectTimeout=8 -o ServerAliveInterval=5 -o StrictHostKeyChecking=accept-new)
t0=$(date +%s)
tar czf - -C data/gb10_bundle . | "${SSH[@]}" "$GB10_HOST" \
  "mkdir -p ~/$GB10_DIR && tar xzf - -C ~/$GB10_DIR && HID=\$HOME/$GB10_DIR RESTART=${RESTART:-0} bash ~/$GB10_DIR/gb10/start.sh"
echo "launch done in $(( $(date +%s) - t0 )) s — the VPN can drop now; training continues on the GB10."
