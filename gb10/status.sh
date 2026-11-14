#!/usr/bin/env bash
# Any VPN window — one quick ssh: progress JSON + last log lines.   bash gb10/status.sh
set -euo pipefail
cd "$(dirname "$0")/.."
source gb10/config.env
ssh -p "$GB10_PORT" -o ConnectTimeout=8 "$GB10_HOST" \
  "cd ~/$GB10_DIR && cat work/status.json 2>/dev/null; echo; tail -n 12 job.log 2>/dev/null; ls out 2>/dev/null | tr '\n' ' '; echo"
