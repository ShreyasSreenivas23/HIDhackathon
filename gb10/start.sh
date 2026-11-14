#!/usr/bin/env bash
# Runs ON THE GB10 right after the bundle is unpacked (same SSH connection as the upload).
# Starts job.sh detached in tmux so it survives the VPN/SSH dropping. Returns in < 1 s.
HID="${HID:-$HOME/hid}"
SESSION=hidtrain
cd "$HID" || exit 1
chmod +x gb10/*.sh 2>/dev/null
if command -v tmux >/dev/null; then
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    if [ "${RESTART:-0}" = "1" ]; then tmux kill-session -t "$SESSION"; else
      echo "ALREADY RUNNING in tmux session '$SESSION' (set RESTART=1 to restart). Last log lines:"; tail -n 5 job.log; exit 0; fi
  fi
  tmux new-session -d -s "$SESSION" "bash $HID/gb10/job.sh 2>&1 | tee -a $HID/job.log"
  echo "STARTED in tmux session '$SESSION' at $(date +%H:%M:%S). Watch: tmux attach -t $SESSION"
else
  nohup setsid bash "$HID/gb10/job.sh" >> "$HID/job.log" 2>&1 < /dev/null &
  echo "STARTED with nohup/setsid (no tmux found), pid $!"
fi
