#!/usr/bin/env bash
# Runs ON THE GB10 inside tmux (started by start.sh). Needs no VPN: only the GB10's own internet
# (HuggingFace for data, GitHub/fbaipublicfiles for DINOv2 weights, PyPI for packages).
set -uo pipefail
HID="${HID:-$HOME/hid}"
cd "$HID"
echo "=== job start $(date) on $(hostname) ($(uname -m))"
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || echo "no nvidia-smi"
df -h "$HID" | tail -1

# ---- python env (reuse system torch if present via --system-site-packages) ----
PY="$(command -v python3 || command -v python)"
[ -d .venv ] || "$PY" -m venv --system-site-packages .venv || { echo "venv failed, using system python"; }
[ -f .venv/bin/activate ] && source .venv/bin/activate
python -m pip install -q --upgrade pip 2>/dev/null
python -m pip install -q numpy pyarrow huggingface_hub opencv-python-headless scikit-learn || { echo "pip deps failed"; exit 1; }
python -m pip install -q hf_transfer 2>/dev/null && python -c "import hf_transfer" 2>/dev/null && export HF_HUB_ENABLE_HF_TRANSFER=1

torch_ok() { python - <<'EOF'
import sys
try:
    import torch
except Exception:
    sys.exit(2)
if torch.cuda.is_available():
    try:
        (torch.randn(8, 8, device="cuda") @ torch.randn(8, 8, device="cuda")).sum().item()
        print("torch", torch.__version__, "CUDA OK:", torch.cuda.get_device_name(0)); sys.exit(0)
    except Exception as e:
        print("torch CUDA present but unusable:", e)
print("torch", torch.__version__, "CPU only"); sys.exit(1)
EOF
}
torch_ok; rc=$?
if [ $rc -ne 0 ]; then
  # try CUDA wheels for aarch64 (sbsa); newest first. Keep whatever works, else CPU is still fine.
  for idx in cu130 cu129 cu128; do
    echo "trying torch from https://download.pytorch.org/whl/$idx"
    python -m pip install -q --upgrade torch torchvision --index-url "https://download.pytorch.org/whl/$idx" && torch_ok && { rc=0; break; }
  done
  if [ $rc -ne 0 ]; then
    python -c "import torch" 2>/dev/null || python -m pip install -q torch torchvision
    echo "continuing on CPU (DINOv2-S on ~20 Grace cores is still OK, just slower)"
  fi
fi

# ---- the actual work (resumable) ----
python training/gb10_train.py --workdir "$HID/work" --kiosk "$HID/kiosk" --out "$HID/out"
echo "=== job end $(date) exit=$?"
