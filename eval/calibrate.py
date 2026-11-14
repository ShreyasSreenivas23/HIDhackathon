"""Pick decision thresholds from measured scores and write api/thresholds.json.

  python eval/calibrate.py --apcer 0.002 --bpcer 0.01 --far-hi 0.001 --far-lo 0.05
Inputs: eval/results/celeba_scores.csv (liveness, fused with the API's fusion) and eval/results/lfw_scores.npz.
  liveness_hi: attacks above it <= --apcer   (ALLOW only when this sure)
  liveness_lo: live users below it <= --bpcer (hard DENY rarely hits real people; the band between goes to the challenge)
  identity_hi: impostors above it <= --far-hi
  identity_lo: impostors above it <= --far-lo
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(REPO / "api"), str(REPO / "eval")]
from app.config import settings  # noqa: E402
from app.liveness.fuse import fuse  # noqa: E402
from metrics import threshold_at_miss, threshold_at_rate  # noqa: E402

RES = REPO / "eval" / "results"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apcer", type=float, default=0.002)
    ap.add_argument("--bpcer", type=float, default=0.01)
    ap.add_argument("--far-hi", type=float, default=0.001)
    ap.add_argument("--far-lo", type=float, default=0.05)
    ap.add_argument("--write", action="store_true", help="actually overwrite api/thresholds.json")
    args = ap.parse_args()

    th = settings.thresholds
    out = json.loads(settings.thresholds_file.read_text()) if settings.thresholds_file.exists() else {}

    f = RES / "celeba_scores.csv"
    if f.exists():
        rows = list(csv.DictReader(open(f)))
        sig_names = [k for k in rows[0] if k in th.weights or k == "device"]
        fused, live = [], []
        for r in rows:
            sig = {k: (float(r[k]) if r[k] not in ("", "None") else None) for k in sig_names}
            s, _ = fuse(sig, th.weights, th.signal_flag)
            fused.append(0.5 if s is None else s)
            live.append(r["live"] == "1")
        fused, live = np.array(fused), np.array(live)
        hi = threshold_at_rate(fused[~live], args.apcer)
        lo = threshold_at_miss(fused[live], args.bpcer)
        if lo >= hi:
            lo = max(0.05, hi - 0.15)
        out.update(liveness_hi=round(min(hi, 0.99), 4), liveness_lo=round(lo, 4))
        print(f"liveness: lo={lo:.4f} hi={hi:.4f}  (live in gray band: {np.mean((fused[live] >= lo) & (fused[live] < hi)):.3f})")

    f = RES / "lfw_scores.npz"
    if f.exists():
        d = np.load(f)
        hi = threshold_at_rate(d["impostor"], args.far_hi)
        lo = threshold_at_rate(d["impostor"], args.far_lo)
        out.update(identity_hi=round(hi, 4), identity_lo=round(min(lo, hi - 0.02), 4))
        print(f"identity: lo={lo:.4f} hi={hi:.4f}  genuine FRR at hi: {np.mean(d['genuine'] < hi):.4f}")

    print(json.dumps(out, indent=2))
    if args.write:
        settings.thresholds_file.write_text(json.dumps(out, indent=2))
        print("wrote", settings.thresholds_file, "-> restart the API")


if __name__ == "__main__":
    main()
