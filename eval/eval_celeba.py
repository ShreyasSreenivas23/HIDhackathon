"""Liveness on CelebA-Spoof test images (printed photo / screen / mask) + ablation over signals.

  python eval/eval_celeba.py --root /data/CelebA_Spoof --per-type 300
Writes eval/results/celeba_scores.csv (per-sample signal scores) and eval/results/celeba_report.json.
Runs the SAME liveness modules as the API (single image -> planarity is n/a here).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(REPO / "api"), str(REPO / "training"), str(REPO / "eval")]
from app.config import settings  # noqa: E402
from app.liveness.device import DeviceDetector  # noqa: E402
from app.liveness.dino_head import DinoHead  # noqa: E402
from app.liveness.fuse import fuse  # noqa: E402
from app.liveness.minifas import MiniFAS  # noqa: E402
from celeba_spoof import load_image, load_labels, sample  # noqa: E402
from metrics import ATTACK_GROUP, per_attack_report, roc_auc  # noqa: E402

OUT = REPO / "eval" / "results"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--per-type", type=int, default=300)
    ap.add_argument("--thr", type=float, default=0.5)
    args = ap.parse_args()

    mods = {"minifas": MiniFAS(settings.minifas_model), "dino": DinoHead(settings.dino_head), "device": DeviceDetector(settings.yolo_model)}
    mods = {k: m for k, m in mods.items() if m.available}
    print("signals:", list(mods))
    items = sample(load_labels(args.root, "test"), args.per_type, seed=1, root=args.root)

    rows = []
    t0 = time.time()
    for i, (rel, t) in enumerate(items):
        img, bbox = load_image(args.root, rel)
        if img is None or bbox is None:
            continue
        row = {"path": rel, "spoof_type": t, "group": "live" if t == 0 else ATTACK_GROUP[t], "live": int(t == 0)}
        for k, m in mods.items():
            row[k] = m.score(img, bbox) if k == "device" else m.score([img], [bbox])
        rows.append(row)
        if i % 500 == 0:
            print(f"{i}/{len(items)} {time.time() - t0:.0f}s")

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "celeba_scores.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    y = np.array([r["live"] for r in rows], bool)
    groups = np.array([r["group"] for r in rows])
    report = {"n": len(rows), "ablation": {}}
    names = list(mods)
    for k in range(1, len(names) + 1):
        for combo in combinations(names, k):
            fused = []
            for r in rows:
                s, _ = fuse({c: r[c] for c in combo}, settings.thresholds.weights, settings.thresholds.signal_flag)
                fused.append(0.5 if s is None else s)
            rep = per_attack_report(fused, y, groups, args.thr)
            rep["AUC"] = roc_auc(fused, y)
            report["ablation"]["+".join(combo)] = rep
            print(f"{'+'.join(combo):25s} AUC={rep['AUC']:.4f} acc={rep['accuracy']:.4f} ACER={rep['ACER']:.4f} "
                  + " ".join(f"{g}={v['APCER']:.3f}" for g, v in rep["attacks"].items()) + f" BPCER={rep['BPCER']:.3f}")
    (OUT / "celeba_report.json").write_text(json.dumps(report, indent=2))
    print("saved", OUT / "celeba_report.json")


if __name__ == "__main__":
    main()
