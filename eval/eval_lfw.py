"""Face-authentication accuracy / FAR / FRR on LFW (standard 6,000 pairs).

  python eval/eval_lfw.py --lfw data/lfw --pairs data/lfw/pairs.txt
`--lfw` = folder of person subfolders (original or deep-funneled LFW). pairs.txt = official View-2 file.
Writes eval/results/lfw_scores.npz and eval/results/lfw_report.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(REPO / "api"), str(REPO / "eval")]
from app.config import settings  # noqa: E402
from app.faces import FaceEngine  # noqa: E402
from metrics import far_frr, threshold_at_rate  # noqa: E402

OUT = REPO / "eval" / "results"


def parse_pairs(path: Path):
    lines = path.read_text().strip().splitlines()[1:]
    for ln in lines:
        p = ln.split()
        if len(p) == 3:
            yield (p[0], int(p[1])), (p[0], int(p[2])), 1
        elif len(p) == 4:
            yield (p[0], int(p[1])), (p[2], int(p[3])), 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lfw", type=Path, required=True)
    ap.add_argument("--pairs", type=Path, required=True)
    args = ap.parse_args()

    eng = FaceEngine(settings.models_dir, det_size=640)   # LFW faces are small in a 250px image
    cache: dict = {}

    def emb(name, idx):
        key = (name, idx)
        if key not in cache:
            img = cv2.imread(str(args.lfw / name / f"{name}_{idx:04d}.jpg"))
            faces = eng.detect(img) if img is not None else []
            if not faces:
                cache[key] = None
            else:   # LFW: the face of interest is the one closest to the centre
                c = np.array(img.shape[1::-1]) / 2
                f = min(faces, key=lambda f: np.linalg.norm((np.array(f.bbox[:2]) + np.array(f.bbox[2:])) / 2 - c))
                cache[key] = eng.embed(img, f)
        return cache[key]

    sims, labels, missing = [], [], 0
    for a, b, same in parse_pairs(args.pairs):
        ea, eb = emb(*a), emb(*b)
        if ea is None or eb is None:
            missing += 1
            sims.append(-1.0)   # count as a failure to match (conservative)
        else:
            sims.append(float(ea @ eb))
        labels.append(same)
    sims, labels = np.array(sims), np.array(labels, bool)
    gen, imp = sims[labels], sims[~labels]

    # best-accuracy threshold (reported as "face authentication accuracy")
    cands = np.unique(sims)
    accs = [((sims >= t) == labels).mean() for t in cands]
    best_t = float(cands[int(np.argmax(accs))])
    rep = {"pairs": int(len(sims)), "missing_faces": missing, "best_threshold": best_t, "accuracy": float(max(accs))}
    for target in (0.001, 0.01, 0.05):
        t = threshold_at_rate(imp, target)
        far, frr = far_frr(gen, imp, t)
        rep[f"at_FAR_{target}"] = {"threshold": t, "FAR": far, "FRR": frr}
    th = settings.thresholds
    far, frr = far_frr(gen, imp, th.identity_hi)
    rep["at_identity_hi"] = {"threshold": th.identity_hi, "FAR": far, "FRR": frr}
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / "lfw_scores.npz", genuine=gen, impostor=imp)
    (OUT / "lfw_report.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
