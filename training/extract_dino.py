"""Extract frozen DINOv2 features for CelebA-Spoof face crops (run on the GB10).

  python training/extract_dino.py --root /data/CelebA_Spoof --split train --per-type 4000 --out data/feats_train.npz
  python training/extract_dino.py --root /data/CelebA_Spoof --split test  --per-type 1000 --out data/feats_test.npz

GB10 is ARM64: use the NVIDIA NGC PyTorch container, e.g.
  docker run --gpus all -it -v $PWD:/ws -v /data:/data nvcr.io/nvidia/pytorch:<latest>-py3
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "api"))
sys.path.insert(0, str(REPO / "training"))
from app.liveness.dino_head import preprocess  # noqa: E402  (same crop/normalisation as inference)
from celeba_spoof import load_image, load_labels, sample, subject_of  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--split", default="train", choices=["train", "test"])
    ap.add_argument("--per-type", type=int, default=3000)
    ap.add_argument("--backbone", default="dinov2_vits14")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--expansion", type=float, default=1.5)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = torch.hub.load("facebookresearch/dinov2", args.backbone).to(dev).eval()
    items = sample(load_labels(args.root, args.split), args.per_type, root=args.root)
    print(f"{len(items)} images, device={dev}")

    feats, types, subjects, paths, buf, meta = [], [], [], [], [], []
    t0 = time.time()

    def flush():
        if not buf:
            return
        x = torch.from_numpy(np.stack(buf)).to(dev)
        with torch.inference_mode(), torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
            f = model(x).float().cpu().numpy()
        feats.append(f)
        for rel, t in meta:
            types.append(t); subjects.append(subject_of(rel)); paths.append(rel)
        buf.clear(); meta.clear()

    for i, (rel, t) in enumerate(items):
        img, bbox = load_image(args.root, rel)
        if img is None or bbox is None:
            continue
        buf.append(preprocess(img, bbox, args.expansion, 224))
        meta.append((rel, t))
        if len(buf) == args.batch:
            flush()
        if i % 2000 == 0:
            print(f"{i}/{len(items)}  {time.time() - t0:.0f}s")
    flush()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, feats=np.concatenate(feats), spoof_type=np.array(types),
                        live=(np.array(types) == 0).astype(np.int8), subject=np.array(subjects), path=np.array(paths),
                        backbone=args.backbone, expansion=args.expansion)
    print(f"saved {args.out}  ({len(types)} samples, {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
