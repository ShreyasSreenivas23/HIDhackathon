"""CelebA-Spoof loader. Layout (official release):

  <root>/Data/{train,test}/<subject>/{live,spoof}/<img>.png|jpg
  <root>/Data/.../<img>_BB.txt            "x y w h score", box in a 224x224-scaled frame
  <root>/metas/intra_test/{train,test}_label.json   {"Data/train/.../x.png": [44 ints], ...}
      label[40] = spoof type (0 = live, 1..10 see eval/metrics.py), label[43] = live/spoof flag

VERIFY on the first few samples (draw the box!) — the BB scaling is the classic gotcha.
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

import cv2


def load_labels(root: Path, split: str) -> dict[str, list[int]]:
    return json.loads((root / "metas" / "intra_test" / f"{split}_label.json").read_text())


def read_bbox(img_path: Path, img_w: int, img_h: int) -> list[float] | None:
    bb = img_path.with_name(img_path.stem + "_BB.txt")
    if not bb.exists():
        return None
    vals = bb.read_text().split()
    if len(vals) < 4:
        return None
    x, y, w, h = map(float, vals[:4])
    sx, sy = img_w / 224.0, img_h / 224.0
    x1, y1 = x * sx, y * sy
    return [x1, y1, x1 + w * sx, y1 + h * sy]


def sample(labels: dict, per_type: int, seed: int = 0, root: Path | None = None) -> list[tuple[str, int]]:
    """Balanced sample: up to `per_type` images for each spoof type (live counts as type 0).
    Only keeps files that exist locally (you probably downloaded only some archive parts)."""
    by_type = defaultdict(list)
    for rel, lab in labels.items():
        if root is not None and not (root / rel).exists():
            continue
        by_type[int(lab[40])].append(rel)
    rng = random.Random(seed)
    out = []
    for t, items in by_type.items():
        rng.shuffle(items)
        n = per_type * (3 if t == 0 else 1)   # more live samples: there's 1 live class vs 10 spoof types
        out += [(rel, t) for rel in items[:n]]
    rng.shuffle(out)
    return out


def subject_of(rel: str) -> str:
    parts = Path(rel).parts
    return parts[2] if len(parts) > 2 else "?"


def load_image(root: Path, rel: str):
    img = cv2.imread(str(root / rel), cv2.IMREAD_COLOR)
    if img is None:
        return None, None
    return img, read_bbox(root / rel, img.shape[1], img.shape[0])
