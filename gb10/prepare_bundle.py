"""Build the GB10 bundle locally (run BEFORE the VPN window; takes ~20 s, no network).

  api/.venv/Scripts/python gb10/prepare_bundle.py
Creates data/gb10_bundle/ with: gb10 scripts, training/gb10_train.py, the 4 api modules it imports,
and kiosk/ = face crops from data/debug (labelled via gb10/kiosk_labels.json) + labels.csv.
"""
from __future__ import annotations

import csv
import json
import os
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "api"))
os.environ.setdefault("FACE_BACKEND", "opencv")
from app.config import settings  # noqa: E402
from app.faces import FaceEngine  # noqa: E402
from app.imaging import face_quality, square_crop  # noqa: E402

OUT = REPO / "data" / "gb10_bundle"
CODE = [
    "gb10/job.sh", "gb10/start.sh",
    "training/gb10_train.py",
    "api/app/__init__.py", "api/app/imaging.py",
    "api/app/liveness/__init__.py", "api/app/liveness/dino_head.py",
]
EXPANSION, SIZE, MIN_FACE = 1.5, 256, 100


def looks_like_poster_test(d: Path, fe: FaceEngine) -> bool:
    """My fake-camera test (poster face image streamed into the kiosk): face ~195 px, bright ~0.48, static.
    It is a digital image, neither a live person nor a real recapture attack -> exclude from training."""
    img = cv2.imread(str(d / "frame_0.jpg"))
    if img is None:
        return False
    faces = fe.detect(img)
    if not faces:
        return False
    bright = face_quality(img, faces[0].bbox)["brightness"]
    return 185 <= faces[0].size <= 202 and 0.45 <= bright <= 0.52


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    for rel in CODE:
        dst = OUT / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        data = (REPO / rel).read_bytes().replace(b"\r\n", b"\n")      # LF line endings for Linux
        dst.write_bytes(data)

    labels = json.loads((REPO / "gb10" / "kiosk_labels.json").read_text())
    spoof, excl = set(labels["spoof_bursts"]), set(labels["exclude_bursts"])
    fe = FaceEngine(settings.models_dir)
    kdir = OUT / "kiosk"
    kdir.mkdir(parents=True)
    rows, n_b = [], {"live": 0, "spoof": 0, "skipped": 0}
    for d in sorted((REPO / "data" / "debug").iterdir()):
        ts = d.name.split("_")[0]
        if ts in excl or looks_like_poster_test(d, fe):
            n_b["skipped"] += 1
            continue
        live = 0 if ts in spoof else 1
        n_b["spoof" if live == 0 else "live"] += 1
        for f in sorted(d.glob("frame_*.jpg")):
            img = cv2.imread(str(f))
            faces = fe.detect(img) if img is not None else []
            if not faces or faces[0].size < MIN_FACE:
                continue
            crop = cv2.resize(square_crop(img, faces[0].bbox, EXPANSION), (SIZE, SIZE), interpolation=cv2.INTER_AREA)
            name = f"{ts}_{f.stem}.jpg"
            cv2.imwrite(str(kdir / name), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
            rows.append({"file": name, "live": live, "burst": ts})
    with open(kdir / "labels.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["file", "live", "burst"])
        w.writeheader()
        w.writerows(rows)
    size = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file()) / 1e6
    print(f"bundle ready: {OUT}  ({size:.1f} MB)")
    print(f"kiosk bursts: {n_b}  crops: {len(rows)} ({sum(r['live'] for r in rows)} live, {sum(1 - r['live'] for r in rows)} spoof)")


if __name__ == "__main__":
    main()
