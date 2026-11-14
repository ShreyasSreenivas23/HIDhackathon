"""Download pretrained weights into api/models/.  Run from api/:  python scripts/download_models.py

- MiniFASNet-V2-SE ONNX (face-antispoof-onnx, Apache-2.0, 98.2% on CelebA-Spoof test)
- YOLO11n COCO weights (ultralytics, AGPL-3.0 — fine for a hackathon demo)
- insightface buffalo_l downloads itself on first API start into models/insightface/
"""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

MODELS = Path(__file__).resolve().parent.parent / "models"
FILES = {
    "minifas_best_model.onnx": "https://github.com/facenox/face-antispoof-onnx/raw/main/models/best/98.20/best_model.onnx",
    "yolo11n.pt": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt",
    # OpenCV face backend (works natively on Windows; used when insightface is not installed)
    "face_detection_yunet_2023mar.onnx": "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    "face_recognition_sface_2021dec.onnx": "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
}


def main() -> int:
    MODELS.mkdir(parents=True, exist_ok=True)
    for name, url in FILES.items():
        dst = MODELS / name
        if dst.exists() and dst.stat().st_size > 0:
            print(f"ok      {name}")
            continue
        print(f"fetch   {name} <- {url}")
        try:
            urllib.request.urlretrieve(url, dst)
        except Exception as e:
            print(f"FAILED  {name}: {e}", file=sys.stderr)
            dst.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
