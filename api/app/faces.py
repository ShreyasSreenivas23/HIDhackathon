"""Face detection + recognition embeddings.

Two interchangeable backends (same interface):
  1. insightface buffalo_l (SCRFD + ArcFace)   — preferred when installed (Docker/Linux).
  2. OpenCV YuNet + SFace (opencv_zoo, Apache-2.0) — built into opencv-python, works natively on
     Windows with no compiler. SFace: ~99.4% on LFW. Models fetched by scripts/download_models.py.

Keypoint order for both: [eye_imgleft, eye_imgright, nose, mouth_imgleft, mouth_imgright] (raw frames).
Set FACE_BACKEND=opencv|insightface to force one; default = insightface if importable, else opencv.
NOTE: similarity scales differ slightly between backends — recalibrate thresholds with eval/ after switching.
"""
from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)

YUNET_FILE = "face_detection_yunet_2023mar.onnx"
SFACE_FILE = "face_recognition_sface_2021dec.onnx"


@dataclass
class DetectedFace:
    bbox: list[float]      # x1, y1, x2, y2
    kps: np.ndarray        # 5x2
    det_score: float
    raw: np.ndarray | None = field(default=None, repr=False)   # backend-specific (YuNet row for SFace alignCrop)

    @property
    def size(self) -> float:
        return min(self.bbox[2] - self.bbox[0], self.bbox[3] - self.bbox[1])


class _InsightfaceBackend:
    name = "insightface"

    def __init__(self, models_dir: Path, det_size: int):
        from insightface.app import FaceAnalysis

        self.app = FaceAnalysis(name="buffalo_l", root=str(models_dir / "insightface"),
                                allowed_modules=["detection", "recognition"], providers=["CPUExecutionProvider"])
        self.app.prepare(ctx_id=-1, det_size=(det_size, det_size))
        self.rec = self.app.models["recognition"]

    def detect(self, img):
        bboxes, kpss = self.app.det_model.detect(img, max_num=0, metric="default")
        return [DetectedFace(bboxes[i, :4].tolist(), kpss[i], float(bboxes[i, 4])) for i in range(bboxes.shape[0])]

    def embed(self, img, face):
        from insightface.app.common import Face

        return self.rec.get(img, Face(bbox=np.array(face.bbox), kps=face.kps, det_score=face.det_score))


class _OpenCVBackend:
    name = "opencv"

    def __init__(self, models_dir: Path, det_size: int, score_threshold: float = 0.7):
        det_path, rec_path = models_dir / YUNET_FILE, models_dir / SFACE_FILE
        if not det_path.exists() or not rec_path.exists():
            raise FileNotFoundError(f"missing {det_path.name}/{rec_path.name} — run scripts/download_models.py")
        self.det = cv2.FaceDetectorYN.create(str(det_path), "", (320, 320), score_threshold, 0.3, 5000)
        self.rec = cv2.FaceRecognizerSF.create(str(rec_path), "")
        self.max_side = max(det_size * 2, 640)   # detect on a downscaled copy for speed
        self.lock = threading.Lock()             # OpenCV DNN objects are not thread-safe

    def detect(self, img):
        h, w = img.shape[:2]
        s = min(1.0, self.max_side / max(h, w))
        small = cv2.resize(img, (int(w * s), int(h * s))) if s < 1 else img
        with self.lock:
            self.det.setInputSize((small.shape[1], small.shape[0]))
            _, rows = self.det.detect(small)
        out = []
        for r in (rows if rows is not None else []):
            r = r.copy()
            r[:14] /= s                                    # back to full-resolution coordinates
            x, y, bw, bh = r[:4]
            out.append(DetectedFace([float(x), float(y), float(x + bw), float(y + bh)],
                                    r[4:14].reshape(5, 2).astype(np.float32), float(r[14]), raw=r))
        return out

    def embed(self, img, face):
        with self.lock:
            aligned = self.rec.alignCrop(img, face.raw)
            return self.rec.feature(aligned).reshape(-1)


class FaceEngine:
    def __init__(self, models_dir: Path, det_size: int = 320):
        self.backend = None
        choice = os.getenv("FACE_BACKEND", "auto").lower()
        order = {"insightface": [_InsightfaceBackend], "opencv": [_OpenCVBackend]}.get(choice, [_InsightfaceBackend, _OpenCVBackend])
        for cls in order:
            try:
                self.backend = cls(models_dir, det_size)
                log.info("face backend: %s", cls.name)
                break
            except Exception as e:
                log.warning("face backend %s unavailable: %s", cls.name, e)

    @property
    def available(self) -> bool:
        return self.backend is not None

    @property
    def name(self) -> str | None:
        return self.backend.name if self.backend else None

    def detect(self, img_bgr: np.ndarray) -> list[DetectedFace]:
        faces = self.backend.detect(img_bgr)
        faces.sort(key=lambda f: f.size, reverse=True)
        return faces

    def embed(self, img_bgr: np.ndarray, face: DetectedFace) -> np.ndarray:
        emb = np.asarray(self.backend.embed(img_bgr, face), dtype=np.float32)
        return emb / (np.linalg.norm(emb) + 1e-9)
