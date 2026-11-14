"""Face detection (SCRFD) + ArcFace embeddings via insightface buffalo_l.

Only the detection and recognition models are loaded (skip age/gender/landmark heads -> faster).
Model zoo licence: non-commercial research — fine for the hackathon, mention in Q&A.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class DetectedFace:
    bbox: list[float]      # x1, y1, x2, y2
    kps: np.ndarray        # 5x2
    det_score: float

    @property
    def size(self) -> float:
        return min(self.bbox[2] - self.bbox[0], self.bbox[3] - self.bbox[1])


class FaceEngine:
    def __init__(self, models_dir: Path, det_size: int = 320):
        self.app = None
        try:
            from insightface.app import FaceAnalysis

            self.app = FaceAnalysis(
                name="buffalo_l",
                root=str(models_dir / "insightface"),
                allowed_modules=["detection", "recognition"],
                providers=["CPUExecutionProvider"],
            )
            self.app.prepare(ctx_id=-1, det_size=(det_size, det_size))
            self.rec = self.app.models["recognition"]
        except Exception as e:
            log.warning("FaceEngine disabled (insightface missing?): %s", e)
            self.app = None

    @property
    def available(self) -> bool:
        return self.app is not None

    def detect(self, img_bgr: np.ndarray) -> list[DetectedFace]:
        bboxes, kpss = self.app.det_model.detect(img_bgr, max_num=0, metric="default")
        faces = []
        for i in range(bboxes.shape[0]):
            faces.append(DetectedFace(bbox=bboxes[i, :4].tolist(), kps=kpss[i], det_score=float(bboxes[i, 4])))
        faces.sort(key=lambda f: f.size, reverse=True)
        return faces

    def embed(self, img_bgr: np.ndarray, face: DetectedFace) -> np.ndarray:
        from insightface.app.common import Face

        f = Face(bbox=np.array(face.bbox), kps=face.kps, det_score=face.det_score)
        emb = self.rec.get(img_bgr, f)
        return (emb / (np.linalg.norm(emb) + 1e-9)).astype(np.float32)
