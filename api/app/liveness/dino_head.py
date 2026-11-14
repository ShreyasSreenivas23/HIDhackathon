"""DINOv2-S frozen features + logistic-regression head (trained in training/train_head.py on GB10).

Head file (joblib) = {"clf": sklearn classifier with predict_proba, "live_label": 1,
                      "backbone": "dinov2_vits14", "crop_expansion": 1.5, "img_size": 224}
"""
from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from ..imaging import square_crop

log = logging.getLogger(__name__)

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess(img_bgr: np.ndarray, bbox, expansion: float = 1.5, size: int = 224) -> np.ndarray:
    """Shared with training/extract_dino.py — keep identical. Returns CHW float32."""
    crop = square_crop(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), bbox, expansion)
    crop = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    return ((crop - MEAN) / STD).transpose(2, 0, 1)


class DinoHead:
    def __init__(self, head_path: Path):
        self.model = self.head = None
        if not head_path.exists():
            log.warning("DINO head missing at %s (train it with training/train_head.py)", head_path)
            return
        try:
            import joblib
            import torch

            self.head = joblib.load(head_path)
            self.torch = torch
            self.model = torch.hub.load("facebookresearch/dinov2", self.head.get("backbone", "dinov2_vits14"))
            self.model.eval()
            torch.set_num_threads(4)
        except Exception as e:  # torch missing / no internet for hub on first run
            log.warning("DINO head disabled: %s", e)
            self.model = None

    @property
    def available(self) -> bool:
        return self.model is not None

    def score(self, frames: list[np.ndarray], bboxes: list, max_frames: int = 2) -> float | None:
        if not self.available:
            return None
        pairs = [(f, b) for f, b in zip(frames, bboxes) if b is not None][:max_frames]
        if not pairs:
            return None
        exp, size = self.head.get("crop_expansion", 1.5), self.head.get("img_size", 224)
        x = np.stack([preprocess(f, b, exp, size) for f, b in pairs])
        with self.torch.inference_mode():
            feats = self.model(self.torch.from_numpy(x)).numpy()
        clf = self.head["clf"]
        live_col = list(clf.classes_).index(self.head.get("live_label", 1))
        return float(clf.predict_proba(feats)[:, live_col].mean())
