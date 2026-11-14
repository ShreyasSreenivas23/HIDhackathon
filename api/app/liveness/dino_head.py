"""DINOv2-S frozen features + linear liveness head (trained on the GB10 by training/train_head_hf.py).

Head file = portable .npz (no pickled sklearn objects, so GB10/laptop library versions can't break it):
    mean[D], scale[D], coef[D], intercept[1]   ->  P(live) = sigmoid(((f - mean) / scale) @ coef + intercept)
    backbone (str), crop_expansion (float), img_size (int)
Legacy .joblib heads ({"clf": sklearn model, ...}) are still accepted.
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
    """Shared with training/ — keep identical. Returns CHW float32."""
    crop = square_crop(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), bbox, expansion)
    crop = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    return ((crop - MEAN) / STD).transpose(2, 0, 1)


class LinearHead:
    def __init__(self, mean, scale, coef, intercept):
        self.mean, self.scale = np.asarray(mean, np.float32), np.asarray(scale, np.float32)
        self.coef, self.intercept = np.asarray(coef, np.float32).reshape(-1), float(np.asarray(intercept).reshape(-1)[0])

    def p_live(self, feats: np.ndarray) -> np.ndarray:
        z = ((feats - self.mean) / self.scale) @ self.coef + self.intercept
        return 1.0 / (1.0 + np.exp(-z))


def load_head(path: Path) -> tuple[LinearHead | object, dict]:
    if path.suffix == ".npz":
        d = np.load(path, allow_pickle=False)
        meta = {"backbone": str(d["backbone"]), "crop_expansion": float(d["crop_expansion"]), "img_size": int(d["img_size"])}
        return LinearHead(d["mean"], d["scale"], d["coef"], d["intercept"]), meta
    import joblib

    h = joblib.load(path)
    return h["clf"], {k: h.get(k, v) for k, v in {"backbone": "dinov2_vits14", "crop_expansion": 1.5, "img_size": 224}.items()}


class DinoHead:
    def __init__(self, head_path: Path):
        self.model = self.head = None
        if not head_path.exists():
            alt = head_path.with_suffix(".joblib" if head_path.suffix == ".npz" else ".npz")
            if not alt.exists():
                log.warning("DINO head missing at %s (train it on the GB10: see gb10/README.md)", head_path)
                return
            head_path = alt
        try:
            import torch

            self.head, self.meta = load_head(head_path)
            self.torch = torch
            self.model = torch.hub.load("facebookresearch/dinov2", self.meta["backbone"])
            self.model.eval()
            torch.set_num_threads(4)
            log.info("DINO head loaded from %s", head_path)
        except Exception as e:  # torch missing / no internet for hub on first run
            log.warning("DINO head disabled: %s", e)
            self.model = None

    @property
    def available(self) -> bool:
        return self.model is not None

    def score(self, frames: list[np.ndarray], bboxes: list, max_frames: int = 3) -> float | None:
        if not self.available:
            return None
        pairs = [(f, b) for f, b in zip(frames, bboxes) if b is not None][:max_frames]
        if not pairs:
            return None
        x = np.stack([preprocess(f, b, self.meta["crop_expansion"], self.meta["img_size"]) for f, b in pairs])
        with self.torch.inference_mode():
            feats = self.model(self.torch.from_numpy(x)).numpy()
        if isinstance(self.head, LinearHead):
            p = self.head.p_live(feats)
        else:
            p = self.head.predict_proba(feats)[:, list(self.head.classes_).index(1)]
        return float(np.median(p))
