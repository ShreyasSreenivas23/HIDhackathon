"""MiniFASNet-V2-SE passive texture model (face-antispoof-onnx, Apache-2.0, trained on CelebA-Spoof).

Preprocessing (matches upstream src/inference/preprocess.py):
  RGB -> square crop (bbox * 1.5) -> letterbox 128 -> /255 -> CHW.  Output logits [real, spoof].
"""
from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from ..imaging import letterbox, square_crop

log = logging.getLogger(__name__)


class MiniFAS:
    """Measured on real kiosk webcam bursts (warm dim hall light): single-crop, per-frame scores swing 0.01-0.99
    for the same still person. So we average logits over two crop scales (1.5x = upstream default, 2.7x =
    MiniFASNet-V2 original context) and take the MEDIAN over frames. Treat it as a soft signal only — the
    decision engine never hard-DENIES on texture alone."""

    def __init__(self, model_path: Path, expansions: tuple[float, ...] = (1.5, 2.7), size: int = 128, logit_offset: float = 0.0):
        self.expansions, self.size, self.logit_offset = expansions, size, logit_offset
        self.session = None
        if not model_path.exists():
            log.warning("MiniFAS model missing at %s (run scripts/download_models.py)", model_path)
            return
        import onnxruntime as ort

        self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    @property
    def available(self) -> bool:
        return self.session is not None

    def _prep(self, rgb: np.ndarray, bbox, expansion: float) -> np.ndarray:
        crop = square_crop(rgb, bbox, expansion)
        return letterbox(crop, self.size).transpose(2, 0, 1).astype(np.float32) / 255.0

    def score(self, frames: list[np.ndarray], bboxes: list) -> float | None:
        """P(live) = sigmoid(median over frames of the crop-averaged logit). bboxes[i] is None if no face."""
        if not self.available:
            return None
        pairs = [(cv2.cvtColor(f, cv2.COLOR_BGR2RGB), b) for f, b in zip(frames, bboxes) if b is not None]
        if not pairs:
            return None
        x = np.stack([self._prep(rgb, b, e) for rgb, b in pairs for e in self.expansions])
        try:
            logits = self.session.run(None, {self.input_name: x})[0]
        except Exception:  # model exported with fixed batch=1
            logits = np.concatenate([self.session.run(None, {self.input_name: x[i : i + 1]})[0] for i in range(len(x))])
        diff = (logits[:, 0] - logits[:, 1]).reshape(len(pairs), len(self.expansions)).mean(axis=1)
        z = float(np.median(diff)) - self.logit_offset
        return float(1.0 / (1.0 + np.exp(-z)))
