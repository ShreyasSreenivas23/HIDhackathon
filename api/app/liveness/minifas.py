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
    def __init__(self, model_path: Path, expansion: float = 1.5, size: int = 128, logit_offset: float = 0.0):
        self.expansion, self.size, self.logit_offset = expansion, size, logit_offset
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

    def _prep(self, img_bgr: np.ndarray, bbox) -> np.ndarray:
        crop = square_crop(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), bbox, self.expansion)
        return letterbox(crop, self.size).transpose(2, 0, 1).astype(np.float32) / 255.0

    def score(self, frames: list[np.ndarray], bboxes: list) -> float | None:
        """Mean P(live) over frames that have a face. bboxes[i] is None when frame i has no face."""
        if not self.available:
            return None
        batch = [self._prep(f, b) for f, b in zip(frames, bboxes) if b is not None]
        if not batch:
            return None
        x = np.stack(batch)
        try:
            logits = self.session.run(None, {self.input_name: x})[0]
        except Exception:  # model exported with fixed batch=1
            logits = np.concatenate([self.session.run(None, {self.input_name: x[i : i + 1]})[0] for i in range(len(x))])
        diff = logits[:, 0] - logits[:, 1] - self.logit_offset
        p_live = 1.0 / (1.0 + np.exp(-diff))
        return float(p_live.mean())
