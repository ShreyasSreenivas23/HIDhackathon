"""Screen-device detector: is the face displayed INSIDE a phone / laptop / tv? (YOLO COCO classes)

Catches screen replay (photo) and video replay when any part of the device is visible.
Score = 1 - max confidence of a device box that contains the face (1.0 = no device around the face).
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from ..imaging import intersection_over_first

log = logging.getLogger(__name__)

DEVICE_CLASSES = {62: "tv", 63: "laptop", 67: "cell phone"}


class DeviceDetector:
    def __init__(self, model_path: Path, conf: float = 0.25, containment: float = 0.6):
        self.conf, self.containment = conf, containment
        self.model = None
        try:
            from ultralytics import YOLO

            # if the file is missing, ultralytics downloads the named weights automatically
            self.model = YOLO(str(model_path) if model_path.exists() else model_path.name)
        except Exception as e:
            log.warning("Device detector disabled: %s", e)

    @property
    def available(self) -> bool:
        return self.model is not None

    def detect(self, img_bgr: np.ndarray) -> list[dict]:
        res = self.model.predict(img_bgr, classes=list(DEVICE_CLASSES), conf=self.conf, imgsz=640, verbose=False)[0]
        out = []
        for box, c, k in zip(res.boxes.xyxy.tolist(), res.boxes.conf.tolist(), res.boxes.cls.tolist()):
            out.append({"bbox": box, "conf": float(c), "label": DEVICE_CLASSES.get(int(k), str(k))})
        return out

    def score(self, img_bgr: np.ndarray, face_bbox) -> float | None:
        if not self.available or face_bbox is None:
            return None
        worst = 0.0
        for d in self.detect(img_bgr):
            if intersection_over_first(face_bbox, d["bbox"]) >= self.containment:
                worst = max(worst, d["conf"])
        return float(1.0 - worst)
