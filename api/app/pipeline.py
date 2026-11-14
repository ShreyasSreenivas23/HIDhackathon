"""Runs detection, liveness signals (in parallel) and identity on a burst of frames -> Evidence + timings."""
from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np

from .config import Settings
from .decision import Evidence
from .faces import DetectedFace, FaceEngine
from .imaging import decode_b64_image, face_quality, yaw_proxy
from .liveness.device import DeviceDetector
from .liveness.dino_head import DinoHead
from .liveness.fuse import fuse
from .liveness.minifas import MiniFAS
from .liveness.planarity import planarity_score
from .store import make_store, search

log = logging.getLogger(__name__)


@dataclass
class Analysis:
    evidence: Evidence
    timings: dict[str, float]
    frames: list[np.ndarray] = field(default_factory=list)
    faces: list[DetectedFace | None] = field(default_factory=list)
    best_idx: int | None = None


class _Timer:
    def __init__(self):
        self.t0 = time.perf_counter()
        self.last = self.t0
        self.marks: dict[str, float] = {}

    def mark(self, name: str):
        now = time.perf_counter()
        self.marks[name] = round((now - self.last) * 1000, 1)
        self.last = now

    def done(self) -> dict[str, float]:
        self.marks["total"] = round((time.perf_counter() - self.t0) * 1000, 1)
        return self.marks


class Pipeline:
    def __init__(self, settings: Settings):
        self.s = settings
        self.th = settings.thresholds
        self.pool = ThreadPoolExecutor(max_workers=8)
        self.faces = FaceEngine(settings.models_dir, settings.det_size)
        self.minifas = MiniFAS(settings.minifas_model) if settings.enable_minifas else None
        self.dino = DinoHead(settings.dino_head) if settings.enable_dino else None
        self.device = DeviceDetector(settings.yolo_model) if settings.enable_device else None
        self.store = make_store(settings)

    def warmup(self) -> None:
        """Run every model once at startup so the first real request isn't slow (YOLO's first call ~2 s)."""
        dummy = np.full((480, 640, 3), 127, np.uint8)
        box = [220.0, 140.0, 420.0, 380.0]
        try:
            if self.faces.available:
                self.faces.detect(dummy)
            if self.minifas and self.minifas.available:
                self.minifas.score([dummy], [box])
            if self.device and self.device.available:
                self.device.score(dummy, box)
            if self.dino and self.dino.available:
                self.dino.score([dummy], [box])
        except Exception as e:
            log.warning("warmup failed: %s", e)

    def components(self) -> dict[str, bool]:
        return {
            "detector": self.faces.available,
            "identity": self.faces.available,
            "minifas": bool(self.minifas and self.minifas.available),
            "dino": bool(self.dino and self.dino.available),
            "device": bool(self.device and self.device.available),
            "planarity": self.s.enable_planarity,
        }

    # ------------------------------------------------------------------ helpers
    def _detect_all(self, frames: list[np.ndarray]) -> tuple[list[DetectedFace | None], bool]:
        dets = list(self.pool.map(self.faces.detect, frames))
        main, multiple = [], 0
        for faces in dets:
            if not faces:
                main.append(None)
                continue
            main.append(faces[0])
            # a second face at least half the size of the main one = another person in frame
            if len(faces) > 1 and faces[1].size > 0.5 * faces[0].size:
                multiple += 1
        return main, multiple > len(frames) // 2

    def _best_frame(self, frames, faces) -> int | None:
        best, best_score = None, -1.0
        for i, (f, face) in enumerate(zip(frames, faces)):
            if face is None:
                continue
            q = face_quality(f, face.bbox)
            score = face.det_score * min(q["blur"], 300) * (1 - abs(yaw_proxy(face.kps)))
            if score > best_score:
                best, best_score = i, score
        return best

    def _quality_issues(self, img, face: DetectedFace) -> list[str]:
        q, th, issues = face_quality(img, face.bbox), self.th, []
        if q["face_px"] < th.min_face_px:
            issues.append("face_too_small")
        if q["blur"] < th.min_blur:
            issues.append("blurry")
        if q["brightness"] < th.min_brightness:
            issues.append("too_dark")
        if q["brightness"] > th.max_brightness:
            issues.append("too_bright")
        if abs(yaw_proxy(face.kps)) > th.max_abs_yaw:
            issues.append("face_not_frontal")
        return issues

    def liveness_signals(self, frames, faces, best_idx) -> dict[str, float | None]:
        bboxes = [f.bbox if f is not None else None for f in faces]
        futs = {}
        if self.minifas:
            futs["minifas"] = self.pool.submit(self.minifas.score, frames, bboxes)
        if self.dino:
            order = [best_idx] + [i for i in range(len(frames)) if i != best_idx]
            futs["dino"] = self.pool.submit(self.dino.score, [frames[i] for i in order], [bboxes[i] for i in order])
        if self.device:
            futs["device"] = self.pool.submit(self.device.score, frames[best_idx], bboxes[best_idx])
        if self.s.enable_planarity:
            futs["planarity"] = self.pool.submit(planarity_score, [f.kps if f is not None else None for f in faces])
        out = {}
        for k, fut in futs.items():
            try:
                out[k] = fut.result()
            except Exception as e:  # a broken signal must never break the kiosk
                log.exception("signal %s failed: %s", k, e)
                out[k] = None
        return out

    # ------------------------------------------------------------------ main entry points
    def analyze(self, frames_b64: list[str], claimed_user_id: str | None = None, check_quality: bool = True) -> Analysis:
        t = _Timer()
        frames = [decode_b64_image(f) for f in frames_b64]
        t.mark("decode")
        faces, multiple = self._detect_all(frames)
        t.mark("detect")
        ev = Evidence(claimed=claimed_user_id is not None)
        n_found = sum(f is not None for f in faces)
        if n_found == 0 or n_found < len(frames) / 2:
            ev.face_found = False
            return Analysis(ev, t.done(), frames, faces)
        ev.multiple_faces = multiple
        best = self._best_frame(frames, faces)
        if check_quality:
            ev.quality_issues = self._quality_issues(frames[best], faces[best])

        id_fut = self.pool.submit(self.faces.embed, frames[best], faces[best])
        ev.signals = self.liveness_signals(frames, faces, best)
        ev.liveness, ev.liveness_reasons = fuse(ev.signals, self.th.weights, self.th.signal_flag)
        t.mark("liveness")
        emb = id_fut.result()
        ev.identity_user, ev.identity_sim = search(self.store.all(), emb, claimed_user_id)
        t.mark("identity")
        return Analysis(ev, t.done(), frames, faces, best)

    def enroll(self, user_id: str, frames_b64: list[str]) -> tuple[bool, int, float, str]:
        frames = [decode_b64_image(f) for f in frames_b64]
        embs, quals = [], []
        for img in frames:
            faces = self.faces.detect(img)
            if not faces or (len(faces) > 1 and faces[1].size > 0.5 * faces[0].size):
                continue   # no face, or two people in frame
            if self._quality_issues(img, faces[0]):
                continue
            embs.append(self.faces.embed(img, faces[0]))
            quals.append(faces[0].det_score)
        if not embs:
            return False, 0, 0.0, "No usable frame: one clear, frontal, well-lit face needed"
        template = np.mean(embs, axis=0)
        template /= np.linalg.norm(template) + 1e-9
        self.store.upsert(user_id, template.astype(np.float32))
        return True, len(embs), float(np.mean(quals)), ""
