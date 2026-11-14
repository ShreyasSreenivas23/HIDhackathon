"""Planarity / parallax check from facial keypoints across the burst.

Idea: eyes + mouth corners are roughly coplanar; the NOSE TIP sticks out of that plane.
Fit a homography on the 4 "planar" points between a reference frame and each other frame, then
measure how far the nose lands from where the homography predicts ("nose parallax").
  * Real face rotating even a few degrees: nose parallax of several px  -> LIVE evidence.
  * Flat photo / phone screen that is tilted or waved: the 4 points deform strongly (perspective)
    but the nose follows the plane exactly                               -> SPOOF evidence.
  * Neither (person/photo held perfectly still, or pure translation)    -> None (no opinion).

Limitation (say it in Q&A): a VIDEO replay of a head that turns contains real parallax inside the
video, so this check mainly catches static photos/screens. Video replay is caught by the device
detector, texture models and the randomized challenge.

Synthetic check (tests/test_decision.py, 90 px eye distance): 6° yaw -> nose parallax ~4 px,
12° -> ~8 px; plane deformation stays < 1 px until ~20°. Thresholds below are first guesses —
calibrate with eval/ on real live vs. photo captures (SCRFD landmark jitter is ~1–2 px).
"""
from __future__ import annotations

import cv2
import numpy as np

PLANAR_IDX = [0, 1, 3, 4]   # eyes + mouth corners (insightface 5-point order)
NOSE_IDX = 2


def _deformation(ref: np.ndarray, cur: np.ndarray) -> float:
    """Non-similarity change of the 4 planar points (perspective / out-of-plane motion), in px."""
    m, _ = cv2.estimateAffinePartial2D(ref, cur)
    if m is None:
        return 0.0
    pred = ref @ m[:, :2].T + m[:, 2]
    return float(np.linalg.norm(pred - cur, axis=1).mean())


def _nose_parallax(ref: np.ndarray, cur: np.ndarray) -> float | None:
    h, _ = cv2.findHomography(ref[PLANAR_IDX], cur[PLANAR_IDX], 0)
    if h is None:
        return None
    pred = cv2.perspectiveTransform(ref[NOSE_IDX].reshape(1, 1, 2).astype(np.float32), h).reshape(2)
    return float(np.linalg.norm(pred - cur[NOSE_IDX]))


def planarity_score(
    kps_list: list,
    live_parallax: float = 0.06,   # nose parallax / eye distance = clear 3D (YuNet noise reaches ~0.025)
    spoof_deform: float = 0.12,    # plane deformation / eye distance = plane clearly tilted (noise reaches ~0.05)
) -> float | None:
    """kps_list: per-frame 5x2 keypoints (None if no face). Returns P(live) in [0,1] or None."""
    pts = [np.asarray(k, dtype=np.float32) for k in kps_list if k is not None]
    if len(pts) < 2:
        return None
    ref = pts[0]
    eye_dist = float(np.linalg.norm(ref[1] - ref[0])) + 1e-6
    ns, ds = [], []
    for cur in pts[1:]:
        par = _nose_parallax(ref, cur)
        if par is not None:
            ns.append(par / eye_dist)
            ds.append(_deformation(ref[PLANAR_IDX], cur[PLANAR_IDX]) / eye_dist)
    if len(ns) < 2:
        return None
    # MEDIAN over frames: a single noisy landmark frame must not decide (max() amplified noise)
    n, d = float(np.median(ns)), float(np.median(ds))
    if n >= live_parallax:
        return min(1.0, 0.5 + 5.0 * n)                  # 0.06 -> 0.8, 0.10 -> 1.0
    if d >= spoof_deform:
        return 0.5 * min(1.0, n / (2.0 * d))            # plane clearly tilted, nose followed it -> ~0
    return None
