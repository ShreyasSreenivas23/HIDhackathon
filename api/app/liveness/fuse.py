"""Fuse signals into one liveness score + human-readable reasons.

v1: weighted mean of soft signals, device acts as a hard veto (min).
v2 (optional, after eval): replace with a logistic regression trained on eval outputs.
"""
from __future__ import annotations

REASON_FOR = {
    "minifas": "texture_spoof",
    "dino": "texture_spoof",
    "planarity": "planar_motion",
    "device": "phone_detected",
}


def fuse(signals: dict[str, float | None], weights: dict[str, float], flag_below: float) -> tuple[float | None, list[str]]:
    soft = {k: v for k, v in signals.items() if v is not None and k in weights}
    score = None
    if soft:
        wsum = sum(weights[k] for k in soft)
        score = sum(weights[k] * v for k, v in soft.items()) / wsum
    dev = signals.get("device")
    if dev is not None and dev < 0.5:
        score = dev if score is None else min(score, dev)
    reasons: list[str] = []
    for k, v in signals.items():
        if v is not None and v < flag_below:
            r = REASON_FOR.get(k, f"{k}_spoof")
            if r not in reasons:
                reasons.append(r)
    return score, reasons
