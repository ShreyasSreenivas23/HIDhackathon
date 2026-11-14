"""Decision engine — pure function, unit-tested in tests/test_decision.py.

Order of rules:
 1. no face / several faces / poor quality -> REQUEST_VERIFICATION (recapture with guidance, no challenge)
 2. hard spoof evidence: phone/screen detected (device or extremely low texture score), or flat face -> DENY
 3. no enrolled match / similarity < lo           -> DENY
 4. liveness >= hi and identity >= hi             -> ALLOW
 5. anything in between (incl. low TEXTURE score) -> REQUEST_VERIFICATION + active challenge
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .config import Thresholds

MESSAGES = {
    "ALLOW": "Live person detected & identity verified",
    "DENY_SPOOF": "Presentation attack detected",
    "DENY_ID": "Identity could not be verified",
    "RECAPTURE": "Please adjust and try again",
    "CHALLENGE": "Additional verification required",
}


@dataclass
class Evidence:
    face_found: bool = True
    multiple_faces: bool = False
    quality_issues: list[str] = field(default_factory=list)
    quality_metrics: dict = field(default_factory=dict)
    liveness: float | None = None
    liveness_reasons: list[str] = field(default_factory=list)
    signals: dict[str, float | None] = field(default_factory=dict)
    identity_user: str | None = None
    identity_sim: float | None = None
    claimed: bool = False


@dataclass
class Outcome:
    decision: str
    confidence: float
    reasons: list[str]
    message: str
    needs_challenge: bool = False


def identity_confidence(sim: float | None, th: Thresholds) -> float:
    """Map cosine similarity to [0,1]: lo -> 0.5, hi -> 1.0 (linear, clipped)."""
    if sim is None:
        return 0.0
    span = max(th.identity_hi - th.identity_lo, 1e-6)
    return float(min(1.0, max(0.0, 0.5 + 0.5 * (sim - th.identity_lo) / span)))


def decide(ev: Evidence, th: Thresholds) -> Outcome:
    if not ev.face_found:
        return Outcome("REQUEST_VERIFICATION", 0.0, ["no_face"], "No face detected — look at the camera")
    if ev.multiple_faces:
        return Outcome("REQUEST_VERIFICATION", 0.0, ["multiple_faces"], "One person at a time, please")

    dev = ev.signals.get("device")
    if dev is not None and dev < th.device_hard:
        return Outcome("DENY", round(1 - dev, 3), ["phone_detected"] + [r for r in ev.liveness_reasons if r != "phone_detected"], MESSAGES["DENY_SPOOF"])

    tex, dino = ev.signals.get("minifas"), ev.signals.get("dino")
    # screen DENY only when the texture models AGREE: MiniFAS alone hit 0.005 on a real person (live test),
    # while the GB10-trained DINOv2 head said 0.78-0.90 live; on phone attacks DINOv2 gave 0.001 / 0.031.
    if tex is not None and tex < th.texture_hard and (dino is None or dino < th.dino_screen):
        return Outcome("DENY", round(1 - tex, 3), ["screen_texture"], MESSAGES["DENY_SPOOF"])

    if ev.quality_issues:
        return Outcome("REQUEST_VERIFICATION", 0.0, ["low_quality"] + ev.quality_issues, MESSAGES["RECAPTURE"])

    L = ev.liveness if ev.liveness is not None else 0.5
    # Hard spoof DENY needs GEOMETRIC evidence (flat face). The texture models alone proved unreliable on real
    # kiosk webcams (warm/dim light), so low texture only escalates to the active challenge (rule 5).
    plan = ev.signals.get("planarity")
    if L < th.liveness_lo and plan is not None and plan < th.planar_hard:
        return Outcome("DENY", round(1 - L, 3), ev.liveness_reasons or ["planar_motion"], MESSAGES["DENY_SPOOF"])

    if ev.identity_sim is None or ev.identity_sim < th.identity_lo:
        reason = "identity_mismatch" if ev.claimed else "unknown_person"
        conf = 1.0 if ev.identity_sim is None else 1 - identity_confidence(ev.identity_sim, th)
        return Outcome("DENY", round(conf, 3), [reason], MESSAGES["DENY_ID"])

    id_conf = identity_confidence(ev.identity_sim, th)
    # texture models DISAGREE (MiniFAS says screen, DINOv2 says live): never ALLOW, never DENY -> challenge.
    # Seen both ways in live data: real person (MiniFAS 0.005 / DINO 0.85) and a held-out phone attack
    # (MiniFAS 0.008 / DINO 0.99). A real person passes the gestures; a phone cannot.
    if tex is not None and tex < th.texture_hard:
        return Outcome("REQUEST_VERIFICATION", round(min(L, id_conf), 3), ["liveness_uncertain", "texture_disagreement"],
                       MESSAGES["CHALLENGE"], needs_challenge=True)
    if L >= th.liveness_hi and ev.identity_sim >= th.identity_hi:
        return Outcome("ALLOW", round(min(L, id_conf), 3), [], MESSAGES["ALLOW"])

    reasons = []
    if L < th.liveness_hi:
        reasons.append("liveness_uncertain")
        reasons += [r for r in ev.liveness_reasons if r not in reasons]
    if ev.identity_sim < th.identity_hi:
        reasons.append("identity_uncertain")
    return Outcome("REQUEST_VERIFICATION", round(min(L, id_conf), 3), reasons, MESSAGES["CHALLENGE"], needs_challenge=True)
