"""Contract-correct responses for local UI development without AI models.
Normal verification is fail-closed for unknown users and succeeds for enrolled users.
The force query parameter remains available for explicitly testing other outcomes."""
from __future__ import annotations

import random

from .schemas import Challenge, Identity, Liveness, Quality, VerifyResponse

def stub_verify(force: str | None = None, identity_user: str | None = None) -> VerifyResponse:
    d = force or ("ALLOW" if identity_user is not None else "DENY")
    t = {"decode": 6.0, "detect": 31.0, "liveness": 142.0, "identity": 14.0, "total": 196.0 + random.uniform(0, 80)}
    if d == "ALLOW":
        if identity_user is None:
            return VerifyResponse(
                decision="DENY", confidence=1.0, reasons=["unknown_person"],
                message="Person not enrolled",
                liveness=Liveness(score=0.97, signals={"minifas": 0.95, "dino": 0.98, "device": 1.0, "planarity": None}),
                identity=Identity(), timings_ms=t)
        return VerifyResponse(
            decision="ALLOW", confidence=0.96, message="Live person detected & identity verified",
            liveness=Liveness(score=0.97, signals={"minifas": 0.95, "dino": 0.98, "device": 1.0, "planarity": None}),
            identity=Identity(user_id=identity_user, similarity=0.68 if identity_user else None, matched=identity_user is not None), timings_ms=t)
    if d == "DENY":
        return VerifyResponse(
            decision="DENY", confidence=0.93, reasons=["phone_detected", "texture_spoof"], message="Presentation attack detected",
            liveness=Liveness(score=0.07, signals={"minifas": 0.12, "dino": 0.09, "device": 0.07, "planarity": None}),
            identity=Identity(user_id=identity_user, similarity=0.61 if identity_user else None, matched=False), timings_ms=t)
    return VerifyResponse(
        decision="REQUEST_VERIFICATION", confidence=0.62, reasons=["liveness_uncertain"], message="Additional verification required",
        liveness=Liveness(score=0.62, signals={"minifas": 0.58, "dino": 0.66, "device": 1.0, "planarity": None}),
        identity=Identity(user_id=identity_user, similarity=0.55 if identity_user else None, matched=False), quality=Quality(ok=True),
        challenge=Challenge(nonce="stub01", actions=random.sample(["blink", "turn_left", "smile"], 2)), timings_ms=t)
