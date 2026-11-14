"""STUB_MODE: fake but contract-correct responses so the kiosk UI can be built before models exist.
/verify cycles ALLOW -> DENY -> REQUEST_VERIFICATION so every screen can be exercised."""
from __future__ import annotations

import itertools
import random

from .schemas import Challenge, Identity, Liveness, Quality, VerifyResponse

_cycle = itertools.cycle(["ALLOW", "DENY", "REQUEST_VERIFICATION"])


def stub_verify(force: str | None = None) -> VerifyResponse:
    d = force or next(_cycle)
    t = {"decode": 6.0, "detect": 31.0, "liveness": 142.0, "identity": 14.0, "total": 196.0 + random.uniform(0, 80)}
    if d == "ALLOW":
        return VerifyResponse(
            decision="ALLOW", confidence=0.96, message="Live person detected & identity verified",
            liveness=Liveness(score=0.97, signals={"minifas": 0.95, "dino": 0.98, "device": 1.0, "planarity": None}),
            identity=Identity(user_id="demo_user", similarity=0.68, matched=True), timings_ms=t)
    if d == "DENY":
        return VerifyResponse(
            decision="DENY", confidence=0.93, reasons=["phone_detected", "texture_spoof"], message="Presentation attack detected",
            liveness=Liveness(score=0.07, signals={"minifas": 0.12, "dino": 0.09, "device": 0.07, "planarity": None}),
            identity=Identity(user_id="demo_user", similarity=0.61, matched=False), timings_ms=t)
    return VerifyResponse(
        decision="REQUEST_VERIFICATION", confidence=0.62, reasons=["liveness_uncertain"], message="Additional verification required",
        liveness=Liveness(score=0.62, signals={"minifas": 0.58, "dino": 0.66, "device": 1.0, "planarity": None}),
        identity=Identity(user_id="demo_user", similarity=0.55, matched=False), quality=Quality(ok=True),
        challenge=Challenge(nonce="stub01", actions=random.sample(["blink", "turn_left", "smile"], 2)), timings_ms=t)
