"""Active challenge (random actions) + simulated OTP. In-memory: fine for a single kiosk / demo.
For multi-instance cloud mode, move these dicts to DynamoDB with a TTL attribute."""
from __future__ import annotations

import random
import secrets
import time
from dataclasses import dataclass, field

import numpy as np

from .imaging import yaw_proxy

ACTIONS = ["blink", "turn_left", "turn_right", "smile"]
TTL_S = 20


@dataclass
class PendingChallenge:
    nonce: str
    actions: list[str]
    candidate_user: str | None
    claimed: bool
    created: float = field(default_factory=time.time)

    @property
    def expired(self) -> bool:
        return time.time() - self.created > TTL_S


class ChallengeManager:
    def __init__(self):
        self.pending: dict[str, PendingChallenge] = {}
        self.otps: dict[str, tuple[str, float]] = {}

    def create(self, candidate_user: str | None, claimed: bool) -> PendingChallenge:
        # always one head turn (gives the server 3D-parallax evidence) + one blink/smile, in random order
        acts = [random.choice(["turn_left", "turn_right"]), random.choice(["blink", "smile"])]
        random.shuffle(acts)
        ch = PendingChallenge(secrets.token_hex(4), acts, candidate_user, claimed)
        self.pending[ch.nonce] = ch
        self._gc()
        return ch

    def pop(self, nonce: str) -> PendingChallenge | None:
        ch = self.pending.pop(nonce, None)
        return None if ch is None or ch.expired else ch

    def _gc(self):
        for k in [k for k, v in self.pending.items() if v.expired]:
            self.pending.pop(k, None)

    # ---- OTP (simulated: real deployment would send via SNS to the enrolled phone) ----
    def otp_request(self, user_id: str) -> str:
        code = f"{secrets.randbelow(10**6):06d}"
        self.otps[user_id] = (code, time.time())
        return code

    def otp_verify(self, user_id: str, code: str) -> bool:
        rec = self.otps.pop(user_id, None)
        return rec is not None and rec[0] == code and time.time() - rec[1] < 120


def events_in_order(required: list[str], events: list[dict]) -> bool:
    """Client must report the required actions in the given order (extra events allowed)."""
    i = 0
    for e in sorted(events, key=lambda e: e["t_ms"]):
        if i < len(required) and e["action"] == required[i]:
            i += 1
    return i == len(required)


def server_turn_check(required: list[str], kps_list: list, yaw_needed: float) -> bool:
    """Independent server-side check of head-turn actions from the frames (don't trust the client alone).
    blink/smile are verified client-side (blendshapes) + by the liveness score on challenge frames."""
    yaws = [yaw_proxy(k) for k in kps_list if k is not None]
    if len(yaws) < 2:
        return False
    yaws = np.array(yaws)
    # measure the turn RELATIVE to this person's own straight-ahead pose (their most frontal frame), not
    # absolute 0: people/cameras have a small natural offset (+0.04 measured), and the browser's MediaPipe
    # landmarks read turns larger than the server's YuNet ones (kiosk saw 0.25 where the server saw 0.19).
    # (Not the median: a burst with many turned frames drags the median toward the turn, which made a
    #  right-only turn also count as "left".)
    base = float(yaws[np.argmin(np.abs(yaws))])
    if "turn_left" in required and yaws.max() - base < yaw_needed:
        return False
    if "turn_right" in required and base - yaws.min() < yaw_needed:
        return False
    return True
