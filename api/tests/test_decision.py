import numpy as np

from app.challenge import events_in_order, server_turn_check
from app.config import Thresholds
from app.decision import Evidence, decide
from app.liveness.fuse import fuse
from app.liveness.planarity import planarity_score

TH = Thresholds()


def ev(**kw):
    base = dict(liveness=0.9, identity_user="alice", identity_sim=0.6, signals={"minifas": 0.9, "device": 1.0})
    base.update(kw)
    return Evidence(**base)


def test_allow():
    assert decide(ev(), TH).decision == "ALLOW"


def test_phone_is_hard_deny():
    o = decide(ev(signals={"device": 0.1}, liveness=0.9), TH)
    assert o.decision == "DENY" and o.reasons[0] == "phone_detected"


def test_low_liveness_deny():
    o = decide(ev(liveness=0.1, liveness_reasons=["texture_spoof"]), TH)
    assert o.decision == "DENY" and "texture_spoof" in o.reasons


def test_unknown_person_deny():
    assert decide(ev(identity_sim=0.1), TH).reasons == ["unknown_person"]
    assert decide(ev(identity_sim=0.1, claimed=True), TH).reasons == ["identity_mismatch"]


def test_gray_zone_requests_challenge():
    o = decide(ev(liveness=0.6), TH)
    assert o.decision == "REQUEST_VERIFICATION" and o.needs_challenge
    o = decide(ev(identity_sim=0.33), TH)
    assert o.decision == "REQUEST_VERIFICATION" and "identity_uncertain" in o.reasons


def test_quality_and_faces():
    assert decide(ev(quality_issues=["too_dark"]), TH).decision == "REQUEST_VERIFICATION"
    assert decide(ev(face_found=False), TH).reasons == ["no_face"]
    assert decide(ev(multiple_faces=True), TH).reasons == ["multiple_faces"]


def test_fuse():
    s, r = fuse({"minifas": 0.9, "dino": 0.8, "planarity": None, "device": 1.0}, TH.weights, 0.4)
    assert 0.8 < s < 0.9 and r == []
    s, r = fuse({"minifas": 0.9, "device": 0.05}, TH.weights, 0.4)
    assert s == 0.05 and r == ["phone_detected"]


def _face(yaw_shift=0.0, depth=1.0):
    # camera coords: x = image right, z = away from camera. Nose sticks out TOWARD the camera (-z).
    # eyes, nose, mouth corners; `depth`=0 -> flat photo
    pts = np.array([[-30, -20, 0], [30, -20, 0], [0, 5, -25 * depth], [-22, 30, 0], [22, 30, 0]], dtype=np.float32)
    c, s = np.cos(yaw_shift), np.sin(yaw_shift)
    rot = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
    p = pts @ rot.T
    f = 600.0
    z = p[:, 2] + 400
    return np.stack([f * p[:, 0] / z + 320, f * p[:, 1] / z + 240], axis=1)


def test_planarity_real_vs_flat():
    # small natural head motion is enough to prove 3D
    real = planarity_score([_face(y, 1.0) for y in [0, 0.05, 0.1, 0.2]])
    assert real is not None and real > 0.8
    # a photo tilted/waved strongly: the plane deforms but the nose follows it -> spoof
    flat = planarity_score([_face(y, 0.0) for y in [0, 0.2, 0.4, 0.6]])
    assert flat is not None and flat < 0.2
    # no motion / tiny motion of a flat image -> no opinion (never guess)
    assert planarity_score([_face(0, 1.0)] * 4) is None
    assert planarity_score([_face(y, 0.0) for y in [0, 0.05, 0.1]]) is None


def test_challenge_helpers():
    assert events_in_order(["blink", "turn_left"], [{"action": "blink", "t_ms": 1}, {"action": "turn_left", "t_ms": 2}])
    assert not events_in_order(["blink", "turn_left"], [{"action": "turn_left", "t_ms": 1}, {"action": "blink", "t_ms": 2}])
    # user turns to THEIR left -> nose moves to IMAGE right -> positive yaw proxy (see CONTRACT.md)
    from app.imaging import yaw_proxy

    left, right = _face(-0.5), _face(0.5)
    assert yaw_proxy(left) > 0.2 and yaw_proxy(right) < -0.2
    assert server_turn_check(["turn_left"], [_face(0), left], 0.2)
    assert not server_turn_check(["turn_left"], [_face(0), right], 0.2)
    assert server_turn_check(["turn_right", "blink"], [_face(0), right], 0.2)
