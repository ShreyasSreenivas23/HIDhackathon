"""FastAPI decision service. Run:  uvicorn app.main:app --port 8000   (from api/)"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .challenge import ChallengeManager, events_in_order, server_turn_check
from .config import settings
from .decision import Evidence, Outcome, decide
from .imaging import yaw_proxy
from .schemas import (
    Challenge, ChallengeRequest, EnrollRequest, EnrollResponse, Identity, Liveness, OtpRequest, OtpVerify,
    Quality, VerifyRequest, VerifyResponse,
)
from .stub import stub_verify

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("api")

app = FastAPI(title="HID Spot-a-Spoof", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

pipeline = None
if not settings.stub_mode:
    from .pipeline import Pipeline

    pipeline = Pipeline(settings)
    pipeline.warmup()
    log.info("components: %s  face backend: %s", pipeline.components(), pipeline.faces.name)
challenges = ChallengeManager()
th = settings.thresholds


def _require_engine():
    if not pipeline.faces.available:
        raise HTTPException(503, "Face engine unavailable (insightface not installed). Use the Docker image or STUB_MODE=1.")


def _response(outcome: Outcome, ev: Evidence, timings: dict, challenge: Challenge | None = None) -> VerifyResponse:
    resp = VerifyResponse(
        decision=outcome.decision,
        confidence=outcome.confidence,
        reasons=outcome.reasons,
        message=outcome.message,
        liveness=Liveness(score=None if ev.liveness is None else round(ev.liveness, 4),
                          signals={k: None if v is None else round(v, 4) for k, v in ev.signals.items()}),
        identity=Identity(user_id=ev.identity_user,
                          similarity=None if ev.identity_sim is None else round(ev.identity_sim, 4),
                          matched=ev.identity_sim is not None and ev.identity_sim >= th.identity_hi),
        quality=Quality(ok=not ev.quality_issues and ev.face_found and not ev.multiple_faces, issues=ev.quality_issues),
        challenge=challenge,
        timings_ms=timings,
    )
    # structured log line -> CloudWatch Logs in cloud mode (metric filters on decision / total ms)
    log.info("DECISION %s", json.dumps({"decision": resp.decision, "reasons": resp.reasons,
                                        "liveness": resp.liveness.score, "signals": resp.liveness.signals,
                                        "sim": resp.identity.similarity, "user": resp.identity.user_id,
                                        "quality": ev.quality_metrics, "total_ms": timings.get("total")}))
    return resp


def _debug_dump(kind: str, analysis, resp: VerifyResponse, extra: dict | None = None) -> None:
    """DEBUG_DUMP_DIR=<dir>: save each request's frames + response locally (for tuning thresholds). Dev only."""
    d = os.getenv("DEBUG_DUMP_DIR")
    if not d or analysis is None or not analysis.frames:
        return

    def _save():
        from .imaging import encode_jpeg

        out = Path(d) / f"{int(time.time() * 1000)}_{kind}_{resp.decision}"
        out.mkdir(parents=True, exist_ok=True)
        for i, f in enumerate(analysis.frames):
            (out / f"frame_{i}.jpg").write_bytes(encode_jpeg(f, 92))
        (out / "response.json").write_text(resp.model_dump_json(indent=2))
        if extra:
            (out / "challenge.json").write_text(json.dumps(extra, indent=2))

    threading.Thread(target=_save, daemon=True).start()


def _audit(kind: str, analysis, resp: VerifyResponse, extra: dict | None = None) -> None:
    """Optional: store best frame + decision in S3 (background thread, never blocks the response)."""
    _debug_dump(kind, analysis, resp, extra)
    if not settings.audit_bucket or analysis is None or analysis.best_idx is None:
        return

    def _put():
        try:
            import boto3

            from .imaging import encode_jpeg

            s3 = boto3.client("s3", region_name=settings.aws_region)
            key = f"audit/{time.strftime('%Y/%m/%d')}/{int(time.time() * 1000)}_{kind}_{resp.decision}"
            s3.put_object(Bucket=settings.audit_bucket, Key=key + ".jpg", Body=encode_jpeg(analysis.frames[analysis.best_idx]))
            s3.put_object(Bucket=settings.audit_bucket, Key=key + ".json", Body=resp.model_dump_json().encode())
        except Exception as e:
            log.warning("audit upload failed: %s", e)

    threading.Thread(target=_put, daemon=True).start()


@app.get("/health")
def health():
    return {
        "ok": True,
        "stub_mode": settings.stub_mode,
        "components": pipeline.components() if pipeline else {},
        "enrolled": len(pipeline.store.all()) if pipeline else 0,
        "thresholds": th.to_dict(),
    }


@app.post("/enroll", response_model=EnrollResponse)
def enroll(req: EnrollRequest):
    if settings.stub_mode:
        return EnrollResponse(ok=True, user_id=req.user_id, frames_used=len(req.frames), quality=0.9)
    _require_engine()
    ok, n, q, msg = pipeline.enroll(req.user_id, req.frames)
    return EnrollResponse(ok=ok, user_id=req.user_id, frames_used=n, quality=round(q, 3), message=msg)


@app.get("/users")
def users():
    return {"users": sorted(pipeline.store.all()) if pipeline else ["demo_user"]}


@app.delete("/users/{user_id}")
def delete_user(user_id: str):
    if pipeline:
        pipeline.store.delete(user_id)
    return {"ok": True}


@app.post("/verify", response_model=VerifyResponse)
def verify(req: VerifyRequest, force: str | None = None):
    if settings.stub_mode:
        return stub_verify(force)
    _require_engine()
    a = pipeline.analyze(req.frames, req.claimed_user_id)
    t0 = time.perf_counter()
    outcome = decide(a.evidence, th)
    a.timings["decision"] = round((time.perf_counter() - t0) * 1000, 2)
    ch = None
    if outcome.needs_challenge:
        p = challenges.create(a.evidence.identity_user, a.evidence.claimed)
        ch = Challenge(nonce=p.nonce, actions=p.actions)
    resp = _response(outcome, a.evidence, a.timings, ch)
    _audit("verify", a, resp)
    return resp


@app.post("/challenge", response_model=VerifyResponse)
def challenge(req: ChallengeRequest):
    if settings.stub_mode:
        return stub_verify("ALLOW")
    _require_engine()
    p = challenges.pop(req.nonce)
    if p is None:
        raise HTTPException(status_code=410, detail="challenge expired or unknown")
    claimed = p.candidate_user if p.claimed else None
    # challenge frames contain head turns on purpose -> skip the frontal-pose quality gate
    a = pipeline.analyze(req.frames, claimed, check_quality=False)
    ev = a.evidence
    events = [e.model_dump() for e in req.client_events]
    kps = [f.kps if f is not None else None for f in a.faces]
    ok_events = events_in_order(p.actions, events)
    ok_turns = server_turn_check(p.actions, kps, th.challenge_yaw)
    # Liveness proof on the challenge = the gestures themselves:
    #  * random action ORDER, done live within 10 s       -> defeats pre-recorded video
    #  * server-side head-turn check (ok_turns): yaw_proxy = (nose - eye_mid) / eye_dist is a RATIO, so
    #    tilting a flat photo barely changes it, while a real turn moves it by 0.25+  -> defeats photos/screens
    #  * device detector still applies (ok_device)
    # Texture (MiniFAS) and nose-parallax are NOT required: measured on real kiosk webcams both were too noisy
    # (MiniFAS 0.05 for a real person; YuNet 5-point parallax indistinguishable from landmark noise).
    # A clearly flat face is still a hard fail.
    plan = ev.signals.get("planarity")
    flat = plan is not None and plan < th.planar_hard
    ok_live = not flat
    dev = ev.signals.get("device")
    tex = ev.signals.get("minifas")
    ok_device = (dev is None or dev >= th.device_hard) and (tex is None or tex >= th.texture_hard)
    same_person = p.candidate_user is None or ev.identity_user == p.candidate_user
    ok_id = ev.identity_sim is not None and ev.identity_sim >= th.identity_hi and same_person

    if not ev.face_found:
        outcome = Outcome("DENY", 0.9, ["challenge_failed", "no_face"], "Challenge failed — use OTP instead")
    elif ok_events and ok_turns and ok_live and ok_device and ok_id:
        outcome = Outcome("ALLOW", round(min(0.9, 0.5 + ev.identity_sim / 2), 3), [], "Verified after additional check")
    else:
        reasons = ["challenge_failed"]
        reasons += [] if ok_events else ["actions_not_completed"]
        reasons += [] if ok_turns else ["head_turn_not_seen"]
        reasons += [] if ok_device else (["phone_detected"] if dev is not None and dev < th.device_hard else ["screen_texture"])
        reasons += [] if ok_live else ["planar_motion"]
        reasons += [] if ok_id else ["identity_uncertain"]
        outcome = Outcome("DENY", 0.8, reasons, "Challenge failed — use OTP instead")
    resp = _response(outcome, ev, a.timings)
    _audit("challenge", a, resp, {"actions": p.actions, "client_events": events,
                                  "yaws": [None if k is None else round(yaw_proxy(k), 3) for k in kps]})
    return resp


@app.post("/otp/request")
def otp_request(req: OtpRequest):
    # Simulated: production would publish via Amazon SNS to the enrolled phone number.
    return {"ok": True, "demo_code": challenges.otp_request(req.user_id)}


@app.post("/otp/verify", response_model=VerifyResponse)
def otp_verify(req: OtpVerify):
    ok = challenges.otp_verify(req.user_id, req.code)
    ev = Evidence(identity_user=req.user_id)
    outcome = Outcome("ALLOW", 1.0, [], "Verified by one-time code") if ok else Outcome("DENY", 1.0, ["otp_invalid"], "Invalid or expired code")
    return _response(outcome, ev, {"total": 0.0})
