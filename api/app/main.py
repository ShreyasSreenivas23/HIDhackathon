"""FastAPI decision service. Run:  uvicorn app.main:app --port 8000   (from api/)"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .access import AccessRepository, schedule_from_payload, schedule_payload
from .aws_services import AwsServices, Metrics
from .challenge import ChallengeManager, events_in_order, server_turn_check
from .config import settings
from .decision import Evidence, Outcome, decide
from .imaging import yaw_proxy
from .schemas import (
    AccessScheduleRequest, AccessScheduleResponse, Challenge, ChallengeRequest, EnrollRequest, EnrollResponse,
    Identity, Liveness, OfflineStatus, OfflineSyncRequest, OfflineVerifyRequest, OtpRequest, OtpVerify,
    Quality, ScheduleStatus, UserProfile, UserUpdateRequest, VerifyRequest, VerifyResponse,
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
stub_profiles: dict[str, dict] = {}
challenges = ChallengeManager()
aws = AwsServices(settings.aws_region)
metrics = Metrics()
access = AccessRepository(settings.sqlite_path)
th = settings.thresholds


def _require_engine():
    if not pipeline.faces.available:
        raise HTTPException(503, "Face engine unavailable (insightface not installed). Use the Docker image or STUB_MODE=1.")


def _schedule_status(user_id: str | None, at: datetime | None = None) -> ScheduleStatus | None:
    if not user_id:
        return None
    schedule = access.get_schedule(user_id)
    if schedule is None:
        return None
    evaluated_at = at or datetime.now(timezone.utc)
    authorized, reason, zone = schedule.evaluate(evaluated_at)
    return ScheduleStatus(authorized=authorized, reason=reason, timezone=zone, evaluated_at=evaluated_at)


def _profile(user_id: str | None):
    if not user_id:
        return None
    return stub_profiles.get(user_id) if settings.stub_mode else pipeline.store.profile(user_id)


def _profile_reason(user_id: str | None, at: datetime | None = None) -> str | None:
    profile = _profile(user_id)
    if profile is None:
        return "unknown_person"
    if not profile["active"]:
        return "user_inactive"
    if profile["expires_at"] and datetime.fromisoformat(profile["expires_at"]).astimezone(timezone.utc) <= (at or datetime.now(timezone.utc)):
        return "enrollment_expired"
    return None


def _user_profiles():
    return list(stub_profiles.values()) if settings.stub_mode else pipeline.store.profiles()


def _apply_schedule(outcome: Outcome, user_id: str | None, *, at: datetime | None = None, enforce_missing: bool = True) -> tuple[Outcome, ScheduleStatus | None]:
    status = _schedule_status(user_id, at)
    if status is None:
        if enforce_missing and outcome.decision == "ALLOW":
            return Outcome("DENY", 1.0, ["no_campus_permission"], "Access denied — no campus permission"), None
        return outcome, None
    if status.authorized:
        return outcome, status
    if outcome.decision == "REQUEST_VERIFICATION":
        return Outcome("DENY", 1.0, [status.reason], f"Access denied — {status.reason.replace('_', ' ')}"), status
    return Outcome("DENY", 1.0, [status.reason], f"Access denied — {status.reason.replace('_', ' ')}"), status


def _response(
    outcome: Outcome,
    ev: Evidence,
    timings: dict,
    challenge: Challenge | None = None,
    schedule: ScheduleStatus | None = None,
    offline: OfflineStatus | None = None,
) -> VerifyResponse:
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
        schedule=schedule,
        offline=offline,
    )
    metrics.record(resp)
    aws.audit("decision", resp)
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
    pending = access.event_status()["pending"]
    return {
        "ok": True,
        "stub_mode": settings.stub_mode,
        "components": pipeline.components() if pipeline else {},
        "enrolled": len(_user_profiles()),
        "thresholds": th.to_dict(),
        "connectivity": "online",
        "aws": aws.enabled,
        "offline_events_pending": pending,
        "schedules": [schedule_payload(item) for item in access.all_schedules()],
    }


@app.get("/metrics")
def get_metrics():
    """Decision counts, top reasons and server latency percentiles since startup."""
    return metrics.snapshot() | {"aws": aws.enabled}


@app.get("/access/schedules", response_model=list[AccessScheduleResponse])
def list_schedules():
    return [AccessScheduleResponse(**schedule_payload(item)) for item in access.all_schedules()]


@app.get("/access/schedules/{user_id}", response_model=AccessScheduleResponse)
def get_schedule(user_id: str):
    schedule = access.get_schedule(user_id)
    if schedule is None:
        raise HTTPException(404, "No campus permission configured")
    return AccessScheduleResponse(**schedule_payload(schedule))


@app.put("/access/schedules/{user_id}", response_model=AccessScheduleResponse)
def put_schedule(user_id: str, req: AccessScheduleRequest):
    try:
        payload = req.model_dump()
        schedule = schedule_from_payload(user_id, payload)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    access.put_schedule(schedule)
    return AccessScheduleResponse(**schedule_payload(schedule))


@app.delete("/access/schedules/{user_id}")
def delete_schedule(user_id: str):
    access.delete_schedule(user_id)
    return {"ok": True}


@app.post("/enroll", response_model=EnrollResponse)
def enroll(req: EnrollRequest):
    expires_at = (datetime.now(timezone.utc) + timedelta(seconds=req.expires_in_seconds)).isoformat() if req.expires_in_seconds else None
    if settings.stub_mode:
        stub_profiles[req.user_id] = {"user_id": req.user_id, "authority": req.authority, "active": req.active,
                                      "expires_at": expires_at, "created_at": datetime.now(timezone.utc).isoformat()}
        return EnrollResponse(ok=True, user_id=req.user_id, frames_used=len(req.frames), quality=0.9)
    _require_engine()
    ok, n, q, msg = pipeline.enroll(req.user_id, req.frames, authority=req.authority, active=req.active, expires_at=expires_at)
    if ok:
        import base64 as _b64

        aws.store_enrolled_face(req.user_id, _b64.b64decode(req.frames[0].split(",", 1)[-1]))
    return EnrollResponse(ok=ok, user_id=req.user_id, frames_used=n, quality=round(q, 3), message=msg)


@app.get("/users")
def users():
    return {"users": sorted(item["user_id"] for item in _user_profiles())}


@app.get("/users/details", response_model=list[UserProfile])
def user_details():
    return [UserProfile(**item) for item in _user_profiles()]


@app.patch("/users/{user_id}", response_model=UserProfile)
def update_user(user_id: str, req: UserUpdateRequest):
    current = _profile(user_id)
    if current is None:
        raise HTTPException(404, "User is not enrolled")
    expires_at = current["expires_at"]
    if req.expires_in_seconds is not None and req.expires_in_seconds > 0:
        expires_at = (datetime.now(timezone.utc) + timedelta(seconds=req.expires_in_seconds)).isoformat()
    elif req.expires_in_seconds == 0:
        expires_at = None
    if settings.stub_mode:
        current.update({"authority": req.authority or current["authority"],
                        "active": current["active"] if req.active is None else req.active,
                        "expires_at": expires_at})
        return UserProfile(**current)
    return UserProfile(**pipeline.store.update_profile(user_id, authority=req.authority, active=req.active, expires_at=expires_at))


@app.delete("/users/{user_id}")
def delete_user(user_id: str):
    if pipeline:
        pipeline.store.delete(user_id)
    elif settings.stub_mode:
        stub_profiles.pop(user_id, None)
    return {"ok": True}


@app.post("/verify", response_model=VerifyResponse)
def verify(req: VerifyRequest, force: str | None = None):
    if settings.stub_mode:
        identity_user = next(iter(stub_profiles), None)
        response = stub_verify(force, identity_user)
        profile_reason = _profile_reason(response.identity.user_id)
        if profile_reason and response.decision == "ALLOW":
            response.decision, response.reasons, response.message = "DENY", [profile_reason], f"Access denied — {profile_reason.replace('_', ' ')}"
        status = _schedule_status(response.identity.user_id)
        if status is not None and not status.authorized:
            response.decision = "DENY"
            response.reasons = [status.reason]
            response.message = f"Access denied — {status.reason.replace('_', ' ')}"
        response.schedule = status
        return response
    _require_engine()
    a = pipeline.analyze(req.frames, req.claimed_user_id)
    ev0 = a.evidence
    if (aws.rekognition_enabled and ev0.identity_user and ev0.identity_sim is not None
            and th.identity_lo <= ev0.identity_sim < th.identity_hi and a.best_idx is not None):
        from .imaging import encode_jpeg

        t_rek = time.perf_counter()
        rek = aws.compare_with_enrolled(ev0.identity_user, encode_jpeg(a.frames[a.best_idx]))
        a.timings["rekognition"] = round((time.perf_counter() - t_rek) * 1000, 1)
        if rek is not None:
            ev0.signals["rekognition"] = rek
            if rek >= aws.rek_min_similarity:      # AWS confirms the uncertain match -> treat as confident
                ev0.identity_sim = th.identity_hi
    t0 = time.perf_counter()
    outcome = decide(a.evidence, th)
    profile_reason = _profile_reason(a.evidence.identity_user)
    if profile_reason and outcome.decision == "ALLOW":
        outcome = Outcome("DENY", 1.0, [profile_reason], f"Access denied — {profile_reason.replace('_', ' ')}")
    a.timings["decision"] = round((time.perf_counter() - t0) * 1000, 2)
    ch = None
    if outcome.needs_challenge:
        p = challenges.create(a.evidence.identity_user, a.evidence.claimed)
        ch = Challenge(nonce=p.nonce, actions=p.actions)
    outcome, schedule = _apply_schedule(outcome, a.evidence.identity_user)
    resp = _response(outcome, a.evidence, a.timings, ch if outcome.decision == "REQUEST_VERIFICATION" else None, schedule)
    _audit("verify", a, resp)
    return resp


@app.post("/challenge", response_model=VerifyResponse)
def challenge(req: ChallengeRequest):
    if settings.stub_mode:
        response = stub_verify("ALLOW", next(iter(stub_profiles), None))
        status = _schedule_status(response.identity.user_id)
        if status is not None and not status.authorized:
            response.decision = "DENY"
            response.reasons = [status.reason]
            response.message = f"Access denied — {status.reason.replace('_', ' ')}"
        response.schedule = status
        return response
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
    outcome, schedule = _apply_schedule(outcome, ev.identity_user)
    resp = _response(outcome, ev, a.timings, schedule=schedule)
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
    outcome, schedule = _apply_schedule(outcome, req.user_id, enforce_missing=settings.stub_mode is False)
    return _response(outcome, ev, {"total": 0.0}, schedule=schedule)


@app.post("/offline/verify", response_model=VerifyResponse)
def offline_verify(req: OfflineVerifyRequest):
    occurred_at = req.occurred_at or datetime.now(timezone.utc)
    schedule = access.get_schedule(req.user_id)
    if schedule is None:
        outcome = Outcome("DENY", 1.0, ["no_cached_permission"], "Access denied — no cached campus permission")
        return _response(outcome, Evidence(identity_user=req.user_id), {"total": 0.0},
                         offline=OfflineStatus(local=True, sync_status="not_applicable"))
    allowed, reason, zone = schedule.evaluate(occurred_at)
    ev = Evidence(
        identity_user=req.user_id,
        identity_sim=req.identity_similarity,
        liveness=req.liveness,
        signals=req.signals,
    )
    if not allowed:
        outcome = Outcome("DENY", 1.0, [reason], f"Access denied — {reason.replace('_', ' ')}")
    elif req.identity_similarity is None or req.identity_similarity < th.identity_hi:
        outcome = Outcome("DENY", 1.0, ["offline_identity_uncertain"], "Access denied — offline identity is uncertain")
    elif req.liveness is None or req.liveness < th.liveness_hi:
        outcome = Outcome("DENY", 1.0, ["offline_liveness_uncertain"], "Access denied — offline liveness is uncertain")
    elif req.signals.get("device") is not None and req.signals["device"] < th.device_hard:
        outcome = Outcome("DENY", 1.0, ["phone_detected"], "Presentation attack detected")
    else:
        outcome = Outcome("ALLOW", min(req.liveness, req.identity_similarity), [], "Access granted — scheduled access")
    event_payload = req.model_dump(mode="json")
    event_payload["schedule_updated_at"] = schedule.updated_at if schedule else None
    event_id = access.queue_event(req.user_id, outcome.decision, occurred_at, event_payload)
    status = ScheduleStatus(authorized=allowed, reason=reason, timezone=zone, evaluated_at=occurred_at)
    return _response(outcome, ev, {"total": 0.0}, schedule=status,
                     offline=OfflineStatus(local=True, sync_status="pending", event_id=event_id))


@app.get("/offline/status")
def offline_status():
    return {"local": True, "sync": access.event_status()}


@app.post("/offline/sync")
def offline_sync(req: OfflineSyncRequest):
    accepted, rejected = [], []
    for event in req.events:
        if event.decision == "ALLOW":
            schedule = access.get_schedule(event.user_id or "")
            valid = schedule is not None and schedule.evaluate(event.occurred_at)[0]
            cached_updated = event.payload.get("schedule_updated_at")
            if valid and cached_updated is not None and float(cached_updated) < schedule.updated_at:
                valid = False
            if not valid:
                rejected.append({"event_id": event.event_id, "reason": "stale_or_invalid_permission"})
                continue
        access.queue_event(event.user_id, event.decision, event.occurred_at, event.payload, event.event_id)
        accepted.append(event.event_id)
    access.mark_synced(accepted)
    return {"accepted": accepted, "rejected": rejected, "sync": access.event_status()}
