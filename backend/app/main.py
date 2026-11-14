import base64
import statistics
import time
from concurrent.futures import ThreadPoolExecutor

from botocore.exceptions import ClientError
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app import detectors
from app.aws_clients import (
    bedrock_step_up_instruction,
    compare_faces,
    detect_face_quality,
    fetch_enrolled_face,
    log_audit_event,
)
from app.challenge import build_challenge, verify_challenge
from app.config import settings
from app.models import (
    MetricsResponse,
    StartSessionResponse,
    StepUpRequest,
    StepUpResponse,
    VerifyRequest,
    VerifyResponse,
)
from app.scoring import decide

app = FastAPI(title="HID Hackathon Zero-Trust Liveness API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_METRICS = {
    "latencies": [],
    "allow": 0,
    "request_verification": 0,
    "deny": 0,
}


def _decode_bytes(frame_b64: str) -> bytes:
    if "," in frame_b64:
        frame_b64 = frame_b64.split(",", 1)[1]
    return base64.b64decode(frame_b64)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/start-session", response_model=StartSessionResponse)
def start_session() -> StartSessionResponse:
    session_id, token, colors, prompt_hint = build_challenge()
    return StartSessionResponse(
        session_id=session_id,
        challenge_token=token,
        flash_colors=colors,
        prompt_hint=prompt_hint,
        expires_in_seconds=settings.challenge_ttl_seconds,
    )


@app.post("/verify", response_model=VerifyResponse)
def verify(payload: VerifyRequest) -> VerifyResponse:
    start = time.perf_counter()

    try:
        verify_challenge(payload.challenge_token, payload.session_id)
    except Exception as exc:
        raise HTTPException(status_code=401, detail=f"Invalid or expired challenge token: {exc}") from exc

    try:
        enrolled = fetch_enrolled_face(payload.employee_id)
    except ClientError as exc:
        raise HTTPException(
            status_code=404,
            detail=f"Enrolled employee face not found in S3: {payload.employee_id}",
        ) from exc

    baseline_bytes = _decode_bytes(payload.baseline_b64)

    with ThreadPoolExecutor(max_workers=4) as executor:
        f_spoof_photo = executor.submit(
            detectors.photometric_spoof_score,
            payload.baseline_b64,
            payload.flash1_b64,
            payload.flash2_b64,
        )
        f_spoof_moire = executor.submit(detectors.moire_screen_score, payload.baseline_b64)
        f_identity = executor.submit(compare_faces, enrolled, baseline_bytes)
        f_liveness = executor.submit(detect_face_quality, baseline_bytes)

        photometric = f_spoof_photo.result()
        moire = f_spoof_moire.result()
        identity = f_identity.result()
        liveness = f_liveness.result()

    spoof_score = min(1.0, 0.6 * photometric + 0.4 * moire)
    decision, confidence, reasons = decide(identity, liveness, spoof_score)

    latency_ms = int((time.perf_counter() - start) * 1000)
    _METRICS["latencies"].append(latency_ms)
    if decision.value == "ALLOW":
        _METRICS["allow"] += 1
    elif decision.value == "REQUEST_VERIFICATION":
        _METRICS["request_verification"] += 1
    else:
        _METRICS["deny"] += 1

    try:
        log_audit_event(
            session_id=payload.session_id,
            employee_id=payload.employee_id,
            decision=decision.value,
            confidence=confidence,
            latency_ms=latency_ms,
            identity_score=identity,
            liveness_score=liveness,
            spoof_score=spoof_score,
            reasons=reasons,
        )
    except Exception:
        pass

    return VerifyResponse(
        decision=decision,
        confidence=round(confidence, 4),
        latency_ms=latency_ms,
        identity_score=round(identity, 4),
        liveness_score=round(liveness, 4),
        spoof_score=round(spoof_score, 4),
        reasons=reasons,
    )


@app.post("/step-up", response_model=StepUpResponse)
def step_up(payload: StepUpRequest) -> StepUpResponse:
    try:
        verify_challenge(payload.challenge_token, payload.session_id)
    except Exception as exc:
        raise HTTPException(status_code=401, detail=f"Invalid or expired challenge token: {exc}") from exc

    try:
        instruction = bedrock_step_up_instruction(payload.prior_reasons)
    except Exception:
        instruction = "Turn your head slightly right and blink twice."

    return StepUpResponse(instruction=instruction, timeout_ms=settings.step_up_timeout_ms)


@app.get("/metrics", response_model=MetricsResponse)
def metrics() -> MetricsResponse:
    latencies = _METRICS["latencies"]
    p95 = int(statistics.quantiles(latencies, n=20)[-1]) if len(latencies) >= 20 else (max(latencies) if latencies else 0)
    total = len(latencies)

    return MetricsResponse(
        total_requests=total,
        allow_count=_METRICS["allow"],
        request_verification_count=_METRICS["request_verification"],
        deny_count=_METRICS["deny"],
        p95_latency_ms=p95,
    )