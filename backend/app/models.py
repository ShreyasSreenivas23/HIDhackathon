from enum import Enum
from pydantic import BaseModel, Field


class Decision(str, Enum):
    ALLOW = "ALLOW"
    REQUEST_VERIFICATION = "REQUEST_VERIFICATION"
    DENY = "DENY"


class StartSessionResponse(BaseModel):
    session_id: str
    challenge_token: str
    flash_colors: list[str]
    prompt_hint: str
    expires_in_seconds: int


class VerifyRequest(BaseModel):
    session_id: str
    challenge_token: str
    employee_id: str
    baseline_b64: str = Field(description="Base64 JPEG frame")
    flash1_b64: str = Field(description="Base64 JPEG frame under first color")
    flash2_b64: str = Field(description="Base64 JPEG frame under second color")


class VerifyResponse(BaseModel):
    decision: Decision
    confidence: float
    latency_ms: int
    identity_score: float
    liveness_score: float
    spoof_score: float
    reasons: list[str]


class StepUpRequest(BaseModel):
    session_id: str
    challenge_token: str
    prior_reasons: list[str]


class StepUpResponse(BaseModel):
    instruction: str
    timeout_ms: int


class MetricsResponse(BaseModel):
    total_requests: int
    allow_count: int
    request_verification_count: int
    deny_count: int
    p95_latency_ms: int
