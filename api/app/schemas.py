"""Pydantic models — must match CONTRACT.md exactly."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

Decision = Literal["ALLOW", "REQUEST_VERIFICATION", "DENY"]


class ClientQuality(BaseModel):
    blur: Optional[float] = None
    brightness: Optional[float] = None
    face_px: Optional[float] = None
    yaw: Optional[float] = None


class EnrollRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=64)
    frames: list[str] = Field(min_length=1, max_length=10)


class EnrollResponse(BaseModel):
    ok: bool
    user_id: str
    frames_used: int = 0
    quality: float = 0.0
    message: str = ""


class VerifyRequest(BaseModel):
    frames: list[str] = Field(min_length=1, max_length=12)
    client_quality: Optional[ClientQuality] = None
    claimed_user_id: Optional[str] = None


class ClientEvent(BaseModel):
    action: str
    t_ms: float


class ChallengeRequest(BaseModel):
    nonce: str
    frames: list[str] = Field(min_length=1, max_length=16)
    client_events: list[ClientEvent] = []


class Liveness(BaseModel):
    score: Optional[float] = None
    signals: dict[str, Optional[float]] = {}


class Identity(BaseModel):
    user_id: Optional[str] = None
    similarity: Optional[float] = None
    matched: bool = False


class Quality(BaseModel):
    ok: bool = True
    issues: list[str] = []


class Challenge(BaseModel):
    nonce: str
    actions: list[str]
    expires_in_s: int = 20


class VerifyResponse(BaseModel):
    decision: Decision
    confidence: float
    reasons: list[str] = []
    message: str = ""
    liveness: Liveness = Liveness()
    identity: Identity = Identity()
    quality: Quality = Quality()
    challenge: Optional[Challenge] = None
    timings_ms: dict[str, float] = {}


class OtpRequest(BaseModel):
    user_id: str


class OtpVerify(BaseModel):
    user_id: str
    code: str
