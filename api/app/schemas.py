"""Pydantic models — must match CONTRACT.md exactly."""
from __future__ import annotations

from datetime import datetime
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
    authority: Literal["student", "teacher", "admin"] = "student"
    active: bool = True
    expires_in_seconds: Optional[int] = Field(default=None, ge=1, le=31536000)
    # campus permission granted at enrollment: "24x7" (default), "campus_hours" (Mon-Fri 08:00-18:00), "none"
    schedule_preset: Literal["24x7", "campus_hours", "none"] = "24x7"


class EnrollResponse(BaseModel):
    ok: bool
    user_id: str
    frames_used: int = 0
    quality: float = 0.0
    message: str = ""


class UserProfile(BaseModel):
    user_id: str
    authority: Literal["student", "teacher", "admin"] = "student"
    active: bool = True
    expires_at: Optional[datetime] = None
    created_at: datetime


class UserUpdateRequest(BaseModel):
    authority: Optional[Literal["student", "teacher", "admin"]] = None
    active: Optional[bool] = None
    expires_in_seconds: Optional[int] = Field(default=None, ge=0, le=31536000)


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
    schedule: Optional["ScheduleStatus"] = None
    offline: Optional["OfflineStatus"] = None
    attestation: Optional[dict] = None   # ML-DSA-65 signed decision (see app/pqc.py)



class AccessWindowModel(BaseModel):
    start: str
    end: str


class AccessScheduleRequest(BaseModel):
    enabled: bool = False
    timezone: str = "UTC"
    weekly: dict[str, list[AccessWindowModel]] = {}
    exceptions: dict[str, Optional[list[AccessWindowModel]]] = {}


class AccessScheduleResponse(AccessScheduleRequest):
    user_id: str
    updated_at: float


class ScheduleStatus(BaseModel):
    authorized: bool
    reason: str
    timezone: str
    evaluated_at: datetime


class OfflineStatus(BaseModel):
    local: bool = False
    sync_status: Literal["not_applicable", "pending", "synced", "rejected"] = "not_applicable"
    event_id: Optional[str] = None


class OfflineVerifyRequest(BaseModel):
    user_id: str
    liveness: Optional[float] = None
    identity_similarity: Optional[float] = None
    signals: dict[str, Optional[float]] = {}
    occurred_at: Optional[datetime] = None
    challenge_completed: bool = False


class OfflineSyncEvent(BaseModel):
    event_id: str
    user_id: Optional[str] = None
    decision: Decision
    occurred_at: datetime
    payload: dict = {}


class OfflineSyncRequest(BaseModel):
    events: list[OfflineSyncEvent] = Field(default_factory=list, max_length=100)


class OtpRequest(BaseModel):
    user_id: str


class OtpVerify(BaseModel):
    user_id: str
    code: str
