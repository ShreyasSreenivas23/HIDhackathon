# API Contract (FROZEN after hour 0:30 — change only with whole-team agreement)

Base URL: `http://localhost:8000` (the kiosk dev server proxies `/api/*` to it).
All images are **base64 JPEG strings** (with or without a `data:image/jpeg;base64,` prefix), raw camera frames (**not mirrored**), at least 640×480.

**Score convention:** every score is **1.0 = live / genuine**, **0.0 = spoof / mismatch**. `null` = the signal had no opinion (model missing, not enough motion, …).

## GET /health
```json
{ "ok": true, "stub_mode": false, "components": { "detector": true, "minifas": true, "dino": false, "device": true, "identity": true }, "enrolled": 3 }
```

## POST /enroll
Request:
```json
{
  "user_id": "alice",
  "frames": ["<b64>", "<b64>", "<b64>"],
  "authority": "student",
  "active": true,
  "expires_in_seconds": 60
}
```
Response:
```json
{ "ok": true, "user_id": "alice", "frames_used": 3, "quality": 0.93, "message": "" }
```
`ok:false` + `message` if no face / several faces / low quality.

`authority` is `student`, `teacher`, or `admin`. `active` controls whether
the enrollment is enabled. `expires_in_seconds` is optional; omit it for no
expiry. An enabled enrollment can still be denied after its expiry time.
Use `GET /users/details` to view `active`, `expires_at`, and `authority`.
Use `PATCH /users/{user_id}` to change these fields. Send
`{"expires_in_seconds": 0}` to clear an existing expiry without re-enrolling.
`DELETE /users/{user_id}` removes the enrollment.

## POST /verify
Request:
```json
{
  "frames": ["<b64>", "... 5 frames captured over ~500 ms"],
  "client_quality": { "blur": 120.0, "brightness": 0.55, "face_px": 210, "yaw": 0.04 },
  "claimed_user_id": null
}
```
`claimed_user_id` optional: if set → 1:1 verification, else 1:N search over the gallery.

Response (`VerifyResponse`):
```json
{
  "decision": "ALLOW",
  "confidence": 0.96,
  "reasons": [],
  "message": "Live person detected & identity verified",
  "liveness": {
    "score": 0.97,
    "signals": { "minifas": 0.95, "dino": 0.98, "device": 1.0, "planarity": 0.91 }
  },
  "identity": { "user_id": "alice", "similarity": 0.71, "matched": true },
  "quality": { "ok": true, "issues": [] },
  "challenge": null,
  "timings_ms": { "decode": 8, "detect": 25, "liveness": 140, "identity": 12, "decision": 0, "total": 190 }
}
```
- `decision` ∈ `"ALLOW" | "REQUEST_VERIFICATION" | "DENY"`
- `reasons`: machine-readable codes shown in the UI, e.g. `phone_detected`, `planar_motion`, `texture_spoof`, `low_liveness`, `identity_mismatch`, `unknown_person`, `low_quality`, `no_face`, `multiple_faces`, `challenge_failed`.
- `challenge`: present only when `decision == "REQUEST_VERIFICATION"`:
  ```json
  { "nonce": "x9f2a1", "actions": ["blink", "turn_left"], "expires_in_s": 20 }
  ```

## POST /challenge
Request:
```json
{
  "nonce": "x9f2a1",
  "frames": ["<b64>", "... 6–12 frames sampled during the challenge"],
  "client_events": [ { "action": "blink", "t_ms": 412 }, { "action": "turn_left", "t_ms": 1630 } ]
}
```
Response: same shape as `/verify`. `challenge` is always `null` here — on failure the decision is `DENY` with reason `challenge_failed` and the kiosk offers the OTP fallback.

### Action definitions (client and server must agree)
Computed on **raw (unmirrored) camera frames**:
- `yaw_proxy = (nose_x - eye_mid_x) / eye_distance`  (≈0 when frontal)
- `turn_left`  = user turns head to **their** left  → nose moves to **image right** → `yaw_proxy > +0.25`
- `turn_right` = user turns head to **their** right → `yaw_proxy < -0.25`
- `blink` = MediaPipe `eyeBlinkLeft` and `eyeBlinkRight` both > 0.5
- `smile` = MediaPipe `mouthSmileLeft` and `mouthSmileRight` both > 0.5

## Campus access schedules

Schedules are evaluated by the backend using the configured IANA timezone; the
browser clock is never used for authorization. `weekly` keys are
`monday` through `sunday`, and each window is half-open (`start` inclusive,
`end` exclusive). An exception value of `null` blocks a date; a list replaces
the weekly windows for that date.

```http
PUT /access/schedules/alice
Content-Type: application/json
```

```json
{
  "enabled": true,
  "timezone": "America/New_York",
  "weekly": {
    "monday": [{"start": "08:00", "end": "18:00"}],
    "tuesday": [{"start": "08:00", "end": "18:00"}]
  },
  "exceptions": {"2026-12-25": null}
}
```

Use `GET /access/schedules` to retrieve configured permissions and
`GET /access/schedules/{user_id}` for one user. A real biometric `ALLOW`
without an enabled, in-window schedule becomes `DENY` with
`no_campus_permission`, `permission_disabled`, or `outside_schedule`.
`VerifyResponse.schedule` reports the authoritative evaluation.

## Offline mode

`POST /offline/verify` is the local decision endpoint. It accepts only local
biometric scores and a cached user schedule; missing/expired permissions,
unknown identities, uncertain scores, and device spoof signals fail closed.
It never accepts raw frames and queues only an event ID, user ID, decision,
timestamp, and score metadata in SQLite.

```json
{
  "user_id": "alice",
  "liveness": 0.92,
  "identity_similarity": 0.71,
  "signals": {"device": 1.0},
  "occurred_at": "2026-11-16T14:00:00Z"
}
```

Use `GET /offline/status` to inspect pending local events. When connectivity
returns, the kiosk sends them to `POST /offline/sync`; duplicate event IDs are
idempotent. Server-side schedule validation rejects stale or invalid cached
`ALLOW` events. If no local biometric service is configured, the browser
displays OFFLINE MODE and denies rather than treating network failure as
authorization.

## POST /otp/request, POST /otp/verify  (simulated fallback)
```json
// /otp/request  { "user_id": "alice" }            → { "ok": true, "demo_code": "482913" }
// /otp/verify   { "user_id": "alice", "code": "482913" } → VerifyResponse with ALLOW or DENY
```
