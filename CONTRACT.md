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
{ "user_id": "alice", "frames": ["<b64>", "<b64>", "<b64>"] }
```
Response:
```json
{ "ok": true, "user_id": "alice", "frames_used": 3, "quality": 0.93, "message": "" }
```
`ok:false` + `message` if no face / several faces / low quality.

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
- `turn_left`  = user turns head to **their** left  → nose moves to **image right** → kiosk: `yaw_proxy > +0.30`; server: max(yaw) − yaw of most frontal frame ≥ 0.12
- `turn_right` = user turns head to **their** right → kiosk: `yaw_proxy < −0.30`; server: yaw of most frontal frame − min(yaw) ≥ 0.12
- `blink` = MediaPipe `eyeBlinkLeft` and `eyeBlinkRight` both > 0.5
- `smile` = MediaPipe `mouthSmileLeft` and `mouthSmileRight` both > 0.5

## POST /otp/request, POST /otp/verify  (simulated fallback)
```json
// /otp/request  { "user_id": "alice" }            → { "ok": true, "demo_code": "482913" }
// /otp/verify   { "user_id": "alice", "code": "482913" } → VerifyResponse with ALLOW or DENY
```
