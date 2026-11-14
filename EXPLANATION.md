# Spot-a-Spoof: full explanation
**HID Hackathon "Can You Spot a Spoof?" · Team Goal Diggers**

A face-liveness and identity kiosk. It decides **ALLOW / REQUEST VERIFICATION (step-up) / DENY** in well under 2 seconds. On top of that it adds campus-schedule authorization, offline operation, admin-protected enrollment and **post-quantum signed decisions**. It is deployed on the HID AWS sandbox.

- **Live (AWS):** https://duuh07ci1fhe6.cloudfront.net (HTTPS, so the camera also works on phones)
- **Local:** API `http://localhost:8000`, kiosk `http://localhost:5173`
- Other docs: [CONTRACT.md](CONTRACT.md) (API), [build_plan.md](build_plan.md) (plan), [spoof_approaches.md](spoof_approaches.md) (research), [gb10/README.md](gb10/README.md) (training), [infra/aws.md](infra/aws.md)

---

## 1. The problem
An HID entry kiosk must tell a **live person** from a **presentation attack** (printed photo, screen replay, video replay, mask), and also confirm **who** it is. The decision must come within **2 seconds**:
- **ALLOW:** live person whose identity matches
- **REQUEST VERIFICATION:** low confidence, so an extra step is needed (recapture, a simple action, or OTP)
- **DENY:** spoof, unknown person, or failed verification

Surprise challenges added during the event:
1. **Campus schedule:** a live, recognised person is not automatically authorized. The face result is combined with a campus access schedule to give ALLOW / STEP-UP / DENY.
2. **Offline mode:** if the kiosk loses the central service, it must keep making safe local decisions, avoid unsafe automatic access, offer fallback verification, store only minimal temporary data, and sync when the connection returns.

---

## 2. Architecture
```
Browser kiosk (React + MediaPipe)                       ── HTTPS ──>  AWS CloudFront
  · quality gate (size, light, blur, pose)                              │
  · 5-frame burst (~400 ms)                                             ▼
  · gesture challenge (turn + smile)                          EC2 c7i.xlarge (Docker)
  · verifies ML-DSA signature before "opening the door"        FastAPI decision service
  · OFFLINE: falls back to the laptop's local API              (same image runs locally)
                                                                │
   FastAPI decision service ─────────────────────────────────────┘
     1. face detection + landmarks ........ OpenCV YuNet (or insightface SCRFD)
     2. liveness signals, in parallel ...... MiniFAS · DINOv2 head (GB10-trained) · YOLO screen detector · 3D parallax
     3. identity ........................... OpenCV SFace embeddings, 1:N gallery search
     4. decision engine .................... allow / step-up / deny + reasons
     5. campus schedule + profile rules .... akshat's access module
     6. post-quantum signature ............. ML-DSA-65 over the decision
     storage: SQLite (edge) or DynamoDB (cloud) · optional S3 / DynamoDB audit / Rekognition
```
Median server time is **~150–400 ms** per decision, plus a ~400 ms capture burst. That is **well under the 2 s budget**.

---

## 3. How a decision is made
### 3.1 Capture (kiosk, `kiosk-web/src/App.jsx`, `lib/faceTracker.js`)
MediaPipe Face Landmarker runs in the browser and checks face size, brightness, sharpness and head pose. When the face has been good for 0.6 s, the kiosk grabs **5 frames 100 ms apart** and sends them. After each decision it waits for the person to step away, so the same person isn't re-checked in a loop.

### 3.2 Liveness signals (`api/app/liveness/`)
Every signal returns 1.0 for live, 0.0 for spoof, or *no opinion*.

| Signal | What it is | Catches | Notes from real testing |
|---|---|---|---|
| **MiniFAS** (`minifas.py`) | pretrained MiniFASNet-V2-SE (CelebA-Spoof), ONNX | print / screen texture | Unreliable in warm, dim hall light: it scored a real person 0.01–0.99 within one burst. Now uses two crop scales averaged and the median over frames, and can **never deny on its own** |
| **DINOv2 head** (`dino_head.py`) | frozen DINOv2-S features + linear head, **trained by us on the GB10** on 54k CelebA-Spoof images plus our own kiosk frames | all attack types learned from data | Fixed real people being rejected (see §6). Main "live" signal |
| **Screen detector** (`device.py`) | YOLO11n COCO classes phone / laptop / tv. Is the face *inside* a device? | screen photo / video replay with a visible device | Hard DENY. Misses phones held so close that they fill the frame |
| **3D parallax** (`planarity.py`) | does the nose move off the eye/mouth plane between frames? | flat photos tilted in front of the camera | Weak with 5-point landmarks; only gives evidence on large motion, otherwise "no opinion" |

Signals are fused (`fuse.py`) with weights favouring the domain-trained DINOv2 head (MiniFAS 0.3, DINOv2 2.0, parallax 0.7).

### 3.3 Identity (`api/app/faces.py`, `store.py`)
OpenCV **YuNet** (detection, 5 landmarks) and **SFace** (embeddings, ~99.4 % on LFW). They run natively on Windows; insightface SCRFD + ArcFace is used automatically when installed. Matching is cosine similarity against the gallery. Each enrolled person is stored as **one 128/512-number template, not a photo**. The kiosk shows *who* was recognised on every result: the name and score, "name?" when uncertain, or "Not recognised (closest: …)".

### 3.4 Decision rules (`api/app/decision.py`, unit-tested)
1. No face, several people, or poor quality → **REQUEST VERIFICATION** (recapture, with guidance)
2. Face is inside a phone/laptop/tv → **DENY** `phone_detected`
3. **Both** texture models say screen (MiniFAS < 0.01 **and** DINOv2 < 0.3) → **DENY** `screen_texture`
4. Clearly flat face + low liveness → **DENY** `planar_motion`
5. Not enrolled / face doesn't match → **DENY** `unknown_person` / `identity_mismatch`
6. Texture models **disagree** → **REQUEST VERIFICATION** (never ALLOW, never DENY)
7. Confident liveness **and** identity → **ALLOW**
8. Anything in between → **REQUEST VERIFICATION** + challenge

### 3.5 Step-up verification (`api/app/challenge.py`)
- **Gesture challenge:** **one head turn (left or right) + smile**, in random order, within 10 s.
  - **Random order** defeats pre-recorded video.
  - The **head turn is checked on the server** from the frames: the turn is measured relative to the person's own most frontal frame. Tilting a flat photo 52° moves this measure only 0.03, while a real 17° head turn moves it 0.15 (unit-tested).
  - **Smile** defeats rigid paper and 3D masks, which cannot deform the mouth.
  - The screen and texture checks still apply during the challenge.
- **OTP fallback** if the challenge fails (simulated; production would send it via Amazon SNS).

---

## 4. Surprise challenge 1: campus schedule (akshat + shreyas)
`api/app/access.py`, `main.py::_apply_schedule`
- Per-person **weekly time windows, IANA time zone, date exceptions** (e.g. holidays), enable/disable. Evaluated on the **server clock**, never the browser's.
- Profiles: authority (student / teacher / admin), active/inactive, optional expiry.
- **All three outcomes from the schedule:**
  - inside the window → decision passes through (**ALLOW** if live and recognised)
  - **within 15 min of the window** (`SCHEDULE_GRACE_MIN`) or **teacher/admin outside hours** → **STEP-UP** (challenge, then OTP)
  - otherwise / disabled / expired / no permission → **DENY** (fail-closed)
- **Enrollment grants a schedule** from a picker: *Any time (24/7)*, *Campus hours (Mon–Fri 08:00–18:00)*, or *No access*. A freshly enrolled person is therefore not silently denied.

## 5. Surprise challenge 2: offline mode with safe access (akshat + shreyas)
`main.py::offline_verify_frames`, `/offline/*`, `kiosk-web/src/lib/api.js`

| Requirement | Implementation |
|---|---|
| Safe local decision process | When the central (cloud) API is unreachable, the kiosk sends the burst to the **local edge API on the laptop**, which runs the **same models** with its locally stored faces and schedules |
| No unsafe automatic access | Stricter offline rules: ALLOW only with confident liveness, identity ≥ threshold + 0.05, **and** a valid schedule. Everything else goes to the challenge or DENY. If the local API is also down, the kiosk denies (fail-closed) |
| Fallback verification | The gesture challenge runs entirely on the local edge (OTP when online) |
| Minimum temporary information | **No images are written offline** (verified: 0 files). The queue keeps only `{decision, user, time, reasons, decision_id}`, and every decision is post-quantum signed |
| Sync when connection returns | The kiosk forwards `/offline/pending` to the central `/offline/sync`, which rejects ALLOWs made on a stale or invalid permission, then `/offline/ack` marks them synced locally |

The cloud kiosk can reach `http://localhost:8000` because the API enables CORS **Private Network Access** (needed by Chrome/Brave).

---

## 6. Training our own liveness model on the GB10
`training/gb10_train.py`, `gb10/*`

**Problem found in live testing:** the pretrained models flagged real people as screens in the hall's lighting.
- **Data:** CelebA-Spoof parquet mirror on HuggingFace (`Ar4ikov/celebA_spoof`), **54,159 training images**, plus **279 face crops from our own kiosk** (real people and phone attacks, labelled from debug dumps).
- **Model:** frozen **DINOv2-S** features, then a logistic-regression head. Kiosk samples carry 20 % of the weight, balanced 50/50 live/spoof. Exported as a portable `.npz` (no pickles across machines).
- **Infrastructure:** the GB10 was only reachable through a VPN that drops after about 3 minutes. Each window therefore uses **one SSH connection**: stream a 4.6 MB bundle and start the job in **tmux**. The GB10 then downloads the data and trains on its own. A later window fetches the result. Downloads are time-boxed and every stage is resumable.

**Results** ([eval/gb10_metrics.json](eval/gb10_metrics.json)), honest held-out numbers:

| Tested on | CelebA-only head | CelebA + kiosk head |
|---|---|---|
| CelebA-Spoof test (2,731 imgs) | AUC 0.963, acc 84 % | AUC 0.968, acc 85 % |
| **Real** kiosk bursts never seen in training (11) | **91 % wrongly rejected** | **0 % rejected** |
| Held-out phone-attack burst (1) | caught | missed (0.99) |

So the head fixes real-person rejection but does not catch phone attacks on its own. That's why rule 3 needs **both** texture models to agree before a screen DENY, and rule 6 sends disagreements to the challenge.

**Replay of all saved kiosk attempts** (40 bursts) with the final logic: **real people 27/38 ALLOW at once, 11 to the challenge, 0 wrongly denied; no phone attack allowed.**

---

## 7. Post-quantum cryptography (NIST FIPS 203 / 204)
`api/app/pqc.py`, `kiosk-web/src/lib/pqc.js`
- **Every decision is signed with ML-DSA-65** (FIPS 204): verify, challenge, OTP and offline. The signed payload holds the decision, user, reasons, scores, `issued_at`, a **30 s expiry**, a random **decision_id** and the key id. About 45 ms per signature (pure-Python `dilithium-py`).
- **The kiosk acts as the door controller:** it verifies with `@noble/post-quantum` before acting. **An ALLOW is honoured only with a valid, unexpired, never-seen signature from the pinned key.** Forged, tampered, stripped, replayed or expired decisions become DENY `signature_invalid`. The result screen shows "🔐 Post-quantum signed · ML-DSA-65 verified".
- End-to-end test (`kiosk-web/scripts/pqc_e2e.mjs`):
  - real ALLOW verified in ~35 ms
  - **replay rejected**
  - **DENY forged into ALLOW rejected**
  - **stripped signature rejected**
  - Python-made signatures were cross-verified in the JS library.
- `GET /pqc/public-key`, `POST /pqc/verify`. Keys live in `api/keys/` (gitignored).
- **ML-KEM-768** (FIPS 203) helpers exist in `pqc.py` (not yet wired to anything). On the wire, HTTPS in current Chrome already negotiates hybrid post-quantum TLS (X25519MLKEM768).
- **Not covered:** camera frames and stored templates are not encrypted with ML-KEM. Sealing templates with ML-KEM + AES-GCM, with the key in KMS/TPM, is the next step.

---

## 8. Security and privacy
- **Admin PIN** (`ADMIN_PIN`, default `2468`, constant-time compare) is required to **enroll**, **view the enrolled list**, change or delete users, and change schedules. The Enroll page opens freely; the PIN is asked when you enroll or unlock the list. `/health` never reveals user ids.
- Stored biometrics are **templates only**, never photos (the optional S3 reference photo is off by default).
- Debug frame dumps (`DEBUG_DUMP_DIR`) are a local development tool only: off in the cloud, and never used in offline mode.
- On AWS, the EC2 port 80 accepts traffic **only from CloudFront's origin-facing IP range**. The S3 deploy bucket is private. The IAM role can only read its own bundle.

## 9. AWS deployment (HID sandbox, us-east-1)
`deploy/aws_deploy.py`, `deploy/Dockerfile.ec2`
- App Runner is **denied** in the sandbox, so we use: **S3** (12 MB bundle) → **EC2 c7i.xlarge**, which builds and runs the container → **CloudFront** for HTTPS.
- Idempotent: `python deploy/aws_deploy.py --update` redeploys code (~8 min), `--status` shows state.
- Optional AWS modules from siri's backend (`api/app/aws_services.py`), all off by default and failure-safe: **DynamoDB audit log**, **S3 enrolled photos**, **Rekognition CompareFaces** as a second opinion only when our match is uncertain. `GET /metrics` gives decision counts and p50/p95 latency.

## 10. Team contributions
- **shreyas:** architecture, API + decision engine, liveness pipeline and fixes from live testing, GB10 training pipeline and model, PQC, offline edge path, admin auth, AWS deployment, integration.
- **akshat:** campus schedules, enrollment controls (authority / active / expiry), offline event store and sync, Windows setup guide.
- **siri:** AWS-native backend (`backend/`): Rekognition, Bedrock step-up prompts, S3/DynamoDB audit, flash/moiré prototypes. Its AWS pieces are ported into `api/`.

## 11. Honest limitations
- **Masks:** covered only indirectly (CelebA-Spoof training data includes paper and 3D masks) plus the **smile** step against rigid masks. **No physical mask was tested;** no per-mask metric exists (the HF mirror only labels live/spoof).
- **Phone attacks held very close** are not seen by the screen detector; they rely on the texture models and the challenge.
- Kiosk metrics come from a **small live set in one hall** (≈40 bursts, 4 phone-attack bursts). The thresholds were tuned on this data, so treat the numbers as indicative.
- The cloud SQLite database lives on the instance (it is seeded from the laptop at deploy). Production would use the DynamoDB store.
- OTP is simulated; SNS delivery is not wired.

## 12. Running it
```bash
# API (Windows, no Docker): see README.md for first-time setup
cd api && FACE_BACKEND=opencv .venv/Scripts/python -m uvicorn app.main:app --port 8000
# kiosk
cd kiosk-web && npm install && npm run dev        # http://localhost:5173
# tests
cd api && .venv/Scripts/python -m pytest -q       # 35 tests
cd kiosk-web && node scripts/pqc_e2e.mjs          # PQC end-to-end
# deploy / redeploy to AWS
api/.venv/Scripts/python deploy/aws_deploy.py --update
```

## 13. Demo script (3 min)
1. **Enroll** a volunteer (admin PIN), with campus access 24/7.
2. Volunteer steps up: **ALLOW**, name shown, ~1 s, "🔐 post-quantum signed".
3. Hold up a **phone** with their photo: **DENY** (screen), still showing "recognised as …". Identity alone isn't enough.
4. Dim the light or look away: **REQUEST VERIFICATION**, then *turn + smile*, then ALLOW.
5. Set their schedule to campus hours and try "outside hours": **STEP-UP** in the grace window, or **DENY**.
6. Turn off Wi-Fi: **"OFFLINE (local)"** decision on the laptop. Wi-Fi back on: the decisions sync.
7. Show the honest numbers: the GB10-trained head took real-person rejection from **91 % to 0 %**, and no attack was ever allowed in the replay.
