# HIDhackathon — Spot-a-Spoof

Team Name - Goal Diggers


Face liveness + identity verification for an entry kiosk. Decides **ALLOW / REQUEST VERIFICATION / DENY** in under 2 s.

- Plan: [build_plan.md](build_plan.md) · Detector research: [spoof_approaches.md](spoof_approaches.md) · API: [CONTRACT.md](CONTRACT.md)

## Layout (one owner per folder)
| Folder | Owner | What |
|---|---|---|
| `kiosk-web/` | P1 | Vite + React kiosk UI, MediaPipe quality gate & challenge |
| `api/` | P3 | FastAPI decision service, identity, decision engine, storage |
| `api/app/liveness/` | P2 | MiniFASNet, DINOv2 head, device detector, planarity, fusion |
| `training/` | P2 | DINOv2 feature extraction + head training (GB10) |
| `eval/` | P4 | Metrics, LFW/CelebA-Spoof eval, live test-set recorder, dashboard |
| `infra/` | P3 | Docker compose, AWS deployment notes |

## Quick start

### 1. API (stub mode — no models needed, unblocks the UI immediately)
```bash
cd api
python -m venv .venv && source .venv/Scripts/activate   # Windows Git Bash
pip install -r requirements-core.txt
STUB_MODE=1 uvicorn app.main:app --reload --port 8000
```

For a clean Python 3.10 PowerShell setup, see [SETUP_WINDOWS.md](SETUP_WINDOWS.md).

### 2. API (real models), natively on Windows, no Docker needed
```bash
cd api
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install ultralytics
python scripts/download_models.py        # MiniFAS, YOLO, YuNet + SFace into api/models/
FACE_BACKEND=opencv uvicorn app.main:app --port 8000
```
Faces use the OpenCV backend (YuNet detection + SFace recognition), which is built into opencv-python.
In Docker/Linux, `insightface` (SCRFD + ArcFace) is used automatically when installed:
`docker compose -f ../infra/docker-compose.yml up --build`.

**Enroll first.** An un-enrolled face is always DENY ("Person not enrolled"). Use the kiosk's Enroll tab.
The REQUEST_VERIFICATION challenge (blink / turn) appears only when liveness or identity is *uncertain*. A clear live, enrolled face gets ALLOW directly.

### Campus access and offline safety

Campus permissions are configured through the FastAPI schedule endpoints
documented in [CONTRACT.md](CONTRACT.md). They use timezone-aware weekly
windows and date exceptions; biometric success alone never grants access
outside an enabled schedule.

The kiosk shows ONLINE/OFFLINE MODE and queues only non-image decision metadata
for synchronization. If the central API is unavailable and no separately
configured local biometric service (`VITE_OFFLINE_API_BASE`) is available, the
browser fails closed with DENY rather than granting access.

### 3. Kiosk
```bash
cd kiosk-web
npm install
npm run dev        # http://localhost:5173 (proxies /api → :8000)
```

## Environment variables (api)
| Var | Default | Meaning |
|---|---|---|
| `STUB_MODE` | `0` | `1` = fake scores, no models loaded |
| `MODELS_DIR` | `api/models` | weights location |
| `STORE` | `sqlite` | `sqlite` or `dynamo` |
| `DYNAMO_TABLE` | `hid-enrollments` | when `STORE=dynamo` |
| `AUDIT_BUCKET` | — | S3 bucket for audit frames (optional) |
| `THRESHOLDS_FILE` | `api/thresholds.json` | calibrated thresholds (written by eval) |
| `ENABLE_DINO` / `ENABLE_DEVICE` / `ENABLE_PLANARITY` | `1` | toggle signals (for ablations) |

## AWS integrations (optional, from siri's `backend/`)
Ported into `api/app/aws_services.py`. Every integration is off unless its variable is set, runs in the background or only in the uncertain zone, and can never break a decision.

| Variable | Effect |
|---|---|
| `AWS_AUDIT_TABLE=AccessAuditLogs` | every decision is written to DynamoDB (audit trail) |
| `ENROLLED_FACES_BUCKET=hid-goaldigger-enrolled-faces` | enrollment uploads a reference photo `<user_id>.jpg` to S3 |
| `REKOGNITION_ENABLED=1` | Amazon Rekognition CompareFaces as a second opinion when our face match is uncertain |
| `AWS_REGION` | default `us-east-1`; credentials come from the normal boto3 chain or the EC2 instance role |

`GET /metrics` returns decision counts, top reasons and p50/p95 server latency since startup.

## Alternative AWS-native backend (`backend/`, by siri)

A separate FastAPI service built on AWS (Rekognition, Bedrock, S3, DynamoDB) with its own API. Its AWS pieces are also ported into the main `api/` as optional modules (see *AWS integrations* above). The kiosk talks to `api/`.

### Zero-Trust Dual-Plane MVP

This repo now includes a backend starter for the architecture you proposed:

- Fast Path (<1.2s target): deterministic CV + Rekognition checks
- Verification Path: Bedrock Nova Lite dynamic step-up challenge
- Audit Path: metrics endpoint for latency and decision distribution

### Folder Structure

```text
.
├── README.md
└── backend
	├── .env.example
	├── requirements.txt
	└── app
		├── __init__.py
		├── aws_clients.py
		├── challenge.py
		├── config.py
		├── detectors.py
		├── main.py
		├── models.py
		└── scoring.py
```

### Quick Start (Local)

1. Create a Python environment and install dependencies.

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Create `.env` from `.env.example` and set values.

```powershell
copy .env.example .env
```

3. Run the API.

```powershell
uvicorn app.main:app --reload --port 8000
```

4. Verify service.

- `GET /health`
- `POST /start-session`
- `POST /verify`
- `POST /step-up`
- `GET /metrics`

### Important: One Required Wiring Step

`/verify` currently raises `501` until enrolled image lookup is wired.

Implement `_mock_enrolled_image(employee_id)` in `backend/app/main.py` using one of these:

- S3 object lookup by employee id (recommended for hackathon)
- DynamoDB metadata + S3 key fetch
- Rekognition face collection workflow

### Request Contracts

#### Start Session

`POST /start-session`

Response contains:

- signed JWT challenge token
- flash color sequence
- short capture hint

#### Verify

`POST /verify`

Input payload:

- `session_id`
- `challenge_token`
- `employee_id`
- `baseline_b64`
- `flash1_b64`
- `flash2_b64`

Backend does parallel execution:

- OpenCV photometric spoof score
- OpenCV moire score
- Rekognition compare faces
- Rekognition face quality as liveness proxy

Decision policy:

- `ALLOW` if confidence and identity are strong
- `REQUEST_VERIFICATION` for gray zone
- `DENY` for low confidence/high spoof risk

#### Step-Up

`POST /step-up`

Uses Bedrock Nova Lite to generate one dynamic motion challenge sentence.
Falls back to deterministic challenge text if Bedrock times out/fails.

### 6-Hour Implementation Checklist

1. Wire enrolled face retrieval from S3.
2. Add DynamoDB audit logging per verification event.
3. Add kiosk UI that captures 3 frames in a single burst.
4. Measure end-to-end latency and show in UI.
5. Execute spoof tests (print/photo replay/video/mask) and collect metrics.
6. Tune thresholds in `backend/app/scoring.py` with observed results.

### Suggested Next Commit

- Add `infra/` IaC for API Gateway + Lambda + IAM + DynamoDB + S3.
- Add `frontend/` kiosk client with screen flash and burst capture.

