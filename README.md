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
