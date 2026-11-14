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
