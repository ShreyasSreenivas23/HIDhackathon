# HIDhackathon

Team Name - Goal Diggers

## Zero-Trust Dual-Plane MVP

This repo now includes a backend starter for the architecture you proposed:

- Fast Path (<1.2s target): deterministic CV + Rekognition checks
- Verification Path: Bedrock Nova Lite dynamic step-up challenge
- Audit Path: metrics endpoint for latency and decision distribution

## Folder Structure

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

## Quick Start (Local)

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

## Important: One Required Wiring Step

`/verify` currently raises `501` until enrolled image lookup is wired.

Implement `_mock_enrolled_image(employee_id)` in `backend/app/main.py` using one of these:

- S3 object lookup by employee id (recommended for hackathon)
- DynamoDB metadata + S3 key fetch
- Rekognition face collection workflow

## Request Contracts

### Start Session

`POST /start-session`

Response contains:

- signed JWT challenge token
- flash color sequence
- short capture hint

### Verify

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

### Step-Up

`POST /step-up`

Uses Bedrock Nova Lite to generate one dynamic motion challenge sentence.
Falls back to deterministic challenge text if Bedrock times out/fails.

## 6-Hour Implementation Checklist

1. Wire enrolled face retrieval from S3.
2. Add DynamoDB audit logging per verification event.
3. Add kiosk UI that captures 3 frames in a single burst.
4. Measure end-to-end latency and show in UI.
5. Execute spoof tests (print/photo replay/video/mask) and collect metrics.
6. Tune thresholds in `backend/app/scoring.py` with observed results.

## Suggested Next Commit

- Add `infra/` IaC for API Gateway + Lambda + IAM + DynamoDB + S3.
- Add `frontend/` kiosk client with screen flash and burst capture.
