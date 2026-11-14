# Windows setup

These commands use the installed Python 3.10 interpreter. Run them in
PowerShell from the repository root.

## Start the API in stub mode

Stub mode exercises the API contract and kiosk flow without loading or
downloading any AI models.

```powershell
cd "C:\Users\shukl\OneDrive\Documents\hid hack\HIDhackathon\api"

# Remove the incomplete environment, if it exists.
if (Test-Path .venv) { Remove-Item -Recurse -Force .venv }

# Confirm that the intended interpreter is available.
& "C:\Program Files\Python310\python.exe" --version

# Create and activate a clean environment.
& "C:\Program Files\Python310\python.exe" -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies used by the API, stub flow, and tests.
python -m pip install --upgrade pip
python -m pip install -r requirements-core.txt

# Keep model loading disabled for local API/UI development.
$env:STUB_MODE = "1"

# Start the API at http://127.0.0.1:8000.
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

In a second PowerShell window, verify the health endpoint:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

The response must contain `"ok": true` and `"stub_mode": true`. The API routes
are also available at `http://127.0.0.1:8000/docs`.

## Start the frontend separately

Leave the API window running and use another PowerShell window:

```powershell
cd "C:\Users\shukl\OneDrive\Documents\hid hack\HIDhackathon\kiosk-web"
npm install
npm run dev
```

Open the URL printed by Vite, normally `http://localhost:5173`.

## Run tests

From the activated API environment:

```powershell
cd "C:\Users\shukl\OneDrive\Documents\hid hack\HIDhackathon\api"
.\.venv\Scripts\Activate.ps1
python -m pytest -q
```

## Switch to real inference

Do not enable real inference until the model files are available. Stop the
stub server, then install the full dependency set:

```powershell
cd "C:\Users\shukl\OneDrive\Documents\hid hack\HIDhackathon\api"
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python scripts/download_models.py
$env:STUB_MODE = "0"
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

On Windows, `insightface` may require native build tooling and may not have a
compatible wheel. The recommended real-inference path remains the supplied
Docker image. The active liveness challenge implementation is unchanged; the
3D mask/planarity detector remains independently toggled by
`ENABLE_PLANARITY`.
