# Cloud image: API + built kiosk UI on ONE port (App Runner gives it an HTTPS URL).
# Build context = data/deploy_ctx (made by deploy/stage.sh): code, models, DINOv2 cache, kiosk dist, seed DB.
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /srv

RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
COPY api/requirements-core.txt /srv/api/requirements-core.txt
RUN pip install --no-cache-dir -r /srv/api/requirements-core.txt ultralytics

COPY torch /srv/torch
COPY api /srv/api
COPY kiosk-web/dist /srv/kiosk-web/dist

ENV FACE_BACKEND=opencv \
    TORCH_HOME=/srv/torch \
    MODELS_DIR=/srv/api/models \
    SQLITE_PATH=/srv/api/enrollments.db \
    KIOSK_DIST=/srv/kiosk-web/dist \
    PQC_KEY_DIR=/srv/api/keys \
    YOLO_CONFIG_DIR=/tmp/ultralytics \
    PYTHONUNBUFFERED=1 \
    PORT=8080
EXPOSE 8080
WORKDIR /srv/api
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT} --workers 1"]
