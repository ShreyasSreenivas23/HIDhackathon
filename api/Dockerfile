FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential g++ libgl1 libglib2.0-0 git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv
COPY requirements-core.txt requirements.txt ./
# CPU-only torch keeps the image small; inference models here are tiny
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY scripts ./scripts
COPY thresholds.json ./thresholds.json

ENV MODELS_DIR=/srv/models SQLITE_PATH=/srv/data/enrollments.db PYTHONUNBUFFERED=1
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
