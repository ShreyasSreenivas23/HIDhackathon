#!/usr/bin/env bash
# Assemble the docker build context for the cloud image in data/deploy_ctx (run from repo root, Git Bash).
set -euo pipefail
cd "$(dirname "$0")/.."
CTX=data/deploy_ctx
rm -rf "$CTX" && mkdir -p "$CTX/api" "$CTX/kiosk-web" "$CTX/torch"

# kiosk built for same-origin API calls
(cd kiosk-web && VITE_API_BASE="" npm run build >/dev/null)
cp -r kiosk-web/dist "$CTX/kiosk-web/dist"

# API code + config + models + seed data (current enrollments & schedules)
cp -r api/app api/requirements-core.txt api/thresholds.json "$CTX/api/"
mkdir -p "$CTX/api/models"
cp api/models/minifas_best_model.onnx api/models/yolo11n.pt api/models/face_detection_yunet_2023mar.onnx \
   api/models/face_recognition_sface_2021dec.onnx api/models/dino_head.npz "$CTX/api/models/"
[ -f api/enrollments.db ] && cp api/enrollments.db "$CTX/api/enrollments.db"
find "$CTX/api" -name __pycache__ -type d -prune -exec rm -rf {} +

# DINOv2 code + weights from the local torch hub cache (no download at container start)
cp -r "$HOME/.cache/torch/hub" "$CTX/torch/hub"

cp deploy/Dockerfile "$CTX/Dockerfile"
du -sh "$CTX"
