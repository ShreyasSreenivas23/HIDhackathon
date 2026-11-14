"""Settings + decision thresholds. Thresholds are overridden by THRESHOLDS_FILE (written by eval/calibrate.py)."""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

API_DIR = Path(__file__).resolve().parent.parent


def _flag(name: str, default: str = "1") -> bool:
    return os.getenv(name, default).strip().lower() not in ("0", "false", "no", "")


@dataclass
class Thresholds:
    # liveness (fused score, 1 = live)
    liveness_lo: float = 0.35          # below -> DENY
    liveness_hi: float = 0.75          # at/above (and identity_hi) -> ALLOW
    # identity (ArcFace cosine similarity)
    identity_lo: float = 0.28          # below -> DENY (unknown / mismatch)
    identity_hi: float = 0.40          # at/above -> confident match
    # signals
    device_hard: float = 0.30          # device signal below this -> hard DENY (face is inside a phone/laptop/tv)
    planar_hard: float = 0.20          # planarity below this (clearly flat face) + low liveness -> hard DENY
    signal_flag: float = 0.40          # any signal below this is listed in `reasons`
    challenge_yaw: float = 0.12        # server-side yaw change vs own frontal pose to accept a turn (noise ~0.03; a tilted flat photo < 0.05)
    # quality gate (server side; client has its own copy)
    min_face_px: int = 90
    min_blur: float = 25.0             # Laplacian variance on the face resized to 112px (soft webcam ~130, very blurry ~15)
    min_brightness: float = 0.18
    max_brightness: float = 0.92
    max_abs_yaw: float = 0.35
    # fusion weights for soft signals (device is a hard veto, not weighted)
    weights: dict = field(default_factory=lambda: {"minifas": 1.0, "dino": 1.5, "planarity": 0.7})

    @classmethod
    def load(cls, path: Path) -> "Thresholds":
        th = cls()
        if path.exists():
            data = json.loads(path.read_text())
            for k, v in data.items():
                if hasattr(th, k):
                    setattr(th, k, v)
        return th

    def to_dict(self) -> dict:
        return asdict(self)


class Settings:
    def __init__(self) -> None:
        self.stub_mode = _flag("STUB_MODE", "0")
        self.models_dir = Path(os.getenv("MODELS_DIR", str(API_DIR / "models")))
        self.store = os.getenv("STORE", "sqlite")
        self.sqlite_path = Path(os.getenv("SQLITE_PATH", str(API_DIR / "enrollments.db")))
        self.dynamo_table = os.getenv("DYNAMO_TABLE", "hid-enrollments")
        self.audit_bucket = os.getenv("AUDIT_BUCKET") or None
        self.aws_region = os.getenv("AWS_REGION", "us-east-1")
        self.thresholds_file = Path(os.getenv("THRESHOLDS_FILE", str(API_DIR / "thresholds.json")))
        self.det_size = int(os.getenv("DET_SIZE", "320"))   # SCRFD input; kiosk faces are large, 320 is fast
        self.enable_minifas = _flag("ENABLE_MINIFAS")
        self.enable_dino = _flag("ENABLE_DINO")
        self.enable_device = _flag("ENABLE_DEVICE")
        self.enable_planarity = _flag("ENABLE_PLANARITY")
        self.minifas_model = self.models_dir / os.getenv("MINIFAS_FILE", "minifas_best_model.onnx")
        self.dino_head = self.models_dir / os.getenv("DINO_HEAD_FILE", "dino_head.joblib")
        self.yolo_model = self.models_dir / os.getenv("YOLO_FILE", "yolo11n.pt")
        self.thresholds = Thresholds.load(self.thresholds_file)


settings = Settings()
