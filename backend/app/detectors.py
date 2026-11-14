import base64
import cv2
import numpy as np


def _decode_b64_jpeg(frame_b64: str) -> np.ndarray:
    raw = base64.b64decode(frame_b64)
    arr = np.frombuffer(raw, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Invalid image payload")
    return image


def photometric_spoof_score(baseline_b64: str, flash1_b64: str, flash2_b64: str) -> float:
    base = _decode_b64_jpeg(baseline_b64)
    f1 = _decode_b64_jpeg(flash1_b64)
    f2 = _decode_b64_jpeg(flash2_b64)

    base_mean = base.mean(axis=(0, 1))
    f1_mean = f1.mean(axis=(0, 1))
    f2_mean = f2.mean(axis=(0, 1))

    delta_1 = np.abs(f1_mean - base_mean).mean()
    delta_2 = np.abs(f2_mean - base_mean).mean()
    chroma_motion = (delta_1 + delta_2) / 2.0

    # Heuristic: too little chromatic response indicates replay/print.
    normalized = float(np.clip(1.0 - (chroma_motion / 35.0), 0.0, 1.0))
    return normalized


def moire_screen_score(frame_b64: str) -> float:
    frame = _decode_b64_jpeg(frame_b64)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    fft = np.fft.fft2(gray)
    fft_shift = np.fft.fftshift(fft)
    magnitude = np.log1p(np.abs(fft_shift))

    h, w = magnitude.shape
    center_h, center_w = h // 2, w // 2
    center_patch = magnitude[center_h - 20:center_h + 20, center_w - 20:center_w + 20].mean()
    global_mean = magnitude.mean()

    # Large high-frequency energy ratio often indicates display pixel patterns.
    ratio = float(np.clip((global_mean - center_patch + 10.0) / 25.0, 0.0, 1.0))
    return ratio
