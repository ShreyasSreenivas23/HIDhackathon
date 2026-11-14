"""Image decoding, crops, quality metrics, geometry helpers."""
from __future__ import annotations

import base64

import cv2
import numpy as np


def decode_b64_image(data: str) -> np.ndarray:
    """base64 JPEG/PNG (optionally a data URL) -> BGR uint8 image."""
    if "," in data[:64]:
        data = data.split(",", 1)[1]
    buf = np.frombuffer(base64.b64decode(data), dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("could not decode image")
    return img


def encode_jpeg(img: np.ndarray, quality: int = 85) -> bytes:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes() if ok else b""


def square_crop(img: np.ndarray, bbox, expansion: float = 1.5) -> np.ndarray:
    """Square crop centred on bbox [x1,y1,x2,y2], side = max(w,h)*expansion, reflect-padded at borders.
    Same convention as face-antispoof-onnx `crop()` — training/ uses this too, keep them identical."""
    h_img, w_img = img.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
    w, h = x2 - x1, y2 - y1
    if w <= 0 or h <= 0:
        raise ValueError("invalid bbox")
    side = int(max(w, h) * expansion)
    cx, cy = x1 + w / 2, y1 + h / 2
    x, y = int(cx - side / 2), int(cy - side / 2)
    cx1, cy1 = max(0, x), max(0, y)
    cx2, cy2 = min(w_img, x + side), min(h_img, y + side)
    crop = img[cy1:cy2, cx1:cx2]
    top, left = max(0, -y), max(0, -x)
    bottom, right = max(0, y + side - h_img), max(0, x + side - w_img)
    if crop.size == 0:
        return np.zeros((side, side, 3), dtype=img.dtype)
    border = cv2.BORDER_REFLECT_101 if min(crop.shape[:2]) > 1 else cv2.BORDER_REPLICATE
    return cv2.copyMakeBorder(crop, top, bottom, left, right, border)


def letterbox(img: np.ndarray, size: int) -> np.ndarray:
    h, w = img.shape[:2]
    r = size / max(h, w)
    nh, nw = int(h * r), int(w * r)
    interp = cv2.INTER_LANCZOS4 if r > 1 else cv2.INTER_AREA
    img = cv2.resize(img, (nw, nh), interpolation=interp)
    dh, dw = size - nh, size - nw
    return cv2.copyMakeBorder(img, dh // 2, dh - dh // 2, dw // 2, dw - dw // 2, cv2.BORDER_REFLECT_101)


BLUR_SIZE = 112   # measure sharpness on a fixed-size face so the score does not depend on resolution


def face_quality(img: np.ndarray, bbox) -> dict:
    """blur = Laplacian variance of the face resized to 112x112 (scale-independent; measured on a test face:
    sharp ~280, soft webcam ~130, blurry ~50, very blurry ~15), brightness = mean gray in [0,1],
    face_px = bbox short side."""
    x1, y1, x2, y2 = [int(v) for v in bbox[:4]]
    h, w = img.shape[:2]
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    face = img[y1:y2, x1:x2]
    if face.size == 0:
        return {"blur": 0.0, "brightness": 0.0, "face_px": 0.0}
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    norm = cv2.resize(gray, (BLUR_SIZE, BLUR_SIZE), interpolation=cv2.INTER_AREA)
    return {
        "blur": float(cv2.Laplacian(norm, cv2.CV_64F).var()),
        "brightness": float(gray.mean() / 255.0),
        "face_px": float(min(x2 - x1, y2 - y1)),
    }


def yaw_proxy(kps) -> float:
    """(nose_x - eye_mid_x) / eye_distance on RAW (unmirrored) frames. >0: user turned to THEIR left.
    kps = insightface 5 points: [eye_imgleft, eye_imgright, nose, mouth_imgleft, mouth_imgright]."""
    kps = np.asarray(kps, dtype=np.float32)
    eye_mid = (kps[0] + kps[1]) / 2
    eye_dist = float(np.linalg.norm(kps[1] - kps[0])) + 1e-6
    return float((kps[2, 0] - eye_mid[0]) / eye_dist)


def bbox_area(b) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def intersection_over_first(a, b) -> float:
    """Fraction of box a covered by box b."""
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    return inter / (bbox_area(a) + 1e-6)
