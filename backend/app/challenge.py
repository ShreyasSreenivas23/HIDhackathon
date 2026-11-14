import time
import uuid
from jose import jwt

from app.config import settings

_COLOR_SEQUENCES = [
    ["cyan", "magenta"],
    ["blue", "yellow"],
    ["green", "red"],
    ["white", "blue"],
]

_PROMPT_HINTS = [
    "Look straight at the camera.",
    "Keep your face centered in the guide.",
    "Do not move during the flash capture.",
]


def build_challenge() -> tuple[str, str, list[str], str]:
    session_id = str(uuid.uuid4())
    nonce = str(uuid.uuid4())
    now = int(time.time())
    colors = _COLOR_SEQUENCES[now % len(_COLOR_SEQUENCES)]
    prompt_hint = _PROMPT_HINTS[now % len(_PROMPT_HINTS)]

    payload = {
        "sid": session_id,
        "jti": nonce,
        "iat": now,
        "exp": now + settings.challenge_ttl_seconds,
        "flash": colors,
    }
    token = jwt.encode(payload, settings.challenge_signing_key, algorithm="HS256")
    return session_id, token, colors, prompt_hint


def verify_challenge(token: str, session_id: str) -> dict:
    payload = jwt.decode(token, settings.challenge_signing_key, algorithms=["HS256"])
    if payload.get("sid") != session_id:
        raise ValueError("session mismatch")
    return payload
