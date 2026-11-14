"""Post-quantum cryptography: every access decision is signed with ML-DSA-65 (NIST FIPS 204).

Why: the decision is what opens the door. A door controller (here: the kiosk) must only act on a decision the
decision service really issued, unmodified, recently, and only once. Classical signatures (RSA/ECDSA) can be
forged by a future quantum computer; ML-DSA is the NIST post-quantum signature standard.

  * keypair generated on first start, kept in api/keys/ (gitignored); key_id = SHA-256(pk)[:16]
  * attestation = canonical JSON payload + detached signature (base64)
  * payload carries issued_at / expires_at (30 s) and a random decision_id -> verifiers reject stale or
    replayed decisions
  * GET /pqc/public-key publishes the verification key; the kiosk verifies with @noble/post-quantum
    (cross-checked: signatures from this module verify in noble, tampered payloads are rejected)
Implementation: dilithium-py (pure Python FIPS 204), ~60 ms per signature on the laptop CPU.
ML-KEM-768 (FIPS 203, kyber-py) is used by kem_wrap/kem_unwrap for post-quantum key encapsulation.
"""
from __future__ import annotations

import base64
import hashlib
import json
import logging
import secrets
import time
from pathlib import Path

log = logging.getLogger(__name__)
ALG = "ML-DSA-65"
TTL_S = 30


def _b64(b: bytes) -> str:
    return base64.b64encode(b).decode()


class PQSigner:
    def __init__(self, key_dir: Path):
        from dilithium_py.ml_dsa import ML_DSA_65

        self.scheme = ML_DSA_65
        key_dir.mkdir(parents=True, exist_ok=True)
        pk_f, sk_f = key_dir / "mldsa65.pk", key_dir / "mldsa65.sk"
        if pk_f.exists() and sk_f.exists():
            self.pk, self.sk = pk_f.read_bytes(), sk_f.read_bytes()
        else:
            self.pk, self.sk = ML_DSA_65.keygen()
            pk_f.write_bytes(self.pk)
            sk_f.write_bytes(self.sk)
            log.info("generated new %s keypair in %s", ALG, key_dir)
        self.key_id = hashlib.sha256(self.pk).hexdigest()[:16]

    def public_info(self) -> dict:
        return {"alg": ALG, "key_id": self.key_id, "public_key_b64": _b64(self.pk), "standard": "NIST FIPS 204"}

    @staticmethod
    def canonical(payload: dict) -> bytes:
        return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()

    def sign(self, payload: dict) -> dict:
        now = time.time()
        payload = dict(payload, issued_at=round(now, 3), expires_at=round(now + TTL_S, 3),
                       decision_id=secrets.token_hex(12), key_id=self.key_id)
        body = self.canonical(payload)
        return {"alg": ALG, "key_id": self.key_id, "payload": body.decode(), "signature_b64": _b64(self.scheme.sign(self.sk, body))}

    def verify(self, payload: str, signature_b64: str) -> bool:
        try:
            return bool(self.scheme.verify(self.pk, payload.encode(), base64.b64decode(signature_b64)))
        except Exception:
            return False


def decision_payload(resp) -> dict:
    """What gets signed: everything a door controller acts on (not the raw images)."""
    return {
        "decision": resp.decision,
        "user_id": resp.identity.user_id,
        "reasons": list(resp.reasons or []),
        "confidence": round(float(resp.confidence), 4),
        "liveness": None if resp.liveness.score is None else round(float(resp.liveness.score), 4),
        "similarity": None if resp.identity.similarity is None else round(float(resp.identity.similarity), 4),
    }


# ---- ML-KEM-768 (FIPS 203): post-quantum key encapsulation helpers ----
def kem_keygen() -> tuple[bytes, bytes]:
    from kyber_py.ml_kem import ML_KEM_768

    return ML_KEM_768.keygen()          # (encapsulation key, decapsulation key)


def kem_wrap(ek: bytes) -> tuple[bytes, bytes]:
    from kyber_py.ml_kem import ML_KEM_768

    return ML_KEM_768.encaps(ek)        # (shared_secret, ciphertext)


def kem_unwrap(dk: bytes, ct: bytes) -> bytes:
    from kyber_py.ml_kem import ML_KEM_768

    return ML_KEM_768.decaps(dk, ct)
