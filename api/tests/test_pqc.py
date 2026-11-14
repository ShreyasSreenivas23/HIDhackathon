import json
import tempfile
from pathlib import Path

from app.pqc import PQSigner, kem_keygen, kem_unwrap, kem_wrap


def test_mldsa_sign_verify_and_tamper():
    s = PQSigner(Path(tempfile.mkdtemp()))
    att = s.sign({"decision": "ALLOW", "user_id": "alice"})
    assert att["alg"] == "ML-DSA-65" and s.verify(att["payload"], att["signature_b64"])
    p = json.loads(att["payload"])
    assert p["expires_at"] > p["issued_at"] and len(p["decision_id"]) == 24
    forged = att["payload"].replace('"ALLOW"', '"DENY"')
    assert not s.verify(forged, att["signature_b64"])


def test_keys_persist_across_restarts():
    d = Path(tempfile.mkdtemp())
    assert PQSigner(d).key_id == PQSigner(d).key_id


def test_mlkem_roundtrip():
    ek, dk = kem_keygen()
    k1, ct = kem_wrap(ek)
    assert kem_unwrap(dk, ct) == k1 and len(k1) == 32
