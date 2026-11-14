import os

os.environ["STUB_MODE"] = "1"

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402

c = TestClient(app)


def test_stub_flow():
    assert c.get("/health").json()["stub_mode"] is True
    c.post("/enroll", json={"user_id": "demo_user", "frames": ["x"]})
    assert c.post("/verify", json={"frames": ["x"]}).json()["decision"] == "ALLOW"
    assert c.post("/verify?force=DENY", json={"frames": ["x"]}).json()["decision"] == "DENY"
    r = c.post("/verify?force=REQUEST_VERIFICATION", json={"frames": ["x"]}).json()
    assert r["challenge"]["actions"]
    assert c.post("/challenge", json={"nonce": "stub01", "frames": ["x"]}).json()["decision"] == "ALLOW"
    code = c.post("/otp/request", json={"user_id": "a"}).json()["demo_code"]
    assert c.post("/otp/verify", json={"user_id": "a", "code": code}).json()["decision"] == "ALLOW"
    assert c.post("/otp/verify", json={"user_id": "a", "code": "000000"}).json()["decision"] == "DENY"
