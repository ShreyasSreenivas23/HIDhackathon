import os
from datetime import datetime, timezone

os.environ["STUB_MODE"] = "1"

from fastapi.testclient import TestClient  # noqa: E402

from app import main  # noqa: E402
from app.access import AccessRepository  # noqa: E402


def test_schedule_controls_stub_decisions(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "access", AccessRepository(tmp_path / "access.db"))
    client = TestClient(main.app)
    client.post("/enroll", json={"user_id": "demo_user", "frames": ["x"]})
    schedule = {
        "enabled": True,
        "timezone": "UTC",
        "weekly": {"monday": [{"start": "00:00", "end": "23:59"}]},
        "exceptions": {},
    }
    response = client.put("/access/schedules/demo_user", json=schedule)
    assert response.status_code == 200
    result = client.post("/verify?force=ALLOW", json={"frames": ["x"]}).json()
    assert result["decision"] == "DENY" or result["schedule"]["authorized"] is True
    assert result["schedule"]["timezone"] == "UTC"
    client.delete("/users/demo_user")


def test_stub_enrollment_updates_user_list():
    client = TestClient(main.app)
    user_id = "stub-enrolled-user"
    response = client.post("/enroll", json={"user_id": user_id, "frames": ["x", "y", "z"]})
    assert response.status_code == 200
    assert response.json()["ok"] is True
    assert user_id in client.get("/users").json()["users"]
    client.delete(f"/users/{user_id}")


def test_stub_verification_fails_closed_without_enrollment():
    client = TestClient(main.app)
    response = client.post("/verify?force=ALLOW", json={"frames": ["x"]})
    assert response.status_code == 200
    assert response.json()["decision"] == "DENY"
    assert response.json()["reasons"] == ["unknown_person"]


def test_user_profile_can_expire_and_be_disabled():
    client = TestClient(main.app)
    client.post("/enroll", json={"user_id": "temporary", "frames": ["x"], "authority": "teacher", "expires_in_seconds": 60})
    profile = next(item for item in client.get("/users/details").json() if item["user_id"] == "temporary")
    assert profile["authority"] == "teacher"
    assert profile["active"] is True
    assert profile["expires_at"] is not None
    cleared = client.patch("/users/temporary", json={"expires_in_seconds": 0}).json()
    assert cleared["expires_at"] is None
    assert client.post("/verify?force=ALLOW", json={"frames": ["x"]}).json()["decision"] == "ALLOW"
    updated = client.patch("/users/temporary", json={"active": False}).json()
    assert updated["active"] is False
    denied = client.post("/verify?force=ALLOW", json={"frames": ["x"]}).json()
    assert denied["decision"] == "DENY"
    assert denied["reasons"] == ["user_inactive"]
    client.delete("/users/temporary")


def test_offline_verify_requires_cached_permission(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "access", AccessRepository(tmp_path / "access.db"))
    client = TestClient(main.app)
    denied = client.post("/offline/verify", json={
        "user_id": "unknown",
        "liveness": 0.99,
        "identity_similarity": 0.99,
    }).json()
    assert denied["decision"] == "DENY"
    assert denied["reasons"] == ["no_cached_permission"]


def test_offline_sync_rejects_stale_allow(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "access", AccessRepository(tmp_path / "access.db"))
    client = TestClient(main.app)
    response = client.post("/offline/sync", json={
        "events": [{
            "event_id": "stale-1",
            "user_id": "alice",
            "decision": "ALLOW",
            "occurred_at": datetime(2026, 1, 1, tzinfo=timezone.utc).isoformat(),
            "payload": {},
        }]
    })
    assert response.status_code == 200
    assert response.json()["rejected"][0]["reason"] == "stale_or_invalid_permission"
