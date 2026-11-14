from datetime import datetime, timezone

from app.access import AccessRepository, schedule_from_payload


def schedule(**overrides):
    payload = {
        "enabled": True,
        "timezone": "UTC",
        "weekly": {day: [] for day in ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")},
        "exceptions": {},
    }
    payload["weekly"]["monday"] = [{"start": "09:00", "end": "17:00"}]
    payload.update(overrides)
    return schedule_from_payload("alice", payload)


def test_schedule_allows_inside_and_denies_boundary():
    item = schedule()
    assert item.evaluate(datetime(2026, 11, 16, 9, 0, tzinfo=timezone.utc))[0]
    assert not item.evaluate(datetime(2026, 11, 16, 17, 0, tzinfo=timezone.utc))[0]
    assert not item.evaluate(datetime(2026, 11, 17, 10, 0, tzinfo=timezone.utc))[0]


def test_disabled_and_date_exception_fail_closed():
    disabled = schedule(enabled=False)
    assert disabled.evaluate(datetime(2026, 11, 16, 10, tzinfo=timezone.utc))[1] == "permission_disabled"
    blocked = schedule(exceptions={"2026-11-16": None})
    assert blocked.evaluate(datetime(2026, 11, 16, 10, tzinfo=timezone.utc))[1] == "date_exception"


def test_schedule_timezone_is_authoritative():
    item = schedule(timezone="America/New_York")
    # 14:00 UTC is 09:00 in New York during standard time.
    allowed, reason, zone = item.evaluate(datetime(2026, 11, 16, 14, 0, tzinfo=timezone.utc))
    assert allowed and reason == "scheduled_access" and zone == "America/New_York"


def test_offline_event_queue_is_idempotent(tmp_path):
    repo = AccessRepository(tmp_path / "access.db")
    when = datetime(2026, 11, 16, 10, tzinfo=timezone.utc)
    event = {"reason": "local"}
    first = repo.queue_event("alice", "DENY", when, event, "event-1")
    second = repo.queue_event("alice", "DENY", when, event, "event-1")
    assert first == second == "event-1"
    assert len(repo.pending_events()) == 1
    repo.mark_synced(["event-1"])
    assert repo.event_status() == {"pending": 0, "synced": 1}


def test_expired_duplicate_does_not_shadow_valid_enrollment():
    import numpy as np
    from types import SimpleNamespace
    from app.config import Thresholds
    from app.pipeline import Pipeline

    e = np.zeros(8, np.float32); e[0] = 1
    near = e.copy(); near[1] = 0.05; near /= np.linalg.norm(near)
    profiles = {"SHREYAS": {"active": True, "expires_at": "2000-01-01T00:00:00+00:00"}, "shreyas": {"active": True, "expires_at": None}}
    p = Pipeline.__new__(Pipeline)
    p.th = Thresholds()
    p.store = SimpleNamespace(all=lambda: {"SHREYAS": e, "shreyas": near}, profile=lambda u: profiles[u])
    assert p._best_valid_match(e)[0] == "shreyas"                 # valid duplicate wins
    profiles["shreyas"]["active"] = False
    assert p._best_valid_match(e)[0] == "SHREYAS"                 # only invalid ones -> report top (expired)
