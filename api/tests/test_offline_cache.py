import numpy as np

from app.access import AccessRepository, schedule_from_payload
from app.offline_cache import build_export, mirror
from app.store import SqliteStore


def test_edge_cache_mirrors_central(tmp_path):
    cen, cacc = SqliteStore(tmp_path / "c.db"), AccessRepository(tmp_path / "c.db")
    edge, eacc = SqliteStore(tmp_path / "e.db"), AccessRepository(tmp_path / "e.db")
    cen.upsert("shreyas", np.ones(4, np.float32), expires_at=None)
    cen.upsert("siri", np.full(4, 2, np.float32))
    cacc.put_schedule(schedule_from_payload("shreyas", {"enabled": True, "timezone": "UTC", "weekly": {}, "exceptions": {}}))
    # edge has a stale, EXPIRED shreyas and someone central no longer has
    edge.upsert("shreyas", np.zeros(4, np.float32), expires_at="2000-01-01T00:00:00+00:00")
    edge.upsert("gone", np.zeros(4, np.float32))
    eacc.put_schedule(schedule_from_payload("gone", {"enabled": True, "timezone": "UTC", "weekly": {}, "exceptions": {}}))

    res = mirror(edge, eacc, build_export(cen, cacc))
    assert res["removed"] == ["gone"]
    assert set(edge.all()) == {"shreyas", "siri"}
    assert edge.profile("shreyas")["expires_at"] is None                    # expiry now matches central
    assert np.allclose(edge.all()["shreyas"], 1.0)                          # template replaced
    assert [s.user_id for s in eacc.all_schedules()] == ["shreyas"]
