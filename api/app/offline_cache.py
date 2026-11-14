"""Offline cache: keep the edge (laptop) copy of faces, profiles and schedules in sync with the central API.

Central side : build_export()  -> served at GET /offline/cache-export (admin only). Face TEMPLATES only, never photos.
Edge side    : CacheSyncer     -> every CACHE_REFRESH_S (default 60) while the central API is reachable, pulls the
               export and MIRRORS it locally, so an offline decision uses the same people, expiry dates and
               schedules as the cloud. If the central API is unreachable, the last good cache is kept untouched.
Configure the edge with CENTRAL_API_BASE (e.g. https://<id>.cloudfront.net) and CENTRAL_ADMIN_SECRET
(the central admin PIN/password). Without CENTRAL_API_BASE nothing runs (the central API itself never syncs).
"""
from __future__ import annotations

import base64
import logging
import os
import shutil
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .access import schedule_from_payload, schedule_payload

log = logging.getLogger(__name__)


def build_export(store, access) -> dict:
    users = []
    for uid, emb in store.all().items():
        prof = (store.profile(uid) if hasattr(store, "profile") else None) or {}
        users.append({
            "user_id": uid,
            "emb_b64": base64.b64encode(np.asarray(emb, np.float32).tobytes()).decode(),
            "authority": prof.get("authority", "student"),
            "active": bool(prof.get("active", True)),
            "expires_at": prof.get("expires_at"),
        })
    return {"generated_at": datetime.now(timezone.utc).isoformat(), "users": users,
            "schedules": [schedule_payload(s) for s in access.all_schedules()]}


def mirror(store, access, export: dict) -> dict:
    """Make the local cache equal to the central export (add/update/remove)."""
    central_users = {u["user_id"]: u for u in export.get("users", [])}
    for uid, u in central_users.items():
        emb = np.frombuffer(base64.b64decode(u["emb_b64"]), dtype=np.float32)
        store.upsert(uid, emb, authority=u.get("authority", "student"), active=u.get("active", True),
                     expires_at=u.get("expires_at"))
    removed = [uid for uid in list(store.all()) if uid not in central_users]
    for uid in removed:
        store.delete(uid)
    central_sched = {s["user_id"]: s for s in export.get("schedules", [])}
    for uid, s in central_sched.items():
        access.put_schedule(schedule_from_payload(uid, s, s.get("updated_at")))
    for s in access.all_schedules():
        if s.user_id not in central_sched:
            access.delete_schedule(s.user_id)
    return {"users": len(central_users), "schedules": len(central_sched), "removed": removed}


class CacheSyncer:
    def __init__(self, store, access, db_path: Path):
        self.base = (os.getenv("CENTRAL_API_BASE") or "").rstrip("/")
        self.secret = os.getenv("CENTRAL_ADMIN_SECRET", "2468")
        self.every = int(os.getenv("CACHE_REFRESH_S", "60"))
        self.store, self.access, self.db_path = store, access, db_path
        self.status = {"enabled": bool(self.base), "central": self.base or None, "last_ok": None,
                       "last_error": None, "users": None, "schedules": None}
        self._lock = threading.Lock()
        self._backed_up = False

    def refresh(self) -> dict:
        if not self.base:
            return self.status | {"error": "CENTRAL_API_BASE not set"}
        import httpx

        with self._lock:
            try:
                r = httpx.get(f"{self.base}/offline/cache-export", timeout=15,
                              headers={"X-Admin-Pin": self.secret, "X-Admin-Password": self.secret})
                r.raise_for_status()
                export = r.json()
                if not self._backed_up and self.db_path.exists():   # keep the pre-sync local data once
                    shutil.copy2(self.db_path, self.db_path.with_suffix(".pre-sync-backup.db"))
                    self._backed_up = True
                res = mirror(self.store, self.access, export)
                self.status.update(last_ok=datetime.now(timezone.utc).isoformat(), last_error=None,
                                   users=res["users"], schedules=res["schedules"])
                if res["removed"]:
                    log.info("offline cache: removed %s (not on central)", res["removed"])
            except Exception as e:      # central unreachable -> keep the last good cache
                self.status["last_error"] = f"{type(e).__name__}: {e}"[:200]
        return self.status

    def start(self) -> None:
        if not self.base:
            return

        def loop():
            while True:
                self.refresh()
                time.sleep(self.every)

        threading.Thread(target=loop, daemon=True, name="offline-cache-sync").start()
        log.info("offline cache sync from %s every %ss", self.base, self.every)
