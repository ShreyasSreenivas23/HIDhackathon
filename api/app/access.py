"""Campus access schedules and fail-closed offline event storage."""
from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, time as dt_time, timezone
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


DAY_NAMES = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")


@dataclass(frozen=True)
class AccessWindow:
    start: dt_time
    end: dt_time

    def contains(self, value: dt_time) -> bool:
        if self.start == self.end:
            return True
        if self.start < self.end:
            return self.start <= value < self.end
        return value >= self.start or value < self.end


@dataclass(frozen=True)
class AccessSchedule:
    user_id: str
    enabled: bool
    timezone: str
    weekly: dict[str, tuple[AccessWindow, ...]]
    exceptions: dict[str, tuple[AccessWindow, ...] | None]
    updated_at: float

    def evaluate(self, at: datetime | None = None) -> tuple[bool, str, str]:
        try:
            zone = ZoneInfo(self.timezone)
        except ZoneInfoNotFoundError:
            return False, "invalid_timezone", self.timezone
        current = (at or datetime.now(timezone.utc)).astimezone(zone)
        if not self.enabled:
            return False, "permission_disabled", self.timezone
        key = current.date().isoformat()
        windows = self.exceptions.get(key, self.weekly.get(DAY_NAMES[current.weekday()], ()))
        if windows is None:
            return False, "date_exception", self.timezone
        if any(window.contains(current.time()) for window in windows):
            return True, "scheduled_access", self.timezone
        return False, "outside_schedule", self.timezone


def _parse_time(value: str) -> dt_time:
    try:
        return dt_time.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"invalid time window value: {value!r}") from exc


def _windows(data: dict[str, list[dict[str, str]]]) -> dict[str, tuple[AccessWindow, ...]]:
    result: dict[str, tuple[AccessWindow, ...]] = {}
    for day, entries in data.items():
        if day not in DAY_NAMES:
            raise ValueError(f"invalid access day: {day}")
        parsed = tuple(AccessWindow(_parse_time(item["start"]), _parse_time(item["end"])) for item in entries)
        result[day] = parsed
    return result


def schedule_from_payload(user_id: str, payload: dict, updated_at: float | None = None) -> AccessSchedule:
    exceptions: dict[str, tuple[AccessWindow, ...] | None] = {}
    for day, entries in payload.get("exceptions", {}).items():
        if entries is None:
            exceptions[day] = None
        else:
            exceptions[day] = tuple(AccessWindow(_parse_time(item["start"]), _parse_time(item["end"])) for item in entries)
    return AccessSchedule(
        user_id=user_id,
        enabled=bool(payload.get("enabled", False)),
        timezone=str(payload.get("timezone", "UTC")),
        weekly=_windows(payload.get("weekly", {})),
        exceptions=exceptions,
        updated_at=updated_at or time.time(),
    )


def schedule_payload(schedule: AccessSchedule) -> dict:
    def serialize(windows: tuple[AccessWindow, ...]) -> list[dict[str, str]]:
        return [{"start": w.start.isoformat(timespec="minutes"), "end": w.end.isoformat(timespec="minutes")} for w in windows]

    return {
        "user_id": schedule.user_id,
        "enabled": schedule.enabled,
        "timezone": schedule.timezone,
        "weekly": {day: serialize(windows) for day, windows in schedule.weekly.items()},
        "exceptions": {
            day: None if windows is None else serialize(windows)
            for day, windows in schedule.exceptions.items()
        },
        "updated_at": schedule.updated_at,
    }


class AccessRepository:
    """Local schedule cache and idempotent offline decision queue."""

    def __init__(self, path: Path):
        self.path = path
        self.lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as conn:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS access_schedules (
                    user_id TEXT PRIMARY KEY, payload TEXT NOT NULL, updated_at REAL NOT NULL
                )"""
            )
            conn.execute(
                """CREATE TABLE IF NOT EXISTS offline_events (
                    event_id TEXT PRIMARY KEY, user_id TEXT, decision TEXT NOT NULL,
                    occurred_at TEXT NOT NULL, payload TEXT NOT NULL,
                    sync_status TEXT NOT NULL DEFAULT 'pending', synced_at REAL
                )"""
            )

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)

    def put_schedule(self, schedule: AccessSchedule) -> AccessSchedule:
        payload = schedule_payload(schedule)
        with self.lock, self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO access_schedules(user_id, payload, updated_at) VALUES (?, ?, ?)",
                (schedule.user_id, json.dumps(payload), schedule.updated_at),
            )
        return schedule

    def get_schedule(self, user_id: str) -> AccessSchedule | None:
        with self._conn() as conn:
            row = conn.execute("SELECT payload, updated_at FROM access_schedules WHERE user_id = ?", (user_id,)).fetchone()
        return None if row is None else schedule_from_payload(user_id, json.loads(row[0]), row[1])

    def all_schedules(self) -> list[AccessSchedule]:
        with self._conn() as conn:
            rows = conn.execute("SELECT payload, updated_at FROM access_schedules ORDER BY user_id").fetchall()
        return [schedule_from_payload(json.loads(row[0])["user_id"], json.loads(row[0]), row[1]) for row in rows]

    def delete_schedule(self, user_id: str) -> None:
        with self.lock, self._conn() as conn:
            conn.execute("DELETE FROM access_schedules WHERE user_id = ?", (user_id,))

    def queue_event(self, user_id: str | None, decision: str, occurred_at: datetime, payload: dict, event_id: str | None = None) -> str:
        event_id = event_id or str(uuid.uuid4())
        with self.lock, self._conn() as conn:
            conn.execute(
                """INSERT OR IGNORE INTO offline_events
                   (event_id, user_id, decision, occurred_at, payload, sync_status)
                   VALUES (?, ?, ?, ?, ?, 'pending')""",
                (event_id, user_id, decision, occurred_at.astimezone(timezone.utc).isoformat(), json.dumps(payload)),
            )
        return event_id

    def pending_events(self, limit: int = 100) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT event_id, user_id, decision, occurred_at, payload, sync_status FROM offline_events "
                "WHERE sync_status = 'pending' ORDER BY occurred_at LIMIT ?",
                (limit,),
            ).fetchall()
        return [
            {"event_id": r[0], "user_id": r[1], "decision": r[2], "occurred_at": r[3],
             "payload": json.loads(r[4]), "sync_status": r[5]}
            for r in rows
        ]

    def mark_synced(self, event_ids: list[str]) -> None:
        if not event_ids:
            return
        with self.lock, self._conn() as conn:
            conn.executemany(
                "UPDATE offline_events SET sync_status = 'synced', synced_at = ? WHERE event_id = ? AND sync_status = 'pending'",
                [(time.time(), event_id) for event_id in event_ids],
            )

    def event_status(self) -> dict[str, int]:
        with self._conn() as conn:
            rows = conn.execute("SELECT sync_status, COUNT(*) FROM offline_events GROUP BY sync_status").fetchall()
        return {"pending": sum(count for status, count in rows if status == "pending"),
                "synced": sum(count for status, count in rows if status == "synced")}
