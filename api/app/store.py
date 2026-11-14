"""Enrollment gallery: SQLite (edge mode) or DynamoDB (cloud mode). Same interface.

Only the 512-d ArcFace template is stored — never the raw face image.
"""
from __future__ import annotations

import sqlite3
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def _profile(authority="student", active=True, expires_at=None, created_at=None):
    return {
        "authority": authority,
        "active": bool(active),
        "expires_at": expires_at,
        "created_at": created_at or datetime.now(timezone.utc).isoformat(),
    }


class SqliteStore:
    def __init__(self, path: Path):
        self.path = path
        self.lock = threading.Lock()
        with self._conn() as c:
            c.execute("CREATE TABLE IF NOT EXISTS users (user_id TEXT PRIMARY KEY, emb BLOB, created REAL)")
            c.execute("""CREATE TABLE IF NOT EXISTS user_profiles (
                user_id TEXT PRIMARY KEY, authority TEXT NOT NULL DEFAULT 'student',
                active INTEGER NOT NULL DEFAULT 1, expires_at TEXT, created_at TEXT NOT NULL
            )""")

    def _conn(self):
        return sqlite3.connect(self.path)

    def upsert(self, user_id: str, emb: np.ndarray, *, authority="student", active=True, expires_at=None) -> None:
        with self.lock, self._conn() as c:
            c.execute(
                "INSERT OR REPLACE INTO users VALUES (?, ?, ?)",
                (user_id, emb.astype(np.float32).tobytes(), time.time()),
            )
            c.execute(
                "INSERT OR REPLACE INTO user_profiles(user_id, authority, active, expires_at, created_at) "
                "VALUES (?, ?, ?, ?, COALESCE((SELECT created_at FROM user_profiles WHERE user_id = ?), ?))",
                (user_id, authority, int(active), expires_at, user_id, datetime.now(timezone.utc).isoformat()),
            )

    def all(self) -> dict[str, np.ndarray]:
        with self._conn() as c:
            rows = c.execute("SELECT user_id, emb FROM users").fetchall()
        return {u: np.frombuffer(e, dtype=np.float32) for u, e in rows}

    def delete(self, user_id: str) -> None:
        with self.lock, self._conn() as c:
            c.execute("DELETE FROM users WHERE user_id = ?", (user_id,))
            c.execute("DELETE FROM user_profiles WHERE user_id = ?", (user_id,))

    def profile(self, user_id: str):
        with self._conn() as c:
            row = c.execute("SELECT authority, active, expires_at, created_at FROM user_profiles WHERE user_id = ?", (user_id,)).fetchone()
            exists = c.execute("SELECT created FROM users WHERE user_id = ?", (user_id,)).fetchone()
        if row is None and exists is not None:
            return {"user_id": user_id, "authority": "student", "active": True, "expires_at": None,
                    "created_at": datetime.fromtimestamp(exists[0], timezone.utc).isoformat()}
        return None if row is None else {"user_id": user_id, "authority": row[0], "active": bool(row[1]), "expires_at": row[2], "created_at": row[3]}

    def profiles(self):
        with self._conn() as c:
            rows = c.execute("SELECT user_id, authority, active, expires_at, created_at FROM user_profiles ORDER BY user_id").fetchall()
        return [{"user_id": r[0], "authority": r[1], "active": bool(r[2]), "expires_at": r[3], "created_at": r[4]} for r in rows]

    def update_profile(self, user_id, *, authority=None, active=None, expires_at=None):
        current = self.profile(user_id)
        if current is None:
            return None
        with self.lock, self._conn() as c:
            c.execute(
                "UPDATE user_profiles SET authority = ?, active = ?, expires_at = ? WHERE user_id = ?",
                (authority if authority is not None else current["authority"],
                 int(active if active is not None else current["active"]),
                 expires_at if expires_at is not None else current["expires_at"], user_id),
            )
        return self.profile(user_id)


class DynamoStore:
    def __init__(self, table: str, region: str):
        import boto3

        self.table = boto3.resource("dynamodb", region_name=region).Table(table)
        self._cache: dict[str, np.ndarray] | None = None

    def upsert(self, user_id: str, emb: np.ndarray, *, authority="student", active=True, expires_at=None) -> None:
        self.table.put_item(Item={"user_id": user_id, "emb": emb.astype(np.float32).tobytes(), "created": int(time.time()),
                                  "authority": authority, "active": active, "expires_at": expires_at,
                                  "created_at": datetime.now(timezone.utc).isoformat()})
        self._cache = None

    def all(self) -> dict[str, np.ndarray]:
        if self._cache is None:
            items, kw = [], {}
            while True:
                resp = self.table.scan(**kw)
                items += resp["Items"]
                if "LastEvaluatedKey" not in resp:
                    break
                kw["ExclusiveStartKey"] = resp["LastEvaluatedKey"]
            self._cache = {i["user_id"]: np.frombuffer(bytes(i["emb"]), dtype=np.float32) for i in items}
        return self._cache

    def delete(self, user_id: str) -> None:
        self.table.delete_item(Key={"user_id": user_id})
        self._cache = None

    def profile(self, user_id):
        item = self.table.get_item(Key={"user_id": user_id}).get("Item")
        return None if item is None else {"user_id": user_id, "authority": item.get("authority", "student"),
                                           "active": item.get("active", True), "expires_at": item.get("expires_at"),
                                           "created_at": item.get("created_at", datetime.now(timezone.utc).isoformat())}

    def profiles(self):
        return [self.profile(user_id) for user_id in self.all()]

    def update_profile(self, user_id, *, authority=None, active=None, expires_at=None):
        profile = self.profile(user_id)
        if profile is None:
            return None
        item = {"user_id": user_id, "authority": authority or profile["authority"],
                "active": profile["active"] if active is None else active,
                "expires_at": expires_at if expires_at is not None else profile["expires_at"],
                "created_at": profile["created_at"]}
        existing = self.table.get_item(Key={"user_id": user_id}).get("Item", {})
        item["emb"] = existing["emb"]
        item["created"] = existing.get("created", int(time.time()))
        self.table.put_item(Item=item)
        self._cache = None
        return item


def make_store(settings):
    if settings.store == "dynamo":
        return DynamoStore(settings.dynamo_table, settings.aws_region)
    return SqliteStore(settings.sqlite_path)


def search(gallery: dict[str, np.ndarray], emb: np.ndarray, claimed: str | None = None) -> tuple[str | None, float | None]:
    """1:1 if claimed else 1:N. Returns (user_id, cosine similarity)."""
    if not gallery:
        return None, None
    if claimed is not None:
        if claimed not in gallery:
            return claimed, 0.0
        return claimed, float(gallery[claimed] @ emb)
    ids = list(gallery)
    sims = np.stack([gallery[i] for i in ids]) @ emb
    j = int(np.argmax(sims))
    return ids[j], float(sims[j])
