"""Enrollment gallery: SQLite (edge mode) or DynamoDB (cloud mode). Same interface.

Only the 512-d ArcFace template is stored — never the raw face image.
"""
from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path

import numpy as np


class SqliteStore:
    def __init__(self, path: Path):
        self.path = path
        self.lock = threading.Lock()
        with self._conn() as c:
            c.execute("CREATE TABLE IF NOT EXISTS users (user_id TEXT PRIMARY KEY, emb BLOB, created REAL)")

    def _conn(self):
        return sqlite3.connect(self.path)

    def upsert(self, user_id: str, emb: np.ndarray) -> None:
        with self.lock, self._conn() as c:
            c.execute(
                "INSERT OR REPLACE INTO users VALUES (?, ?, ?)",
                (user_id, emb.astype(np.float32).tobytes(), time.time()),
            )

    def all(self) -> dict[str, np.ndarray]:
        with self._conn() as c:
            rows = c.execute("SELECT user_id, emb FROM users").fetchall()
        return {u: np.frombuffer(e, dtype=np.float32) for u, e in rows}

    def delete(self, user_id: str) -> None:
        with self.lock, self._conn() as c:
            c.execute("DELETE FROM users WHERE user_id = ?", (user_id,))


class DynamoStore:
    def __init__(self, table: str, region: str):
        import boto3

        self.table = boto3.resource("dynamodb", region_name=region).Table(table)
        self._cache: dict[str, np.ndarray] | None = None

    def upsert(self, user_id: str, emb: np.ndarray) -> None:
        self.table.put_item(Item={"user_id": user_id, "emb": emb.astype(np.float32).tobytes(), "created": int(time.time())})
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
