"""Optional AWS integrations, ported from siri's `backend/` (branch `siri`) onto this API.

All OFF unless configured, never on the critical path, and never able to break a decision:
  AWS_AUDIT_TABLE=AccessAuditLogs        -> every decision written to DynamoDB (background thread)
  ENROLLED_FACES_BUCKET=hid-goaldigger-enrolled-faces
                                         -> enrollment uploads the best frame as <user_id>.jpg to S3
  REKOGNITION_ENABLED=1                  -> Amazon Rekognition CompareFaces as a SECOND OPINION, only when our
                                            own face match is uncertain (keeps the fast path fast)
  AWS_REGION=us-east-1
Credentials come from the normal boto3 chain (env vars, ~/.aws, or the EC2 instance role).
"""
from __future__ import annotations

import logging
import os
import threading
import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal

log = logging.getLogger(__name__)


def _flag(name: str) -> bool:
    return os.getenv(name, "0").strip().lower() in ("1", "true", "yes")


class AwsServices:
    def __init__(self, region: str):
        self.region = region
        self.audit_table = os.getenv("AWS_AUDIT_TABLE") or None
        self.faces_bucket = os.getenv("ENROLLED_FACES_BUCKET") or None
        self.rekognition_enabled = _flag("REKOGNITION_ENABLED")
        self.rek_min_similarity = float(os.getenv("REKOGNITION_MIN_SIMILARITY", "0.95"))
        self._clients: dict = {}
        self._warned: set[str] = set()

    @property
    def enabled(self) -> dict:
        return {"audit_table": self.audit_table, "faces_bucket": self.faces_bucket, "rekognition": self.rekognition_enabled}

    def _client(self, name: str, resource: bool = False):
        if name not in self._clients:
            import boto3

            self._clients[name] = (boto3.resource if resource else boto3.client)(name, region_name=self.region)
        return self._clients[name]

    def _warn_once(self, key: str, e: Exception) -> None:
        if key not in self._warned:
            self._warned.add(key)
            log.warning("AWS %s unavailable (%s) — continuing without it", key, e)

    # ------------------------------------------------------------------ DynamoDB audit (siri: log_audit_event)
    def audit(self, kind: str, resp, session_id: str | None = None) -> None:
        if not self.audit_table:
            return

        def _put():
            try:
                sig = resp.liveness.signals or {}
                item = {
                    "session_id": session_id or str(uuid.uuid4()),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "kind": kind,
                    "employee_id": resp.identity.user_id or "unknown",
                    "decision": resp.decision,
                    "confidence": Decimal(str(round(resp.confidence, 4))),
                    "latency_ms": int(resp.timings_ms.get("total", 0)),
                    "identity_score": Decimal(str(round(resp.identity.similarity or 0.0, 4))),
                    "liveness_score": Decimal(str(round(resp.liveness.score or 0.0, 4))),
                    "signals": {k: Decimal(str(round(v, 4))) for k, v in sig.items() if v is not None},
                    "reasons": resp.reasons or [],
                }
                self._client("dynamodb", resource=True).Table(self.audit_table).put_item(Item=item)
            except Exception as e:
                self._warn_once("dynamodb", e)

        threading.Thread(target=_put, daemon=True).start()

    # ------------------------------------------------------------------ S3 enrolled faces (siri: fetch_enrolled_face)
    def store_enrolled_face(self, user_id: str, jpeg: bytes) -> None:
        if not self.faces_bucket or not jpeg:
            return

        def _put():
            try:
                self._client("s3").put_object(Bucket=self.faces_bucket, Key=f"{user_id}.jpg", Body=jpeg, ContentType="image/jpeg")
            except Exception as e:
                self._warn_once("s3", e)

        threading.Thread(target=_put, daemon=True).start()

    # ------------------------------------------------------------------ Rekognition (siri: compare_faces)
    def compare_with_enrolled(self, user_id: str, probe_jpeg: bytes) -> float | None:
        """Similarity in [0,1] between the probe frame and the enrolled S3 photo, or None if unavailable."""
        if not (self.rekognition_enabled and self.faces_bucket and user_id):
            return None
        try:
            t = time.perf_counter()
            enrolled = self._client("s3").get_object(Bucket=self.faces_bucket, Key=f"{user_id}.jpg")["Body"].read()
            r = self._client("rekognition").compare_faces(SourceImage={"Bytes": enrolled}, TargetImage={"Bytes": probe_jpeg},
                                                          SimilarityThreshold=0)
            matches = r.get("FaceMatches", [])
            sim = float(matches[0]["Similarity"]) / 100.0 if matches else 0.0
            log.info("rekognition compare %s: %.3f in %.0f ms", user_id, sim, (time.perf_counter() - t) * 1000)
            return sim
        except Exception as e:
            self._warn_once("rekognition", e)
            return None


class Metrics:
    """In-memory decision + latency counters for GET /metrics (siri's audit-path idea)."""

    def __init__(self, keep: int = 2000):
        self.keep = keep
        self.lock = threading.Lock()
        self.latencies: list[float] = []
        self.counts = {"ALLOW": 0, "REQUEST_VERIFICATION": 0, "DENY": 0}
        self.reasons: dict[str, int] = {}

    def record(self, resp) -> None:
        with self.lock:
            self.counts[resp.decision] = self.counts.get(resp.decision, 0) + 1
            for r in resp.reasons or []:
                self.reasons[r] = self.reasons.get(r, 0) + 1
            total = resp.timings_ms.get("total")
            if total:
                self.latencies = (self.latencies + [float(total)])[-self.keep:]

    def snapshot(self) -> dict:
        import numpy as np

        with self.lock:
            lat = np.array(self.latencies) if self.latencies else None
            return {
                "total_requests": int(sum(self.counts.values())),
                "decisions": dict(self.counts),
                "top_reasons": dict(sorted(self.reasons.items(), key=lambda kv: -kv[1])[:10]),
                "server_latency_ms": None if lat is None else {
                    "p50": round(float(np.percentile(lat, 50)), 1), "p95": round(float(np.percentile(lat, 95)), 1),
                    "max": round(float(lat.max()), 1), "n": int(len(lat))},
            }
