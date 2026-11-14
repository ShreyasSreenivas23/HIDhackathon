"""End-to-end SYSTEM evaluation: replay recorded bursts through the running API (/verify).

  python eval/eval_live.py --api http://localhost:8000
Expectations:
  live      -> ALLOW with identity == person          (DENY = false rejection, REQUEST_VERIFICATION reported separately)
  impostor  -> not ALLOW                              (ALLOW = false acceptance)
  attacks   -> not ALLOW                              (ALLOW = false acceptance / APCER)
Writes eval/results/live_report.json + live_results.csv. Make sure every `person` is enrolled first.
"""
from __future__ import annotations

import argparse
import base64
import csv
import json
import time
from collections import defaultdict
from pathlib import Path

import httpx
import numpy as np

CAP = Path(__file__).resolve().parent / "captures"
OUT = Path(__file__).resolve().parent / "results"


def b64(p: Path) -> str:
    return base64.b64encode(p.read_bytes()).decode()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://localhost:8000")
    args = ap.parse_args()

    rows = []
    with httpx.Client(timeout=30) as cli:
        for meta_f in sorted(CAP.glob("*/*/meta.json")):
            meta = json.loads(meta_f.read_text())
            frames = [b64(p) for p in sorted(meta_f.parent.glob("frame_*.jpg"))]
            t0 = time.perf_counter()
            r = cli.post(f"{args.api}/verify", json={"frames": frames}).json()
            rt = (time.perf_counter() - t0) * 1000
            rows.append({"label": meta["label"], "person": meta["person"], "device": meta.get("device", ""),
                         "decision": r["decision"], "user": r["identity"].get("user_id"), "sim": r["identity"].get("similarity"),
                         "liveness": r["liveness"].get("score"), "reasons": "|".join(r["reasons"]),
                         "server_ms": r["timings_ms"].get("total"), "roundtrip_ms": round(rt, 1),
                         **{f"sig_{k}": v for k, v in r["liveness"].get("signals", {}).items()}})

    if not rows:
        print("no captures found in", CAP)
        return
    by = defaultdict(list)
    for r in rows:
        by[r["label"]].append(r)

    rep = {"n": len(rows), "per_label": {}}
    live = by.get("live", [])
    if live:
        ok = [r["decision"] == "ALLOW" and r["user"] == r["person"] for r in live]
        rep["genuine"] = {
            "n": len(live),
            "accepted": float(np.mean(ok)),
            "FRR_hard_deny": float(np.mean([r["decision"] == "DENY" for r in live])),
            "request_verification_rate": float(np.mean([r["decision"] == "REQUEST_VERIFICATION" for r in live])),
            "wrong_identity_allow": float(np.mean([r["decision"] == "ALLOW" and r["user"] != r["person"] for r in live])),
        }
    attacks = [r for r in rows if r["label"] != "live"]
    if attacks:
        rep["FAR_system"] = float(np.mean([r["decision"] == "ALLOW" for r in attacks]))
    for label, rs in by.items():
        if label == "live":
            continue
        rep["per_label"][label] = {
            "n": len(rs),
            "APCER_or_FAR": float(np.mean([r["decision"] == "ALLOW" for r in rs])),
            "deny_rate": float(np.mean([r["decision"] == "DENY" for r in rs])),
            "verify_rate": float(np.mean([r["decision"] == "REQUEST_VERIFICATION" for r in rs])),
        }
    lat = np.array([r["roundtrip_ms"] for r in rows])
    rep["latency_ms"] = {"server_p50": float(np.median([r["server_ms"] for r in rows])),
                         "roundtrip_p50": float(np.median(lat)), "roundtrip_p95": float(np.percentile(lat, 95)),
                         "note": "kiosk decision time = ~400 ms burst capture + roundtrip"}
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "live_results.csv", "w", newline="") as f:
        keys = sorted({k for r in rows for k in r})
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    (OUT / "live_report.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
