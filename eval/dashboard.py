"""Metrics dashboard in HID's slide format.   streamlit run eval/dashboard.py"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

RES = Path(__file__).resolve().parent / "results"
st.set_page_config(page_title="Spot-a-Spoof metrics", layout="wide")
st.title("Performance Evaluation")


def load(name):
    p = RES / name
    return json.loads(p.read_text()) if p.exists() else None


celeba, lfw, live = load("celeba_report.json"), load("lfw_report.json"), load("live_report.json")

# ---- headline numbers (same 5 metrics as HID's reference slide) ----
best = None
if celeba:
    best_key = max(celeba["ablation"], key=lambda k: celeba["ablation"][k]["AUC"])
    best = celeba["ablation"][best_key]
c = st.columns(5)
c[0].metric("Liveness accuracy", f"{best['accuracy'] * 100:.1f}%" if best else "—", help="CelebA-Spoof test subset, best signal combo")
c[1].metric("Face-auth accuracy", f"{lfw['accuracy'] * 100:.1f}%" if lfw else "—", help="LFW 6,000 pairs")
c[2].metric("FAR (system)", f"{live['FAR_system'] * 100:.1f}%" if live and "FAR_system" in live else "—", help="live test set: attacks + impostors reaching ALLOW")
c[3].metric("FRR (hard deny)", f"{live['genuine']['FRR_hard_deny'] * 100:.1f}%" if live and "genuine" in live else "—", help="live test set")
c[4].metric("Decision time p95", f"{(live['latency_ms']['roundtrip_p95'] + 400) / 1000:.2f} s" if live else "—", help="400 ms burst + API roundtrip")
st.caption("Reference (HID): liveness 99.4% · face-auth 99.2% · FAR 0.2% · FRR 0.6% · < 2 s")

if live:
    st.subheader("Live test set — per attack type (our phones/laptops)")
    st.dataframe(pd.DataFrame(live["per_label"]).T, use_container_width=True)
    if "genuine" in live:
        st.json(live["genuine"])

if celeba:
    st.subheader("CelebA-Spoof — ablation (which signal catches which attack)")
    rows = []
    for combo, r in celeba["ablation"].items():
        row = {"signals": combo, "AUC": r["AUC"], "accuracy": r["accuracy"], "BPCER": r["BPCER"], "ACER": r["ACER"]}
        row.update({f"APCER {g}": v["APCER"] for g, v in r["attacks"].items()})
        rows.append(row)
    st.dataframe(pd.DataFrame(rows).set_index("signals").sort_values("AUC"), use_container_width=True)

if lfw:
    st.subheader("Identity (LFW)")
    st.json(lfw)

if (RES / "live_results.csv").exists():
    df = pd.read_csv(RES / "live_results.csv")
    st.subheader("Latency distribution (ms)")
    st.bar_chart(df["roundtrip_ms"].value_counts(bins=20).sort_index())
    st.subheader("All live attempts")
    st.dataframe(df, use_container_width=True)
