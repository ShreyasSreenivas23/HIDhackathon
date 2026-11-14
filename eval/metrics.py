"""Metrics shared by training/ and eval/ (ISO/IEC 30107-3 terms + HID slide terms).

Conventions: score = P(live) or similarity (higher = more genuine). A sample is ACCEPTED when score >= thr.
"""
from __future__ import annotations

import numpy as np

# CelebA-Spoof spoof_type label (index 40) -> our 4 challenge categories
CELEBA_SPOOF_TYPES = {
    0: "live", 1: "Photo", 2: "Poster", 3: "A4", 4: "Face Mask", 5: "Upper Body Mask",
    6: "Region Mask", 7: "PC", 8: "Pad", 9: "Phone", 10: "3D Mask",
}
ATTACK_GROUP = {
    1: "printed_photo", 2: "printed_photo", 3: "printed_photo",
    4: "mask", 5: "mask", 6: "mask", 10: "mask",
    7: "screen_replay", 8: "screen_replay", 9: "screen_replay",
}


def apcer_bpcer(scores, is_live, thr: float) -> tuple[float, float]:
    s, y = np.asarray(scores, float), np.asarray(is_live, bool)
    apcer = float((s[~y] >= thr).mean()) if (~y).any() else float("nan")   # attacks accepted
    bpcer = float((s[y] < thr).mean()) if y.any() else float("nan")       # bona fide rejected
    return apcer, bpcer


def per_attack_report(scores, is_live, groups, thr: float) -> dict:
    """groups: per-sample attack group name ('live' for bona fide). APCER per group, shared BPCER."""
    s, y, g = np.asarray(scores, float), np.asarray(is_live, bool), np.asarray(groups)
    _, bpcer = apcer_bpcer(s, y, thr)
    out = {"threshold": thr, "BPCER": bpcer, "n_live": int(y.sum()), "attacks": {}}
    worst = 0.0
    for grp in sorted(set(g[~y])):
        m = (g == grp) & ~y
        apcer = float((s[m] >= thr).mean())
        worst = max(worst, apcer)
        out["attacks"][grp] = {"APCER": apcer, "n": int(m.sum())}
    out["APCER_max"] = worst
    out["ACER"] = (worst + bpcer) / 2
    out["accuracy"] = float(((s >= thr) == y).mean())
    return out


def threshold_at_rate(neg_scores, target: float) -> float:
    """Smallest threshold such that the fraction of negatives with score >= thr is <= target."""
    neg = np.sort(np.asarray(neg_scores, float))
    if len(neg) == 0:
        return 0.5
    k = int(np.floor(len(neg) * (1 - target)))
    k = min(max(k, 0), len(neg) - 1)
    return float(np.nextafter(neg[k], np.inf))


def threshold_at_miss(pos_scores, target: float) -> float:
    """Largest threshold such that the fraction of positives with score < thr is <= target."""
    pos = np.sort(np.asarray(pos_scores, float))
    if len(pos) == 0:
        return 0.5
    k = int(np.floor(len(pos) * target))
    return float(pos[min(k, len(pos) - 1)])


def far_frr(genuine, impostor, thr: float) -> tuple[float, float]:
    g, i = np.asarray(genuine, float), np.asarray(impostor, float)
    return float((i >= thr).mean()), float((g < thr).mean())


def roc_auc(scores, labels) -> float:
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(np.asarray(labels, int), np.asarray(scores, float)))
