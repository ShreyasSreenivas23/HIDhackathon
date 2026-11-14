"""Train the liveness head on DINOv2 features. Seconds on any CPU.

  python training/train_head.py --train data/feats_train.npz --test data/feats_test.npz --out api/models/dino_head.joblib
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "eval"))
from metrics import ATTACK_GROUP, per_attack_report, roc_auc  # noqa: E402


def groups_of(types):
    return np.array(["live" if t == 0 else ATTACK_GROUP.get(int(t), "other") for t in types])


def report(name, clf, X, y, types):
    p = clf.predict_proba(X)[:, list(clf.classes_).index(1)]
    r = per_attack_report(p, y.astype(bool), groups_of(types), 0.5)
    r["AUC"] = roc_auc(p, y)
    print(f"\n== {name}  AUC={r['AUC']:.4f}  acc={r['accuracy']:.4f}  BPCER={r['BPCER']:.4f}  ACER={r['ACER']:.4f}")
    for g, v in r["attacks"].items():
        print(f"   {g:15s} APCER={v['APCER']:.4f}  (n={v['n']})")
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=Path, required=True)
    ap.add_argument("--test", type=Path)
    ap.add_argument("--C", type=float, default=0.5)
    ap.add_argument("--out", type=Path, default=REPO / "api" / "models" / "dino_head.joblib")
    args = ap.parse_args()

    d = np.load(args.train, allow_pickle=True)
    X, y, types, subj = d["feats"], d["live"], d["spoof_type"], d["subject"]
    # subject-disjoint validation split (no identity leakage)
    tr, va = next(GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=0).split(X, y, subj))
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=args.C, max_iter=3000, class_weight="balanced"))
    clf.fit(X[tr], y[tr])
    results = {"val": report("val (subject-disjoint)", clf, X[va], y[va], types[va])}

    if args.test and args.test.exists():
        t = np.load(args.test, allow_pickle=True)
        results["test"] = report("CelebA-Spoof intra-test", clf, t["feats"], t["live"], t["spoof_type"])

    clf.fit(X, y)  # refit on everything for deployment
    args.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"clf": clf, "live_label": 1, "backbone": str(d["backbone"]), "crop_expansion": float(d["expansion"]),
                 "img_size": 224}, args.out)
    (args.out.with_suffix(".metrics.json")).write_text(json.dumps(results, indent=2))
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
