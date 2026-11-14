"""End-to-end liveness-head training, designed to run unattended in tmux on the GB10.

Stages (each one is skipped if its output already exists, so a rerun resumes):
  1. download  N train + M test parquet shards of CelebA-Spoof (HF mirror Ar4ikov/celebA_spoof; mixed live/spoof)
  2. extract   frozen DINOv2 features for CelebA crops + our own kiosk face crops (domain data)
  3. train     standardized logistic regression; kiosk samples up-weighted; honest held-out evaluation
  4. export    portable heads (.npz) + metrics.json + DONE marker

  python training/gb10_train.py --workdir ~/hid/work --kiosk ~/hid/kiosk
Progress: <workdir>/status.json and stdout (tee'd to job.log by gb10/job.sh).
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "api"))
from app.liveness.dino_head import MEAN, STD, preprocess  # noqa: E402  (identical to inference)

DATASET = "Ar4ikov/celebA_spoof"
T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def status(workdir: Path, **kw) -> None:
    p = workdir / "status.json"
    s = json.loads(p.read_text()) if p.exists() else {}
    s.update(kw, elapsed_s=round(time.time() - T0, 1), updated=time.strftime("%H:%M:%S"))
    p.write_text(json.dumps(s, indent=1))


# ----------------------------------------------------------------------------- 1. download
def pick_shards(files: list[str], split: str, n: int) -> list[str]:
    shards = sorted(f for f in files if f.startswith(f"data/{split}-") and f.endswith(".parquet"))
    if n >= len(shards):
        return shards
    idx = np.linspace(0, len(shards) - 1, n).round().astype(int)   # spread across the split
    return [shards[i] for i in sorted(set(idx))]


def download(workdir: Path, n_train: int, n_test: int, budget_min: float) -> dict[str, list[Path]]:
    """Time-boxed: 1 train + 1 test shard are guaranteed, then more shards until the budget runs out.
    (Measured venue Wi-Fi: ~1 MB/s = ~8 min per 0.5 GB shard, so never assume all shards arrive.)"""
    from concurrent.futures import TimeoutError as FutTimeout, as_completed

    from huggingface_hub import HfApi, hf_hub_download

    out: dict[str, list[Path]] = {"train": [], "test": []}
    if (workdir / "feats_train.npz").exists() and (workdir / "feats_test.npz").exists():
        log("features already extracted, skipping download")
        return out
    files = HfApi().list_repo_files(DATASET, repo_type="dataset")
    tr, te = pick_shards(files, "train", n_train), pick_shards(files, "test", n_test)
    order = [("train", tr[0]), ("test", te[0])] + [("train", f) for f in tr[1:]] + [("test", f) for f in te[1:]]
    deadline = time.time() + budget_min * 60
    log(f"downloading up to {len(order)} shards (~0.5 GB each), budget {budget_min:.0f} min")

    def get(job):
        split, f = job
        t = time.time()
        p = Path(hf_hub_download(DATASET, f, repo_type="dataset", local_dir=str(workdir / "hf")))
        mb = p.stat().st_size / 1e6
        log(f"  {split} shard ok: {p.name} ({mb:.0f} MB in {time.time() - t:.0f} s = {mb / max(time.time() - t, 1e-3):.1f} MB/s)")
        return split, p

    ex = ThreadPoolExecutor(max_workers=3)
    futs = {ex.submit(get, j): j for j in order}
    pending = set(futs)
    try:
        for fut in as_completed(futs, timeout=max(1.0, deadline - time.time())):
            pending.discard(fut)
            split, p = fut.result()
            out[split].append(p)
            status(workdir, stage="download", train_shards=len(out["train"]), test_shards=len(out["test"]), planned=len(order))
            if time.time() > deadline:
                break
    except FutTimeout:
        log("download budget reached")
    # guarantee at least one shard per split
    for split in ("train", "test"):
        if not out[split]:
            log(f"waiting for the first {split} shard (required)")
            fut = next(f for f in pending if futs[f][0] == split and futs[f][1] == (tr[0] if split == "train" else te[0]))
            out[split].append(fut.result()[1])
    ex.shutdown(wait=False, cancel_futures=True)
    log(f"using {len(out['train'])} train + {len(out['test'])} test shards")
    return out


# ----------------------------------------------------------------------------- 2. extract
def _prep_celeba(args):
    img_bytes, bbox, expansion = args
    img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
    if img is None or bbox is None or len(bbox) < 4:
        return None
    x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
    if x2 - x1 < 24 or y2 - y1 < 24:
        return None
    try:
        return preprocess(img, [x1, y1, x2, y2], expansion, 224)
    except Exception:
        return None


def _prep_crop(path):
    """Kiosk crops are already square crops at the inference expansion -> just resize + normalise."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        return None
    rgb = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (224, 224), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    return ((rgb - MEAN) / STD).transpose(2, 0, 1)


class Embedder:
    def __init__(self, backbone: str):
        import torch

        self.torch = torch
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        if self.dev == "cuda":
            try:   # fail fast if this torch build has no kernels for this GPU
                (torch.randn(64, 64, device="cuda") @ torch.randn(64, 64, device="cuda")).sum().item()
            except Exception as e:
                log(f"CUDA present but unusable ({e}); falling back to CPU")
                self.dev = "cpu"
        if self.dev == "cpu":
            torch.set_num_threads(max(1, (os.cpu_count() or 4) - 2))
        self.model = torch.hub.load("facebookresearch/dinov2", backbone).to(self.dev).eval()
        log(f"DINOv2 {backbone} on {self.dev}" + (f" ({torch.cuda.get_device_name(0)})" if self.dev == "cuda" else f" ({os.cpu_count()} cpus)"))

    def __call__(self, batch: list[np.ndarray]) -> np.ndarray:
        x = self.torch.from_numpy(np.stack(batch)).to(self.dev)
        with self.torch.inference_mode(), self.torch.autocast(self.dev, dtype=self.torch.bfloat16, enabled=self.dev == "cuda"):
            return self.model(x).float().cpu().numpy()


def extract_celeba(workdir: Path, shards: list[Path], split: str, emb: Embedder, pool: Pool, expansion: float, limit: int) -> Path:
    out = workdir / f"feats_{split}.npz"
    if out.exists():
        log(f"{out.name} exists, skipping")
        return out
    import pyarrow.parquet as pq

    feats, labels, n_seen = [], [], 0
    for si, shard in enumerate(shards, 1):
        pf = pq.ParquetFile(shard)
        for rb in pf.iter_batches(batch_size=256, columns=["Filepath", "Bbox", "Class"]):
            col = rb.to_pydict()
            jobs = [((im or {}).get("bytes"), bb, expansion) for im, bb in zip(col["Filepath"], col["Bbox"])]
            prepped = pool.map(_prep_celeba, jobs, chunksize=16)
            keep = [(p, c) for p, c in zip(prepped, col["Class"]) if p is not None]
            if keep:
                feats.append(emb([p for p, _ in keep]))
                labels += [1 if c == "live" else 0 for _, c in keep]
            n_seen += len(jobs)
            if limit and n_seen >= limit:
                break
        log(f"  {split}: shard {si}/{len(shards)} done, {len(labels)} samples")
        status(workdir, stage=f"extract_{split}", done=si, total=len(shards), samples=len(labels))
        if limit and n_seen >= limit:
            break
    np.savez_compressed(out, feats=np.concatenate(feats).astype(np.float32), live=np.array(labels, np.int8))
    log(f"saved {out.name}: {len(labels)} samples ({int(np.sum(labels))} live)")
    return out


def extract_kiosk(workdir: Path, kiosk: Path, emb: Embedder, pool: Pool) -> Path | None:
    out = workdir / "feats_kiosk.npz"
    lab = kiosk / "labels.csv"
    if not lab.exists():
        log("no kiosk data bundled, skipping domain data")
        return None
    rows = list(csv.DictReader(open(lab)))
    prepped = pool.map(_prep_crop, [kiosk / r["file"] for r in rows], chunksize=8)
    keep = [(p, r) for p, r in zip(prepped, rows) if p is not None]
    feats = np.concatenate([emb([p for p, _ in keep[i:i + 128]]) for i in range(0, len(keep), 128)])
    np.savez_compressed(out, feats=feats.astype(np.float32), live=np.array([int(r["live"]) for _, r in keep], np.int8),
                        burst=np.array([r["burst"] for _, r in keep]))
    log(f"saved {out.name}: {len(keep)} kiosk crops ({sum(int(r['live']) for _, r in keep)} live)")
    return out


# ----------------------------------------------------------------------------- 3. train / evaluate
def fit(X, y, w=None, C=0.5):
    from sklearn.linear_model import LogisticRegression

    mean, scale = X.mean(0), X.std(0) + 1e-6
    clf = LogisticRegression(C=C, max_iter=4000, class_weight="balanced")
    clf.fit((X - mean) / scale, y, sample_weight=w)
    return {"mean": mean.astype(np.float32), "scale": scale.astype(np.float32),
            "coef": clf.coef_.reshape(-1).astype(np.float32), "intercept": clf.intercept_.astype(np.float32)}


def predict(head, X):
    z = ((X - head["mean"]) / head["scale"]) @ head["coef"] + head["intercept"][0]
    return 1 / (1 + np.exp(-z))


def report(p, y, thr=0.5):
    from sklearn.metrics import roc_auc_score

    y = y.astype(bool)
    r = {"n": int(len(y)), "n_live": int(y.sum()),
         "accuracy": float(((p >= thr) == y).mean()),
         "APCER": float((p[~y] >= thr).mean()) if (~y).any() else None,   # attacks accepted
         "BPCER": float((p[y] < thr).mean()) if y.any() else None}       # live rejected
    if 0 < y.sum() < len(y):
        r["AUC"] = float(roc_auc_score(y, p))
    return r


def burst_report(p, y, bursts):
    """Kiosk decisions are per 5-frame burst: score = median over the burst's frames."""
    ub = sorted(set(bursts))
    bp = np.array([np.median(p[bursts == b]) for b in ub])
    by = np.array([y[bursts == b][0] for b in ub])
    return report(bp, by) | {"live_min": float(bp[by == 1].min()) if (by == 1).any() else None,
                             "spoof_max": float(bp[by == 0].max()) if (by == 0).any() else None}


def save_head(path: Path, head: dict, backbone: str, expansion: float):
    np.savez(path, **head, backbone=np.array(backbone), crop_expansion=np.array(expansion), img_size=np.array(224))


def train(workdir: Path, outdir: Path, backbone: str, expansion: float, kiosk_weight: float) -> dict:
    tr = np.load(workdir / "feats_train.npz")
    te = np.load(workdir / "feats_test.npz")
    Xtr, ytr = tr["feats"], tr["live"]
    metrics: dict = {"backbone": backbone, "train_samples": int(len(ytr)), "train_live": int(ytr.sum())}

    head_c = fit(Xtr, ytr)
    metrics["celeba_only"] = {"celeba_test": report(predict(head_c, te["feats"]), te["live"])}
    log(f"CelebA-only head  -> CelebA test {metrics['celeba_only']['celeba_test']}")

    kp = workdir / "feats_kiosk.npz"
    if kp.exists():
        k = np.load(kp)
        Xk, yk, bk = k["feats"], k["live"], k["burst"]
        rng = np.random.default_rng(0)
        spoof_b = sorted(set(bk[yk == 0])); live_b = sorted(set(bk[yk == 1]))
        # held-out bursts (never trained on): 1 attack burst + ~30% of live bursts
        held = set(rng.choice(spoof_b, size=min(1, len(spoof_b)), replace=False)) | \
               set(rng.choice(live_b, size=max(1, int(0.3 * len(live_b))), replace=False))
        tmask = np.array([b not in held for b in bk])
        metrics["celeba_only"]["kiosk_heldout_bursts"] = burst_report(predict(head_c, Xk[~tmask]), yk[~tmask], bk[~tmask])

        # kiosk samples carry ~kiosk_weight of the total weight, split 50/50 between live and spoof
        # (only a few attack bursts vs many live ones; unbalanced, the domain data would just teach "kiosk = live")
        def weights(yk_used):
            total = kiosk_weight * len(ytr)
            n_live, n_spoof = max(int((yk_used == 1).sum()), 1), max(int((yk_used == 0).sum()), 1)
            wk = np.where(yk_used == 1, 0.5 * total / n_live, 0.5 * total / n_spoof)
            return np.concatenate([np.ones(len(ytr)), wk])

        head_d = fit(np.concatenate([Xtr, Xk[tmask]]), np.concatenate([ytr, yk[tmask]]), weights(yk[tmask]))
        metrics["celeba_plus_kiosk"] = {"celeba_test": report(predict(head_d, te["feats"]), te["live"]),
                                        "kiosk_heldout_bursts": burst_report(predict(head_d, Xk[~tmask]), yk[~tmask], bk[~tmask])}
        metrics["kiosk_heldout"] = sorted(held)
        log(f"CelebA-only  on kiosk held-out bursts: {metrics['celeba_only']['kiosk_heldout_bursts']}")
        log(f"CelebA+kiosk on kiosk held-out bursts: {metrics['celeba_plus_kiosk']['kiosk_heldout_bursts']}")
        log(f"CelebA+kiosk on CelebA test:           {metrics['celeba_plus_kiosk']['celeba_test']}")
        # final deployable head: refit on ALL kiosk data
        head_final = fit(np.concatenate([Xtr, Xk]), np.concatenate([ytr, yk]), weights(yk))
    else:
        head_final = head_c

    outdir.mkdir(parents=True, exist_ok=True)
    save_head(outdir / "dino_head.npz", head_final, backbone, expansion)
    save_head(outdir / "dino_head_celeba_only.npz", head_c, backbone, expansion)
    (outdir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", type=Path, required=True)
    ap.add_argument("--kiosk", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--n-train", type=int, default=int(os.getenv("N_TRAIN", 12)))
    ap.add_argument("--n-test", type=int, default=int(os.getenv("N_TEST", 3)))
    ap.add_argument("--dl-budget-min", type=float, default=float(os.getenv("DL_BUDGET_MIN", 20)),
                    help="stop fetching more shards after this many minutes (1 train + 1 test always fetched)")
    ap.add_argument("--limit", type=int, default=int(os.getenv("LIMIT", 0)), help="max images per split (0 = all; for dry runs)")
    ap.add_argument("--backbone", default="dinov2_vits14")
    ap.add_argument("--expansion", type=float, default=1.5)
    ap.add_argument("--kiosk-weight", type=float, default=0.2)
    args = ap.parse_args()
    wd = args.workdir.expanduser()
    wd.mkdir(parents=True, exist_ok=True)
    out = (args.out or wd / "out").expanduser()
    (out / "DONE").unlink(missing_ok=True)
    try:
        status(wd, stage="start")
        shards = download(wd, args.n_train, args.n_test, args.dl_budget_min)
        emb = Embedder(args.backbone)
        with Pool(max(2, (os.cpu_count() or 4) - 2)) as pool:
            extract_celeba(wd, shards["train"], "train", emb, pool, args.expansion, args.limit)
            extract_celeba(wd, shards["test"], "test", emb, pool, args.expansion, args.limit // 4 if args.limit else 0)
            if args.kiosk:
                extract_kiosk(wd, args.kiosk.expanduser(), emb, pool)
        status(wd, stage="train")
        train(wd, out, args.backbone, args.expansion, args.kiosk_weight)
        (out / "DONE").write_text(time.strftime("%Y-%m-%d %H:%M:%S"))
        status(wd, stage="done")
        log(f"ALL DONE -> {out}")
    except Exception as e:
        import traceback

        traceback.print_exc()
        status(wd, stage="failed", error=repr(e))
        (out / "FAILED").parent.mkdir(parents=True, exist_ok=True)
        (out / "FAILED").write_text(repr(e))
        raise


if __name__ == "__main__":
    main()
