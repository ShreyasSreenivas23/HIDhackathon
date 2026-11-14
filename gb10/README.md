# Training the DINOv2 liveness head on the GB10 (VPN windows of ~3 min)

Every VPN window needs **one SSH connection only**. Training runs detached in **tmux on the GB10**, so the VPN can drop at any time.

## Before the window (local, no VPN)
```bash
api/.venv/Scripts/python gb10/prepare_bundle.py    # ~20 s: code + kiosk face crops -> data/gb10_bundle (~4 MB)
```
Copy `gb10/config.env.example` to `gb10/config.env` and put the login in it (`GB10_HOST=user@host`). Key-based SSH means no password prompt; a password works too, but costs window time.

## VPN window 1: launch (~5-15 s)
```bash
bash gb10/launch.sh
```
This uploads the bundle, unpacks it to `~/hid`, and starts `tmux` session `hidtrain` running `gb10/job.sh`. It prints `STARTED ...` and exits.

## Any later window: check (~3 s)
```bash
bash gb10/status.sh
```

## Window when it's done: fetch + install (~5 s)
```bash
bash gb10/fetch.sh     # pulls out/ (heads + metrics), copies dino_head.npz into api/models/
```
Then restart the API. `/health` shows `"dino": true`.

## What the job does (on the GB10, unattended)
1. **Env:** creates a venv that reuses system torch, and pip-installs the small dependencies. If torch can't use CUDA, it tries the aarch64 CUDA wheels (cu130 → cu128). If those fail too, it runs on CPU (still fine for DINOv2-S).
2. **Download:** CelebA-Spoof parquet shards from the HuggingFace mirror `Ar4ikov/celebA_spoof`. Shards are mixed live/spoof, about 0.5 GB / 4.5k images each. **Time-boxed** (`DL_BUDGET_MIN`, default 20): 1 train + 1 test shard are guaranteed, then up to 12 + 3.
3. **Extract:** frozen DINOv2-S features, using the same crop and normalisation as inference. Our kiosk crops are included as domain data.
4. **Train:** logistic regression, with kiosk samples up-weighted to 20 % of the total weight. **Honest evaluation:** CelebA test shards, plus *held-out kiosk bursts* (1 attack burst + ~30 % of live bursts never seen in training). It reports CelebA-only vs CelebA+kiosk.
5. **Export:** `out/dino_head.npz` (final, trained on all data), `out/dino_head_celeba_only.npz`, `out/metrics.json`, `out/DONE`.

Resumable: re-running `launch.sh` skips stages whose outputs exist (`RESTART=1` restarts a running job).

## How long?
| Stage | GPU | CPU fallback |
|---|---|---|
| env setup | 1-3 min (more if a torch wheel is needed: ~3-6 min) | same |
| download | **network-bound**: 0.5 GB/shard. At 50 MB/s: ~2 min for 15 shards. At venue-Wi-Fi speed (1 MB/s, measured): capped by the 20-min budget, so ~2-3 shards | same |
| features (~55k imgs at full size) | ~2-5 min | ~10-20 min |
| train + eval | < 1 min | < 1 min |
| **total** | **~10-30 min** | **~20-45 min** |
