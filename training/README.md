# Training (P2) — DINOv2 liveness head on the GB10

1. **Data:** download a few CelebA-Spoof archive parts from the official repo (Google Drive) and unzip them so `Data/` and `metas/` sit under one root. The loader skips files you don't have.
2. **Container (GB10 is ARM64):** `docker run --gpus all -it -v $PWD:/ws -v /data:/data nvcr.io/nvidia/pytorch:<tag>-py3`, then `pip install opencv-python-headless scikit-learn joblib`.
3. **Sanity check the boxes first:** load 5 images with `celeba_spoof.load_image`, draw the bbox and look at it. If the boxes are off, fix `read_bbox`.
4. **Extract features:**
   ```bash
   python training/extract_dino.py --root /data/CelebA_Spoof --split train --per-type 3000 --out data/feats_train.npz
   python training/extract_dino.py --root /data/CelebA_Spoof --split test  --per-type 800  --out data/feats_test.npz
   ```
5. **Train the head** (CPU is fine; copy the `.npz` files to the laptop if the VPN is flaky):
   ```bash
   python training/train_head.py --train data/feats_train.npz --test data/feats_test.npz
   ```
   This writes `api/models/dino_head.joblib` plus a `.metrics.json` file (per-attack APCER/BPCER, used for the slides).
6. Restart the API and check that `/health` shows `"dino": true`.

The crop and normalisation code comes from `api/app/liveness/dino_head.py`, so training and inference preprocessing can't drift apart.
