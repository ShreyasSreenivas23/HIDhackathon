# 8-Hour Build Plan — HID "Can You Spot a Spoof?"

**Team:** 4 people, AI-assisted ("vibecoded"), 8 hours.
**Compute:** laptop (Core Ultra 9, 64 GB) = demo kiosk + local inference · NVIDIA GB10 (via VPN) = training only · AWS ($200) = cloud deployment + storage.
**Props:** phones/tablets/laptops (screen + video replay, live). Printed photo and mask covered **via datasets** (CelebA-Spoof).
Detector details for each attack live in [spoof_approaches.md](spoof_approaches.md); this file is what we actually build.

---

## 1. What the judges score → what we build for it

| Criterion | Weight | What earns it |
|---|---|---|
| Problem-solution fit | 25% | Checks liveness **and** identity, the 3 outcomes (ALLOW / REQUEST VERIFICATION / DENY), ≤ 2 s, all 4 attack types evaluated |
| Technical implementation | 25% | Clean architecture: kiosk UI → FastAPI → modular detectors; same Docker image runs on the edge (laptop) and on AWS; DynamoDB/S3/CloudWatch; a clear scaling story |
| Innovation & creativity | 20% | (1) Foundation-model liveness head (DINOv2) + classic CV signals fused together, (2) **explainable** decisions, (3) **adaptive** challenge that only appears in the uncertain zone |
| Functionality & usability | 15% | Works live with a phone spoof in front of judges; big clear kiosk UI; guidance prompts ("move closer"); decision time shown |
| Presentation | 15% | Metrics slide in the **same format as HID's reference slide**, per-attack table, non-technical story, backup demo video |

**HID's reference metrics (the bar to report against):** liveness accuracy 99.4% · face-auth accuracy 99.2% · FAR 0.2% · FRR 0.6% · decision time < 2 s · validated on photo, screen, video, mask.
→ We report **exactly these 5 numbers**, plus a breakdown by attack type. Be honest about which test set each number comes from.

---

## 2. Scope (cut for 8 hours)

**MVP (must ship by hour 3):**
- Kiosk web UI: camera, quality gate, 5-frame burst, result screen (decision, confidence, time)
- Liveness: pretrained **MiniFASNet ONNX** (trained on CelebA-Spoof)
- Identity: **InsightFace buffalo_l** (SCRFD detection + ArcFace embedding), 1:N search over enrolled users
- Decision engine with thresholds + gray zone
- Enrollment page (3 frames → averaged embedding)

**Core upgrades (hours 3–5.5):**
- **DINOv2-S frozen features + logistic-regression head** trained on a CelebA-Spoof subset (GB10), fused with MiniFASNet
- **Device detector** (YOLO11n/YOLOv8n COCO classes: cell phone, laptop, tv): is the face inside a screen? This is the live-demo hero for screen/video replay
- **Planarity check** (homography residual of MediaPipe landmarks across the burst): catches flat photos and screens
- **Active challenge** for REQUEST VERIFICATION: 2 random actions (blink / turn left / turn right / smile) checked with MediaPipe blendshapes and head pose, then OTP fallback (simulated)
- AWS deployment + evaluation harness + metrics dashboard

**Stretch (only if ahead):** screen-flash colour challenge · moiré FFT signal · Rekognition Face Liveness/CompareFaces baseline comparison · DWT+LBP mask signal

**Dropped:** rPPG, monocular depth, physical mask capture, training a backbone from scratch.

---

## 3. Architecture

```
Kiosk (laptop browser: Vite + React + MediaPipe tasks-vision)
  ├─ quality gate (face size, blur, brightness, yaw)        ← client-side, instant
  ├─ 5-frame burst (~500 ms) → POST /verify
  └─ challenge UI (random actions) → POST /challenge
                │
FastAPI "decision service" (one Docker image)
  ├─ face detect + align (SCRFD)
  ├─ liveness (run in parallel): MiniFASNet · DINOv2 head · device · planarity → fused L + per-signal scores
  ├─ identity: ArcFace embedding → cosine vs gallery → M
  ├─ decision engine → ALLOW / REQUEST_VERIFICATION / DENY + reasons
  └─ timings_ms for each stage
                │
Storage: local SQLite (edge mode) | DynamoDB (cloud mode) for enrollments
         S3 for audit frames · CloudWatch for latency metrics
```

**Deployment story (Technical implementation):**
- **Edge mode (live demo):** container runs on the laptop. Lowest latency, works offline, keeps biometrics on-device. *This is what we demo* — the venue network is a risk.
- **Cloud mode:** the same image on AWS EC2 (`c7i.xlarge` CPU is enough, about $0.18/h; or `g4dn.xlarge` about $0.53/h if needed) behind an ALB, with DynamoDB/S3. Scaling slide: the service is stateless, so it can run on ECS Fargate with autoscaling.
- **GB10:** training and feature extraction only (ARM64, so use the NVIDIA NGC PyTorch container). **Never in the demo path** (VPN).
- **Budget:** under $40 total, well inside $200.

---

## 4. API contract (agree on this in the first 30 minutes; everyone codes against it)

```jsonc
// POST /enroll
{ "user_id": "alice", "frames": ["<base64 jpeg>", "..."] }      // 3 frames
→ { "ok": true, "quality": 0.93 }

// POST /verify
{ "frames": ["<base64 jpeg>", ...5], "client_quality": {"blur":120,"brightness":0.55,"face_px":210,"yaw":4} }
→ {
  "decision": "ALLOW" | "REQUEST_VERIFICATION" | "DENY",
  "confidence": 0.96,
  "reasons": ["planar_motion", "phone_detected"],          // empty on ALLOW
  "liveness": { "score": 0.97, "signals": { "minifas": 0.95, "dino": 0.98, "device": 0.0, "planarity": 0.91 } },
  "identity": { "user_id": "alice", "similarity": 0.71 },
  "challenge": { "nonce": "x9f2", "actions": ["blink", "turn_left"] } | null,
  "timings_ms": { "detect": 25, "liveness": 140, "identity": 90, "total": 310 }
}

// POST /challenge
{ "nonce": "x9f2", "frames": ["<base64 jpeg>", ...], "client_events": [{"action":"blink","t":412}, ...] }
→ same shape as /verify (no further challenge; on failure → OTP screen)
```

Score convention: **every signal is 1.0 = live / genuine**, 0.0 = spoof.

**Decision rule:**
```
if quality fails                              → REQUEST_VERIFICATION (with guidance)
if any hard signal fires (device ≥ 0.8) or L < L_lo or M < M_lo → DENY
if L ≥ L_hi and M ≥ M_hi                      → ALLOW
else                                          → REQUEST_VERIFICATION → challenge → OTP
```
Threshold calibration (hour 5.5–6.5): choose `L_hi` so spoof acceptance (APCER) ≤ 0.2% on validation data; choose `M_hi` at FAR ≤ 0.1% on LFW pairs. The gray band between the `lo` and `hi` thresholds is what keeps **hard** rejections (FRR) low: uncertain live users are asked to verify rather than denied.

---

## 5. Repo layout (one owner per folder → no AI-generated merge conflicts)

```
kiosk-web/      P1   Vite+React, MediaPipe, UI screens, challenge UI
api/            P3   FastAPI app, decision.py, identity/, store/, Dockerfile
api/liveness/   P2   minifas.py, dino_head.py, device.py, planarity.py, fuse.py
training/       P2   extract_dino.py, train_head.py, export_onnx.py (runs on GB10)
eval/           P4   run_eval.py, metrics.py, dashboard (Streamlit), test-set recorder
infra/          P3   AWS setup notes, docker-compose
data/           —    gitignored: datasets, captures, models
CONTRACT.md     all  the API contract above (frozen after hour 0:30)
```
Use a GitHub repo, and push small commits often. Each person's AI prompts should include `CONTRACT.md` plus their own folder only.

---

## 6. Roles

| Person | Owns | Deliverables |
|---|---|---|
| **P1 – Kiosk UX** | `kiosk-web/` | Camera + face box overlay, quality gate prompts, burst capture, result screen (green/amber/red, decision time, confidence, reasons), enrollment screen, challenge flow, OTP screen. Later: polish, accessibility, demo mode. |
| **P2 – Liveness ML** | `api/liveness/`, `training/` | MiniFASNet ONNX wrapper (correct crop scale!) → CelebA-Spoof subset download → DINOv2 features on GB10 → logistic-regression head → device detector → planarity → `fuse.py`. |
| **P3 – Backend, identity & cloud** | `api/`, `infra/` | FastAPI skeleton with **stub responses in the first 30 minutes** (so P1 is unblocked), InsightFace identity, enrollment store, decision engine, challenge verification (nonce, action order), Docker, AWS deploy (EC2 + DynamoDB + S3 + CloudWatch). |
| **P4 – Evaluation & story** | `eval/`, slides | LFW pairs for identity FAR/FRR, CelebA-Spoof test split grouped into 4 attack categories, live test-set recording tool, metrics + latency report, Streamlit dashboard, slide deck, demo script, backup demo video. |

---

## 7. Timeline

| Time | Everyone / milestones | P1 | P2 | P3 | P4 |
|---|---|---|---|---|---|
| 0:00–0:30 | Repo, CONTRACT.md, **start dataset downloads now** | Vite scaffold | Start CelebA-Spoof + model downloads | FastAPI stub endpoints | Download LFW, metric definitions |
| 0:30–3:00 | Build the MVP | Camera, burst, result UI | MiniFASNet wrapper; DINOv2 extraction on GB10 | InsightFace identity, enrollment, decision engine | Eval harness on CelebA-Spoof + LFW; recorder tool |
| **3:00** | **Integration #1: end-to-end live demo works** (MiniFASNet + ArcFace) | | | | |
| 3:00–5:00 | Upgrades | Challenge UI, OTP, polish | DINOv2 head, device detector, planarity, fusion | Challenge verification, Docker, AWS deploy | Dashboard; slide skeleton |
| **5:00** | **Integration #2** · **feature freeze at 5:30** | | | | |
| 5:30–6:30 | Record live test set with phones (all 4 people, ~20 min) → run evaluation → calibrate thresholds | | | | |
| 6:30–7:30 | Fix bugs only · slides · record backup demo video | | | | |
| 7:30–8:00 | Two full rehearsals, including judge Q&A | | | | |

---

## 8. Data & evaluation plan

**Training (liveness head):** CelebA-Spoof (625k images, 10k subjects; non-commercial research licence, so fine for a hackathon). The full set is very large and split across many archive parts on Google Drive, so **download only a few parts** (target ~30–60k images, balanced live/spoof). Check Kaggle for pre-cropped mirrors first, which are faster. DINOv2-S feature extraction for 50k images on GB10 takes minutes; the logistic regression trains in seconds.

**Attack-type mapping (CelebA-Spoof labels → our 4 categories):**
- Printed photo ← Photo, Poster, A4
- Screen replay ← PC, Pad, Phone
- Mask ← Face Mask (paper), Upper-Body Mask, Region Mask, 3D Mask
- Video replay ← **not in CelebA-Spoof**, so use our own phone/laptop video replays (live test set)

**Live test set (5:30–6:30, phones only):** 4 people × {live ×10 (vary lighting and distance), photo shown on a screen ×10 (2 devices), video replay ×10 (2 devices)} ≈ 120 attempts. Small but real, and it's what judges trust most. Report counts alongside percentages.

**Identity:** LFW standard 6,000 pairs → face-auth accuracy, FAR, FRR (ArcFace typically scores about 99.8% on LFW, which makes the 99.2% bar realistic). Also cross-person impostor tests from the live set.

**Metrics table (mirror HID's slide):**
| Metric | Definition | Source |
|---|---|---|
| Liveness accuracy | correct live/spoof decisions ÷ all | CelebA-Spoof test subset + live set |
| Face-auth accuracy | correct match/non-match ÷ all pairs | LFW + live set |
| FAR | spoof **or** impostor reaching ALLOW ÷ all attacks | system-level, per attack type |
| FRR | genuine live user DENIED ÷ genuine attempts (REQUEST_VERIFICATION reported separately) | live set |
| Decision time | p50 / p95 of `timings_ms.total` plus capture | live runs |

Plus **per-attack APCER/BPCER** (ISO/IEC 30107-3 terms) and an **ablation**: MiniFASNet alone → + DINOv2 → + device → + planarity. The ablation is the "innovation is measurable" slide.

---

## 9. Demo script (3 minutes)
1. Enroll a judge-volunteer or a teammate (5 s).
2. Live face → **ALLOW**, 1.x s, confidence shown.
3. That person's photo shown on a phone → **DENY**, reasons "phone_detected, planar_motion".
4. Video replay on a laptop → **DENY**.
5. Dim the lights or turn half away → **REQUEST VERIFICATION** → random "blink, turn left" challenge → ALLOW.
6. A different person → **DENY** (identity mismatch).
7. Cut to the dashboard: per-attack metrics, including printed photo and mask from the dataset.

---

## 10. Risks & mitigations
| Risk | Mitigation |
|---|---|
| Dataset download too slow | Start at minute 0; the MVP uses the pretrained MiniFASNet, so the DINOv2 head is an upgrade, not a blocker |
| GB10 ARM64 package issues / VPN drops | NGC PyTorch container; extract features once and copy the `.npy` files to the laptop; nothing in the demo depends on GB10 |
| MiniFASNet false rejects on the webcam | Correct crop scale (2.7 / 4.0), fusion with DINOv2, gray zone → challenge instead of deny |
| Venue Wi-Fi | Edge mode by default; the cloud deployment is shown separately |
| AI-generated code drift between folders | Frozen CONTRACT.md, folder ownership, integration checkpoints at 3:00 and 5:00 |
| Overclaiming metrics | Name the test set for every number; show counts; be ready to explain the difference between dataset and live results |

## Sources
- [CelebA-Spoof (official repo, downloads, licence)](https://github.com/ZhangYuanhan-AI/CelebA-Spoof)
- [CelebA-Spoof paper](https://arxiv.org/abs/2007.12342v3)
- [face-antispoof-onnx (MiniFAS ONNX, Apache-2.0)](https://github.com/suriAI/face-antispoof-onnx)
- [hairymax/Face-AntiSpoofing (ONNX, trained on CelebA-Spoof)](https://awesome.ecosyste.ms/projects/github.com%2Fhairymax%2Fface-antispoofing)
- [uniface spoofing module (MiniFASNet V2 / V1SE, crop scales)](https://yakhyo.github.io/uniface/modules/spoofing)
