# Spoof-Specific Detection Approaches — HID "Can You Spot a Spoof?"

Goal: decide ALLOW / REQUEST VERIFICATION / DENY in ≤ 2 s, verifying both **liveness** and **identity**.

## Shared foundation (used by all four detectors)

**Capture:** browser (React + MediaPipe Face Landmarker) grabs a **~0.5 s burst of 5–8 frames** at 640×480+, plus the full frame (not just the face crop — context matters for photo/phone edges).

**Quality gate (client-side, before anything else):** face size, blur (Laplacian variance), exposure, pose angle, single face. Fails → prompt user ("move closer", "more light") or REQUEST VERIFICATION. Low quality is the #1 source of false rejects; filter it out first.

**General passive backbone:** a pretrained face anti-spoofing CNN gives a baseline spoof score for every attack type.
- Fast start: **MiniFASNet** (Silent-Face-Anti-Spoofing) — ~10–30 ms on CPU.
- Stronger option: fine-tune a **foundation-model backbone** (CLIP ViT or **DINOv2 w/ registers**) with a linear/LoRA head on our captures. 2025 work shows these generalize best to unseen attacks/cameras (CCPE, MMDA, VFM benchmark — see Sources).

**Fusion:** each spoof-specific detector below outputs a score in [0,1]. A small logistic regression (trained on our own captures) fuses them with the backbone into one liveness score `L`, and we keep the per-detector scores for **explainability** ("DENY: planar motion + phone bezel").

**Identity:** ArcFace (InsightFace `buffalo_l`) cosine similarity vs. enrolled template, or Rekognition `SearchFacesByImage`. Score `M`.

**Decision:**
```
quality fail / L or M in gray zone   → REQUEST VERIFICATION → active challenge → OTP
any detector confidently fires / M low → DENY
L high & M high                        → ALLOW
```

**Latency budget:** capture 500 ms · upload 100 ms · detectors in parallel ~200 ms · identity ~300 ms → **~1.1–1.3 s**.

---

## 1. Printed Photo

**What gives it away:** the face is **flat** (no 3D parallax), paper has **print texture** (halftone dots, paper grain, reduced color gamut), there's often a **visible paper edge / fingers holding it**, and the **eyes never blink**.

**Fast path (passive, inside the 2 s budget):**
| Signal | Method | Cost |
|---|---|---|
| **Planarity / parallax** (primary) | Track ~50 MediaPipe landmarks across the burst; fit a **homography** (RANSAC). A flat photo moves as one plane → tiny reprojection residual; a real face (nose vs. cheeks vs. ears at different depths) leaves a large residual. Normalize by motion magnitude; if the user is perfectly still, fall back to other cues. | <5 ms |
| Depth from a single frame | Monocular depth (MiDaS-small / Depth-Anything-small) on face crop → variance of depth across face. Flat ⇒ spoof. | 30–60 ms GPU |
| Print texture | **LBP on DWT sub-bands** (Mahore & Tripathi idea) or LBP on HSV/YCbCr channels → SVM. Catches halftone and paper grain. | <10 ms |
| Paper border / hands | Edge + contour detection for a quadrilateral surrounding the face; or YOLO for "hand near face edges". | 10–20 ms |
| Backbone CNN | MiniFASNet / DINOv2 head | 10–30 ms |

**Fallback (gray zone):** random head-turn challenge ("turn left") — a flat photo produces a perspective warp, not a yaw rotation, which the planarity check catches decisively. Photo also can't blink on command.

**Known weak spots:** bent/curved photos (partial parallax) → rely on texture + backbone; very high-quality glossy prints → reflections help (specular highlights are uniform across the paper).

**Test set to record:** each teammate printed on plain paper + glossy photo paper, held flat, bent, with/without visible edges, 3 lighting conditions.

---

## 2. Screen Replay (Photo on a phone/tablet/laptop)

**What gives it away:** **moiré** from camera sampling the pixel grid, **screen glare/specular reflection**, **phone bezel**, **emissive light** (the face *glows* rather than reflects ambient light), color-gamut/gamma shifts, and it's **flat** (same planarity cue as photos).

**Fast path:**
| Signal | Method | Cost |
|---|---|---|
| **Moiré / frequency** (primary) | 2D FFT on face crop per channel (R, G, B, gray) → energy of high-frequency peaks off the main axes. Screens produce periodic peaks; skin doesn't. (Patel, Han, Jain & Ott, TIFS 2016 — moiré analysis for replay detection.) Also a small CNN on the FFT magnitude image works well. | <5 ms |
| **Device detection** | YOLOv8n "cell phone"/"laptop"/"tv" class; check if face bbox lies inside device bbox. Cheap, very high precision when the bezel is visible. | 10–20 ms |
| Planarity | Same homography check as photos. | <5 ms |
| Glare / emissive | Ratio of face brightness to background brightness; specular blob detection; screens often look over-saturated / bluish. | <5 ms |
| Backbone CNN | as above | 10–30 ms |

**Fallback — screen-flash challenge (strongest anti-screen check):** the kiosk display flashes a **random color sequence** (e.g., 4 colors × 150 ms). A real face *reflects* each color with predictable skin response; a phone screen is emissive and barely changes, and a pre-recorded image cannot respond to an unpredictable sequence. (Tang et al., "Face Flashing", NDSS 2018 — 98.8% accuracy; same principle as Rekognition's light challenge and Megvii's flash liveness.) Measure per-channel mean color change of the face region correlated with the emitted sequence. Adds ~0.6–0.8 s — use only in gray zone.

**Known weak spots:** high-PPI OLED screens shot from a distance suppress moiré → device detection + flash + planarity still work; borderless phones held close hide the bezel → moiré + flash.

**Test set:** each face shown on ≥2 phones (one OLED, one LCD), one tablet, one laptop; brightness low/high; near/far; bezel visible/hidden.

---

## 3. Video Replay

**What gives it away:** everything from screen replay (it's still a screen) **plus** it defeats naive blink/motion checks because the face *moves*. So the key is: motion is **not responsive** and still **planar**, and the light behaves like a screen.

**Fast path:**
| Signal | Method | Cost |
|---|---|---|
| **Face–background coupling** (primary) | ⚠️ The nose-parallax planarity check (as built in `api/app/liveness/planarity.py`) does **not** catch a video of a head that turns, because the parallax is recorded *inside* the video. Instead compare optical flow of the face vs. the region around it: on a held phone/laptop, face and surrounding screen content move together. (Not built yet: stretch goal.) | <10 ms |
| Moiré + device detection + glare | Same as #2. | ~20 ms |
| Temporal artifacts | Frame-rate/refresh beating (rolling bands / flicker) across the burst — FFT over time of mean brightness. | <5 ms |
| Backbone CNN (multi-frame) | Average backbone score over the burst. | ~30 ms |

**Fallback — randomized challenge-response (defeats any pre-recorded video):**
1. **Screen-flash color sequence** (above) — a video can't reproduce reflections of a sequence it never saw.
2. **Random action prompt** — "blink twice", "turn right", "smile" in random order; verify with MediaPipe blendshapes (eyeBlink, jawOpen, head yaw) within ~1.5 s. Order randomness is what kills replay.

**Optional (research flex, not on the critical path):** **rPPG** (remote pulse signal). Real skin shows a ~1 Hz blood-volume pulse in the green channel; re-displayed video degrades it. Classic methods need several seconds, but Liu et al. (TIFS 2022, temporal-similarity rPPG) showed detection within ~1 s. Worth running in parallel during the challenge window, not as a gate.

**Known weak spots:** a real-time deepfake/injection attack (virtual camera) bypasses the physical camera entirely — out of scope for presentation attacks but worth a slide; mitigate with the flash challenge (requires physical light interaction).

**Test set:** record each teammate's own enrollment-style video, replay on phone/laptop; include videos with blinking, head turns, and talking.

---

## 4. Mask (3D / Printed)

Two very different sub-cases — we will most likely demo **printed paper masks** (photo printed, cut out, eyes cut out, worn on face); real 3D masks (resin/silicone) are hard to source.

**What gives it away:**
- *Printed mask:* paper texture, **flat-ish** but curved (worn on a face, so some parallax!), visible edges at hairline/jaw, **real eyes through eye holes** with a discontinuity around them, mask doesn't deform when the user talks/smiles.
- *3D mask:* skin-like shape (defeats planarity & depth), but **material texture** differs from skin, **no blood pulse**, rigid — no **facial micro-expressions**, unnatural specular reflection.

**Fast path:**
| Signal | Method | Cost |
|---|---|---|
| **Texture / material** (primary) | DWT+LBP features (Mahore & Tripathi, 2018, 3DMAD) + backbone CNN. Fine-tune the backbone on our mask captures; CelebA-Spoof and HiFiMask contain mask examples for pretraining if license permits. | ~30 ms |
| **Eye-region discontinuity** | Compare texture/color statistics of the periocular ring vs. cheek; paper mask shows a sharp boundary around real eyes. Landmark-anchored patches + small classifier. | <10 ms |
| **Face boundary** | Edge strength along jawline/hairline landmarks (mask edge) vs. smooth skin transition. | <5 ms |
| Depth | Printed flat mask → partially flat (catches the cheap ones). 3D masks pass this; don't rely on it. | 30–60 ms |
| Backbone CNN | as above | 10–30 ms |

**Fallback — expression challenge:** "smile" / "open mouth" / "raise eyebrows". Rigid masks (paper or resin) don't deform: measure landmark displacement in the mouth/cheek region vs. the jaw. Expression blendshapes (`mouthSmile`, `cheekSquint`) stay near zero under a mask while the jaw may move. Screen-flash also helps: resin/paper reflect differently from skin (spectral response).

**Optional (research flex):** rPPG — the canonical 3D-mask defense (Liu et al., ECCV 2016; TransRPPG 2021, 547K params). Masks block the pulse signal. Same caveat: needs ≥1 s of clean video, so run during the challenge window, not in the 2 s fast path.

**Known weak spots:** high-end silicone masks with matching texture → this is where expression + rPPG matter; partial masks (only lower face) → per-region scoring.

**Test set:** printed paper masks (plain + glossy, eyes cut out) of each teammate worn by *another* teammate (this also tests identity: mask of A on B should *pass* identity if it's good — liveness must catch it). If obtainable, a cheap 3D/plastic mask.

---

## Summary matrix

| Detector | Photo | Screen | Video | Mask | Cost |
|---|:-:|:-:|:-:|:-:|---|
| Backbone CNN (MiniFASNet → DINOv2/CLIP head) | ✅ | ✅ | ✅ | ✅ | 10–30 ms |
| Planarity / homography residual | ✅✅ | ✅✅ | ✅✅ | ½ (paper) | <5 ms |
| Moiré / FFT | – | ✅✅ | ✅✅ | – | <5 ms |
| Device (phone/laptop) detection | – | ✅✅ | ✅✅ | – | 10–20 ms |
| DWT + LBP texture | ✅ | ✅ | ✅ | ✅✅ | <10 ms |
| Eye-ring / face-boundary discontinuity | – | – | – | ✅✅ | <10 ms |
| Monocular depth | ✅ | ✅ | ✅ | ½ | 30–60 ms |
| **Fallback:** screen-flash color challenge | ✅ | ✅✅ | ✅✅ | ✅ | +0.7 s |
| **Fallback:** random action / expression | ✅ | ✅ | ✅✅ | ✅✅ | +1.5 s |
| Optional: rPPG pulse | ✅ | ✅ | ✅ | ✅✅ | needs ≥1 s video |

✅✅ = primary detector for that attack.

## Evaluation (ISO/IEC 30107-3 style)
- Per attack type: **APCER** (spoofs accepted), plus **BPCER** (live rejected), **ACER**; identity **FAR/FRR**; latency **p50/p95**.
- Ablation: fusion with vs. without each detector — shows which detector carries each attack.
- Baseline comparison: Rekognition Face Liveness on the same captures.
- Hold out one teammate and one device entirely for testing (prevents overfitting to "our" faces/phones).

## Team split suggestion
1. Capture app + quality gate + challenge UI (flash, action prompts)
2. Planarity + depth + device detection
3. Moiré/FFT + DWT-LBP texture + mask cues
4. Backbone fine-tune + fusion + identity + decision engine
5. Dataset capture + evaluation harness + dashboard (shared)

## Sources
- [CCPE: Domain Generalization for FAS via Content-aware Composite Prompt Engineering (2025)](https://arxiv.org/abs/2504.04470v1)
- [MMDA: Denoising and Alignment for Multimodal FAS (2025)](https://arxiv.org/abs/2505.09484v1)
- [Benchmarking Vision Foundation Models for Domain-Generalizable FAS (review)](https://www.themoonlight.io/review/benchmarking-vision-foundation-models-for-domain-generalizable-face-anti-spoofing)
- [Bottom-Up Domain Prompt Tuning for Generalized FAS (ECCV 2024)](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/8866_ECCV_2024_paper.php)
- [InstructFLIP (ACM MM 2025)](https://arxiv.org/pdf/2507.12060v2)
- [Face Flashing: Secure Liveness Detection via Light Reflections (NDSS 2018)](https://arxiv.org/abs/1801.01949)
- [TransRPPG: rPPG Transformer for 3D Mask PAD (2021)](https://arxiv.org/abs/2104.07419v1)
- [Temporal Similarity of rPPG for Fast 3D Mask PAD (WACV 2020)](https://mlanthology.org/wacv/2020/liu2020wacv-temporal)
- [Mahore & Tripathi, 3D Mask Detection Using DWT and LBP (IEEE)](https://ieeexplore.ieee.org/document/8644807)
- Patel, Han, Jain & Ott, "Secure Face Unlock: Spoof Detection on Smartphones" (IEEE TIFS 2016) — moiré analysis for replay
