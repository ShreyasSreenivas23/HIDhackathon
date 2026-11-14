# Evaluation (P4)

`pip install -r ../api/requirements.txt streamlit pandas`. Results are written to `eval/results/` (gitignored).

| Step | Command | Produces |
|---|---|---|
| Identity (LFW) | `python eval/eval_lfw.py --lfw data/lfw --pairs data/lfw/pairs.txt` | face-auth accuracy, FAR/FRR |
| Liveness (dataset) | `python eval/eval_celeba.py --root /data/CelebA_Spoof --per-type 300` | APCER for each of photo, screen and mask · ablation |
| Record live set | `python eval/record_testset.py --person alice --device pixel7` | `eval/captures/…` |
| Live system test | enroll everyone, then `python eval/eval_live.py` | system FAR/FRR for each attack, latency |
| Calibrate | `python eval/calibrate.py` (dry run) → `--write` | `api/thresholds.json` |
| Dashboard | `streamlit run eval/dashboard.py` | HID-format slide numbers |

**Honesty rules for the slides:** name the test set for every number, show `n`, and report REQUEST_VERIFICATION separately from DENY. Calibrate thresholds on CelebA-Spoof *train/val* or on part of the live set, then report results on held-out data.
