"""Record the LIVE test set with the laptop webcam: 5-frame bursts (same as the kiosk), labelled by key.

  python eval/record_testset.py --person alice
Keys:  l = live attempt   s = photo shown on a SCREEN   v = VIDEO replay   p = printed photo   m = mask
       i = impostor (live person who is NOT enrolled; use --person with their name)   q = quit
Saves eval/captures/<label>/<person>_<timestamp>/frame_{0..4}.jpg  (+ meta.json)
Tip: for s/v, show the enrolled person's face on a phone/laptop in front of the webcam. Vary distance & lighting.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2

LABELS = {"l": "live", "s": "screen_replay", "v": "video_replay", "p": "printed_photo", "m": "mask", "i": "impostor"}
OUT = Path(__file__).resolve().parent / "captures"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--person", required=True, help="whose face is shown (enrolled user id)")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--device", default="", help="e.g. iphone13, pixel7, laptop (for screen/video attacks)")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW) if hasattr(cv2, "CAP_DSHOW") else cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    counts = {v: 0 for v in LABELS.values()}
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        view = cv2.flip(frame, 1).copy()
        cv2.putText(view, f"{args.person} | " + "  ".join(f"{k}:{LABELS[k][:6]}={counts[LABELS[k]]}" for k in LABELS),
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow("recorder (q to quit)", view)
        key = chr(cv2.waitKey(1) & 0xFF)
        if key == "q":
            break
        if key in LABELS:
            label = LABELS[key]
            d = OUT / label / f"{args.person}_{int(time.time() * 1000)}"
            d.mkdir(parents=True, exist_ok=True)
            for i in range(5):                       # same burst as the kiosk: 5 frames, ~100 ms apart
                ok, f = cap.read()
                if ok:
                    cv2.imwrite(str(d / f"frame_{i}.jpg"), f)   # RAW, unmirrored
                time.sleep(0.1)
            (d / "meta.json").write_text(json.dumps({"person": args.person, "label": label, "device": args.device,
                                                     "ts": time.time()}))
            counts[label] += 1
            print("saved", d)
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
