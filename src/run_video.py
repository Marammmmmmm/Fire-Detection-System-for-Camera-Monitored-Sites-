#!/usr/bin/env python3
"""
run_video.py
============
Run the alarm pipeline over a video file, an RTSP stream, or a folder of frames,
and report confirmed alarms plus the false-alarm rate.

This is the script to point at real footage from a customer site. Feed it an
hour of ordinary, event-free video and read the "alarms per hour" line at the
bottom: that number, not mAP, decides whether the system is deployable.

Usage
-----
    # a recorded clip
    python run_video.py --model best.onnx --source clip.mp4

    # a live camera, sampling every 5th frame
    python run_video.py --model best.onnx --source rtsp://user:pw@10.0.0.5/live \
        --stride 5 --save-alarms alarms/

    # custom thresholds from calibrate_thresholds.py
    python run_video.py --model best.onnx --source clip.mp4 \
        --conf-fire 0.55 --conf-smoke 0.42

Press Ctrl-C to stop a live stream; the summary still prints.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fire_smoke_detector import (  # noqa: E402
    AlarmConfig, AlarmPipeline, DetectorConfig, FireSmokeDetector, draw,
)


def frames_from(source: str, stride: int):
    """Yield (index, rgb_frame). Handles files, RTSP URLs and image folders."""
    p = Path(source)
    if p.is_dir():
        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        files = sorted(f for f in p.iterdir() if f.suffix.lower() in exts)
        for i, f in enumerate(files):
            if i % stride:
                continue
            img = cv2.imread(str(f), cv2.IMREAD_COLOR)
            if img is not None:
                yield i, cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"cannot open source: {source}")
    i = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if i % stride == 0:
                yield i, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            i += 1
    finally:
        cap.release()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--stride", type=int, default=1,
                    help="process every Nth frame (CCTV rarely needs 25 fps)")
    ap.add_argument("--fps", type=float, default=None,
                    help="source fps; read from the file when omitted")
    ap.add_argument("--conf-fire", type=float, default=0.50)
    ap.add_argument("--conf-smoke", type=float, default=0.35)
    ap.add_argument("--confirm-hits", type=int, default=5)
    ap.add_argument("--confirm-window", type=int, default=8)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--save-alarms", default=None,
                    help="directory to write an annotated frame per alarm")
    ap.add_argument("--show-raw", action="store_true",
                    help="also print every raw detection, not just alarms")
    a = ap.parse_args()

    if a.imgsz % 32:
        raise SystemExit(f"--imgsz must be a multiple of 32 (got {a.imgsz})")

    det = FireSmokeDetector(a.model, DetectorConfig(
        imgsz=a.imgsz,
        conf={"fire": a.conf_fire, "smoke": a.conf_smoke},
        intra_op_threads=a.threads,
    ))
    pipe = AlarmPipeline(det, AlarmConfig(
        confirm_hits=a.confirm_hits, confirm_window=a.confirm_window))

    src_fps = a.fps
    if src_fps is None and not Path(a.source).is_dir():
        cap = cv2.VideoCapture(a.source)
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        cap.release()
    src_fps = src_fps or 25.0
    eff_fps = src_fps / a.stride

    out_dir = Path(a.save_alarms) if a.save_alarms else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    print(f"model     : {a.model}   classes {det.names}")
    print(f"source    : {a.source}")
    print(f"thresholds: fire {a.conf_fire}  smoke {a.conf_smoke}")
    print(f"confirm   : {a.confirm_hits} hits within {a.confirm_window} frames")
    print(f"sampling  : every {a.stride} frame(s) -> {eff_fps:.2f} analysed fps")
    print("-" * 68)

    n_proc = 0
    counts = {"object": 0, "scene": 0, "tiny": 0}
    alarms_all = []
    t0 = time.time()
    try:
        for idx, rgb in frames_from(a.source, a.stride):
            n_proc += 1
            dets, alarms = pipe.update(rgb)
            for d in dets:
                counts[d.kind] += 1
                if a.show_raw:
                    print(f"  frame {idx:>7d}  raw {d.cls_name:5s} {d.score:.3f} "
                          f"[{d.kind}]")
            for al in alarms:
                t_src = idx / src_fps
                alarms_all.append((t_src, al))
                print(f"ALARM  t={t_src:8.1f}s  frame {idx:>7d}  {al.cls_name:12s} "
                      f"peak {al.peak_score:.3f}  confirmed in "
                      f"{al.frames_to_confirm} frames  track {al.track_id}")
                if out_dir:
                    ann = draw(rgb, [d for d in dets if d.kind == "object"])
                    cv2.imwrite(str(out_dir / f"alarm_{idx:07d}_{al.cls_name.replace(':','_')}.jpg"),
                                cv2.cvtColor(ann, cv2.COLOR_RGB2BGR))
    except KeyboardInterrupt:
        print("\ninterrupted")

    wall = time.time() - t0
    video_s = n_proc / eff_fps if eff_fps else 0.0
    print("-" * 68)
    print(f"frames analysed     : {n_proc}")
    print(f"video duration      : {video_s/60:.1f} min ({video_s:.0f} s)")
    print(f"raw detections      : object {counts['object']}, "
          f"scene {counts['scene']}, tiny {counts['tiny']}")
    if n_proc:
        print(f"raw object det/frame: {counts['object']/n_proc:.3f}")
    print(f"confirmed alarms    : {len(alarms_all)}")
    if video_s > 0:
        print(f"ALARMS PER HOUR     : {len(alarms_all)/video_s*3600:.2f}")
        if counts["object"]:
            print(f"  (without confirmation it would be "
                  f"{counts['object']/video_s*3600:.0f}/hour)")
    print(f"processing speed    : {n_proc/max(wall,1e-9):.1f} fps "
          f"({wall/max(n_proc,1)*1000:.0f} ms/frame)")
    print()
    print("If this clip contained no real fire or smoke, every line above marked")
    print("ALARM is a false alarm. That is the number to take to the customer.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
