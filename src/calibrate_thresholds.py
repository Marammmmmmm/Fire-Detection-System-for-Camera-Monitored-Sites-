#!/usr/bin/env python3
"""
calibrate_thresholds.py
=======================
Measure the correct per-class confidence thresholds for a specific ONNX file
against a specific labelled split. Run this on the machine that holds the
dataset; it needs no GPU.

The thresholds shipped in DetectorConfig are provisional. They were chosen from
a curve that belongs to a DIFFERENT checkpoint. Do not deploy without running
this against the model you actually intend to ship.

Usage
-----
    python calibrate_thresholds.py \
        --model  best.onnx \
        --images Fire--smoke-Detection-1/test/images \
        --labels Fire--smoke-Detection-1/test/labels

Optional:
    --iou 0.5          IoU for calling a prediction a true positive
    --imgsz 640        must be a multiple of 32
    --out report.json

What it reports, per class:
    * F1-optimal threshold and the F1 there
    * precision / recall at that threshold
    * the threshold needed to reach a target precision (--target-precision)
    * false positives per image at several thresholds

Why per-class: fire and smoke peak at very different confidences on this
dataset. A single global threshold is a compromise that is wrong for both.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fire_smoke_detector import (  # noqa: E402
    DetectorConfig, FireSmokeDetector, _iou_matrix, letterbox, nms,
)

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_labels(path: Path, w: int, h: int) -> tuple[np.ndarray, np.ndarray]:
    """Read a YOLO .txt label file -> (xyxy boxes in pixels, class ids)."""
    if not path.exists():
        return np.zeros((0, 4), np.float32), np.zeros(0, np.int64)
    rows = [r.split() for r in path.read_text().strip().splitlines() if r.strip()]
    if not rows:
        return np.zeros((0, 4), np.float32), np.zeros(0, np.int64)
    a = np.array(rows, dtype=np.float64)
    cls = a[:, 0].astype(np.int64)
    cx, cy, bw, bh = a[:, 1] * w, a[:, 2] * h, a[:, 3] * w, a[:, 4] * h
    boxes = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], 1)
    return boxes.astype(np.float32), cls


def read_image(p: Path) -> np.ndarray:
    import cv2
    img = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"cannot read {p}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def raw_predictions(det: FireSmokeDetector, img: np.ndarray):
    """All detections above a floor of 0.01, before any threshold decision."""
    h0, w0 = img.shape[:2]
    x, r, (dw, dh) = det.preprocess(img)
    pred = det.session.run(None, {det.input_name: x})[0][0].T
    sc = pred[:, 4:]
    ids = sc.argmax(1)
    s = sc[np.arange(len(sc)), ids]
    m = s >= 0.01
    if not m.any():
        return np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, np.int64)
    cx, cy, bw, bh = pred[m][:, :4].T
    boxes = np.stack([(cx - bw / 2 - dw) / r, (cy - bh / 2 - dh) / r,
                      (cx + bw / 2 - dw) / r, (cy + bh / 2 - dh) / r], 1)
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w0)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h0)
    s, ids = s[m], ids[m]
    keep_all = []
    for c in np.unique(ids):
        idx = np.where(ids == c)[0]
        for k in nms(boxes[idx], s[idx], det.cfg.iou):
            keep_all.append(idx[k])
    keep_all = np.array(keep_all, np.int64)
    return boxes[keep_all], s[keep_all], ids[keep_all]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--labels", default=None,
                    help="default: sibling 'labels' dir next to --images")
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--target-precision", type=float, default=0.90)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    img_dir = Path(a.images)
    lbl_dir = Path(a.labels) if a.labels else img_dir.parent / "labels"
    files = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXT)
    if not files:
        print(f"no images in {img_dir}", file=sys.stderr)
        return 1

    det = FireSmokeDetector(
        a.model,
        DetectorConfig(imgsz=a.imgsz, conf={n: 0.01 for n in ["fire", "smoke"]},
                       min_box_area_frac=0.0, max_box_area_frac=1.01),
    )
    names = det.names
    print(f"model classes : {names}")
    print(f"images        : {len(files)}  from {img_dir}")
    print(f"labels        : {lbl_dir}")
    print(f"match IoU     : {a.iou}")
    print()

    # scored[c] = list of (score, is_true_positive); n_gt[c] = ground-truth count
    scored = {c: [] for c in range(len(names))}
    n_gt = {c: 0 for c in range(len(names))}
    n_bg_images = 0

    for i, f in enumerate(files, 1):
        if i % 100 == 0 or i == len(files):
            print(f"  {i}/{len(files)}", end="\r", flush=True)
        img = read_image(f)
        h, w = img.shape[:2]
        gt_boxes, gt_cls = load_labels(lbl_dir / (f.stem + ".txt"), w, h)
        if len(gt_boxes) == 0:
            n_bg_images += 1
        for c in gt_cls:
            n_gt[int(c)] += 1

        p_boxes, p_scores, p_cls = raw_predictions(det, img)
        for c in range(len(names)):
            pm = p_cls == c
            gm = gt_cls == c
            pb, ps = p_boxes[pm], p_scores[pm]
            gb = gt_boxes[gm]
            order = ps.argsort()[::-1]
            pb, ps = pb[order], ps[order]
            taken = np.zeros(len(gb), bool)
            ious = _iou_matrix(pb, gb) if len(gb) else np.zeros((len(pb), 0), np.float32)
            for k in range(len(pb)):
                tp = False
                if ious.shape[1]:
                    cand = np.where((ious[k] >= a.iou) & (~taken))[0]
                    if len(cand):
                        j = cand[ious[k][cand].argmax()]
                        taken[j] = True
                        tp = True
                scored[c].append((float(ps[k]), tp))
    print()

    if n_bg_images == 0:
        print("NOTE: this split contains zero background images (every image has "
              "at least one object). False-positive rates measured here are "
              "therefore optimistic — the model is never shown an empty scene.")
        print()

    report = {"model": a.model, "images": len(files), "iou": a.iou,
              "background_images": n_bg_images, "classes": {}}

    grid = np.round(np.arange(0.05, 0.96, 0.01), 2)
    for c, name in enumerate(names):
        arr = sorted(scored[c], key=lambda t: -t[0])
        s = np.array([t[0] for t in arr], np.float32)
        tp = np.array([t[1] for t in arr], bool)
        best = {"f1": -1.0}
        rows = []
        for t in grid:
            m = s >= t
            n_pred = int(m.sum())
            n_tp = int(tp[m].sum())
            n_fp = n_pred - n_tp
            prec = n_tp / n_pred if n_pred else 0.0
            rec = n_tp / n_gt[c] if n_gt[c] else 0.0
            f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
            rows.append({"thr": float(t), "precision": prec, "recall": rec,
                         "f1": f1, "fp_per_image": n_fp / len(files)})
            if f1 > best["f1"]:
                best = {"thr": float(t), "f1": f1, "precision": prec,
                        "recall": rec, "fp_per_image": n_fp / len(files)}

        hit_target = next((r for r in rows if r["precision"] >= a.target_precision), None)

        print(f"=== {name} ===  ground-truth instances: {n_gt[c]}")
        if n_gt[c] == 0:
            print("  no instances of this class in the split — cannot calibrate")
            report["classes"][name] = {"gt": 0}
            continue
        print(f"  F1-optimal threshold      : {best['thr']:.2f}")
        print(f"    F1 / precision / recall : {best['f1']:.3f} / "
              f"{best['precision']:.3f} / {best['recall']:.3f}")
        print(f"    false positives / image : {best['fp_per_image']:.3f}")
        if hit_target:
            print(f"  threshold for precision>={a.target_precision:.2f} : "
                  f"{hit_target['thr']:.2f}  (recall drops to {hit_target['recall']:.3f}, "
                  f"FP/img {hit_target['fp_per_image']:.3f})")
        else:
            print(f"  precision never reaches {a.target_precision:.2f} at any threshold")
        print("  FP/image across thresholds:", "  ".join(
            f"{r['thr']:.2f}:{r['fp_per_image']:.2f}"
            for r in rows if abs(r["thr"] * 100 % 10) < 1e-6))
        print()
        report["classes"][name] = {"gt": n_gt[c], "best": best,
                                   "target_precision": hit_target, "curve": rows}

    usable = {n: report["classes"][n]["best"]["thr"]
              for n in names if report["classes"][n].get("best")}
    if usable:
        print("Suggested DetectorConfig(conf=...):")
        print(f"  conf={{{', '.join(f'{k!r}: {v:.2f}' for k, v in usable.items())}}}")
        print("  (F1-optimal. For an alarm system, prefer the higher "
              "precision-target thresholds above — a missed frame is recovered by "
              "the next frame, a false alarm is not.)")

    if a.out:
        Path(a.out).write_text(json.dumps(report, indent=2))
        print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
