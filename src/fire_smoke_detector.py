"""
fire_smoke_detector.py
======================
Production inference wrapper for the YOLO11n fire/smoke ONNX export.

Fixes three defects found in the original Colab/Gradio inference code:

  1. cv2.resize(img, (640, 640))  ->  aspect-preserving letterbox.
     The model was trained with letterbox. Stretching a 16:9 camera frame to a
     square changes object geometry and costs accuracy.

  2. Single global confidence threshold  ->  per-class thresholds.
     Fire and smoke peak at very different confidences; one threshold is a
     compromise that is wrong for both.

  3. Per-frame alarms  ->  temporal confirmation.
     The model produces ~0.18 false detections per frame. Requiring a detection
     to persist across N consecutive frames in the same place turns that into a
     usable alarm rate.

The model output carries NO built-in NMS (exported with nms=False), so NMS is
implemented here.

Model contract (read from the ONNX metadata, do not hardcode elsewhere):
    input   'images'  : (batch, 3, H, W) float32, RGB, scaled to 0..1
    output  'output0' : (batch, 6, anchors) -> [cx, cy, w, h, score_fire, score_smoke]
    Box coordinates are in LETTERBOXED input pixel space, not source pixels.
    Class scores are already sigmoid-activated.
    H and W must both be multiples of 32.
"""

from __future__ import annotations

import ast
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import onnxruntime as ort

try:
    import cv2
except ImportError:  # pragma: no cover - cv2 is expected in deployment
    cv2 = None


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

@dataclass
class DetectorConfig:
    """Detection-stage settings."""

    imgsz: int = 640
    # Per-class score thresholds. Calibrate with calibrate_thresholds.py on a
    # labelled test split before deployment; these are provisional starting
    # points, not measured optima for any particular checkpoint.
    conf: Dict[str, float] = field(
        default_factory=lambda: {"fire": 0.50, "smoke": 0.35}
    )
    iou: float = 0.45              # NMS IoU threshold
    max_det: int = 100
    # Reject boxes smaller than this fraction of frame area. Set 0.0 to disable.
    min_box_area_frac: float = 0.0015
    # Boxes LARGER than this fraction are reclassified as 'scene' rather than
    # 'object'. Measured behaviour of this checkpoint: on low-texture frames
    # (flat sky, plain wall, night scene with lamps) it emits a degenerate box
    # covering 99-100% of the frame at up to 0.94 confidence. Textured scenes
    # produced none. Scene detections are NOT discarded — they are routed to a
    # separate, much slower confirmation path, because a genuine event can also
    # fill the frame (lens fully obscured by smoke).
    max_box_area_frac: float = 0.80
    providers: Sequence[str] = ("CPUExecutionProvider",)
    intra_op_threads: int = 0      # 0 = let onnxruntime decide


@dataclass
class AlarmConfig:
    """Temporal-confirmation and alarm settings."""

    confirm_hits: int = 5          # detections needed to raise an alarm
    confirm_window: int = 8        # ...within this many consecutive frames
    match_iou: float = 0.30        # IoU to consider two frames' boxes the same object
    max_age: int = 10              # drop a track after this many frames with no hit
    cooldown_s: float = 60.0       # suppress repeat alarms for the same track
    # Scene-level ('whole frame') detections use a much longer confirmation,
    # because this checkpoint produces them spuriously on low-texture frames.
    # A real lens-obscured event persists indefinitely and still gets through.
    scene_confirm_hits: int = 40
    scene_cooldown_s: float = 300.0


# --------------------------------------------------------------------------
# Data types
# --------------------------------------------------------------------------

@dataclass
class Detection:
    """One detection in SOURCE image pixel coordinates.

    `kind` records which size band the box fell into:
      'object' - normal detection, feeds the fast alarm path
      'scene'  - covers more than max_box_area_frac of the frame
      'tiny'   - below min_box_area_frac
    """

    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    cls_id: int
    cls_name: str
    kind: str = "object"

    @property
    def xyxy(self) -> Tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)

    @property
    def area(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)


@dataclass
class Alarm:
    """A confirmed event, emitted once per track (subject to cooldown)."""

    cls_name: str
    box: Tuple[float, float, float, float]
    peak_score: float
    hits: int
    frames_to_confirm: int
    track_id: int
    timestamp: float


# --------------------------------------------------------------------------
# Geometry helpers
# --------------------------------------------------------------------------

def letterbox(
    img: np.ndarray,
    new_shape: int = 640,
    color: Tuple[int, int, int] = (114, 114, 114),
) -> Tuple[np.ndarray, float, Tuple[float, float]]:
    """Resize preserving aspect ratio, pad to a square.

    Returns (padded_image, scale_ratio, (pad_left, pad_top)).
    Padding colour 114 matches the Ultralytics training default; black padding
    would introduce a border the model never saw during training.
    """
    h, w = img.shape[:2]
    r = min(new_shape / h, new_shape / w)
    new_w, new_h = int(round(w * r)), int(round(h * r))

    if (new_w, new_h) != (w, h):
        if cv2 is not None:
            interp = cv2.INTER_LINEAR if r > 1 else cv2.INTER_AREA
            img = cv2.resize(img, (new_w, new_h), interpolation=interp)
        else:
            from PIL import Image as _Image
            img = np.asarray(
                _Image.fromarray(img).resize((new_w, new_h), _Image.BILINEAR)
            )

    dw, dh = (new_shape - new_w) / 2, (new_shape - new_h) / 2
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

    out = np.full((new_shape, new_shape, 3), color, dtype=img.dtype)
    out[top:top + new_h, left:left + new_w] = img
    return out, r, (left, top)


def _iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between two sets of xyxy boxes -> (len(a), len(b))."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    area_a = (a[:, 2] - a[:, 0]).clip(0) * (a[:, 3] - a[:, 1]).clip(0)
    area_b = (b[:, 2] - b[:, 0]).clip(0) * (b[:, 3] - b[:, 1]).clip(0)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = (rb - lt).clip(0)
    inter = wh[..., 0] * wh[..., 1]
    union = area_a[:, None] + area_b[None, :] - inter
    return (inter / np.maximum(union, 1e-9)).astype(np.float32)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> List[int]:
    """Greedy non-maximum suppression. Returns kept indices, highest score first."""
    if len(boxes) == 0:
        return []
    order = scores.argsort()[::-1]
    keep: List[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        ious = _iou_matrix(boxes[i:i + 1], boxes[order[1:]])[0]
        order = order[1:][ious <= iou_thr]
    return keep


# --------------------------------------------------------------------------
# Detector
# --------------------------------------------------------------------------

class FireSmokeDetector:
    """Single-frame detector. Stateless — safe to share across camera streams."""

    def __init__(self, onnx_path: str, cfg: Optional[DetectorConfig] = None):
        self.cfg = cfg or DetectorConfig()

        so = ort.SessionOptions()
        if self.cfg.intra_op_threads:
            so.intra_op_num_threads = self.cfg.intra_op_threads
        self.session = ort.InferenceSession(
            onnx_path, sess_options=so, providers=list(self.cfg.providers)
        )
        self.input_name = self.session.get_inputs()[0].name

        # Class names come from the model file, never from a hardcoded list.
        # A wrong class order silently swaps fire and smoke.
        meta = self.session.get_modelmeta().custom_metadata_map
        if "names" in meta:
            d = ast.literal_eval(meta["names"])
            self.names = [d[i] for i in sorted(d)]
        else:
            raise ValueError(
                "ONNX file has no 'names' metadata; refusing to guess class order. "
                "Re-export with Ultralytics, which embeds it."
            )

        if self.cfg.imgsz % 32:
            raise ValueError(
                f"imgsz={self.cfg.imgsz} is not a multiple of 32. This export fails "
                "at the model.12 Concat node on other sizes."
            )

        self._unknown = [n for n in self.cfg.conf if n not in self.names]
        if self._unknown:
            raise ValueError(
                f"conf thresholds name unknown classes {self._unknown}; "
                f"model classes are {self.names}"
            )

    # -- preprocessing -----------------------------------------------------

    def preprocess(self, img_rgb: np.ndarray):
        padded, r, (dw, dh) = letterbox(img_rgb, self.cfg.imgsz)
        x = padded.astype(np.float32) / 255.0
        x = np.ascontiguousarray(x.transpose(2, 0, 1)[None])
        return x, r, (dw, dh)

    # -- inference ---------------------------------------------------------

    def __call__(self, img_rgb: np.ndarray) -> List[Detection]:
        return self.detect(img_rgb)

    def detect(self, img_rgb: np.ndarray) -> List[Detection]:
        """Run detection on one RGB frame. Returns boxes in source coordinates."""
        if img_rgb.ndim != 3 or img_rgb.shape[2] != 3:
            raise ValueError(f"expected HxWx3 RGB frame, got shape {img_rgb.shape}")

        h0, w0 = img_rgb.shape[:2]
        x, r, (dw, dh) = self.preprocess(img_rgb)
        raw = self.session.run(None, {self.input_name: x})[0]     # (1, 6, N)
        pred = raw[0].T                                            # (N, 6)

        boxes_cxcywh = pred[:, :4]
        scores_all = pred[:, 4:]
        cls_ids = scores_all.argmax(1)
        scores = scores_all[np.arange(len(scores_all)), cls_ids]

        # Per-class threshold, applied before NMS so a weak class cannot be
        # suppressed by a strong neighbour it should not compete with.
        thr = np.array(
            [self.cfg.conf.get(n, 1.01) for n in self.names], dtype=np.float32
        )
        keep = scores >= thr[cls_ids]
        if not keep.any():
            return []

        boxes_cxcywh = boxes_cxcywh[keep]
        scores = scores[keep]
        cls_ids = cls_ids[keep]

        # letterbox space -> source pixels
        cx, cy, bw, bh = boxes_cxcywh.T
        x1 = (cx - bw / 2 - dw) / r
        y1 = (cy - bh / 2 - dh) / r
        x2 = (cx + bw / 2 - dw) / r
        y2 = (cy + bh / 2 - dh) / r
        boxes = np.stack([x1, y1, x2, y2], 1)
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w0)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h0)

        # Class-wise NMS
        frame_area = float(w0 * h0)
        out: List[Detection] = []
        for c in np.unique(cls_ids):
            m = cls_ids == c
            for i in nms(boxes[m], scores[m], self.cfg.iou):
                b = boxes[m][i]
                frac = ((b[2] - b[0]) * (b[3] - b[1])) / frame_area
                if frac < self.cfg.min_box_area_frac:
                    kind = "tiny"
                elif frac > self.cfg.max_box_area_frac:
                    kind = "scene"
                else:
                    kind = "object"
                out.append(
                    Detection(
                        float(b[0]), float(b[1]), float(b[2]), float(b[3]),
                        float(scores[m][i]), int(c), self.names[int(c)], kind,
                    )
                )

        out.sort(key=lambda d: d.score, reverse=True)
        return out[: self.cfg.max_det]

    def detect_objects(self, img_rgb: np.ndarray) -> List[Detection]:
        """Only the detections that feed the fast alarm path."""
        return [d for d in self.detect(img_rgb) if d.kind == "object"]


# --------------------------------------------------------------------------
# Temporal confirmation
# --------------------------------------------------------------------------

class _Track:
    __slots__ = ("id", "cls_id", "box", "hits", "age", "first_frame",
                 "peak_score", "alarmed_at")

    def __init__(self, tid: int, det: Detection, frame_idx: int):
        self.id = tid
        self.cls_id = det.cls_id
        self.box = np.array(det.xyxy, dtype=np.float32)
        self.hits = 1
        self.age = 0
        self.first_frame = frame_idx
        self.peak_score = det.score
        self.alarmed_at: Optional[float] = None


class AlarmPipeline:
    """Stateful per-camera wrapper: detection + temporal confirmation.

    Create ONE instance per camera stream. Feed frames in order.

    A detection only becomes an alarm after it has been seen `confirm_hits`
    times within `confirm_window` consecutive frames at roughly the same
    location. Isolated single-frame detections — which is what most false
    positives look like — never reach the operator.
    """

    def __init__(
        self,
        detector: FireSmokeDetector,
        cfg: Optional[AlarmConfig] = None,
    ):
        self.det = detector
        self.cfg = cfg or AlarmConfig()
        self.tracks: List[_Track] = []
        self._next_id = 1
        self._frame = 0
        self._scene_hits: Dict[int, int] = {}
        self._scene_alarmed: Dict[int, float] = {}

    def reset(self) -> None:
        self.tracks.clear()
        self._scene_hits.clear()
        self._scene_alarmed.clear()
        self._frame = 0

    def _handle_scene(self, scene_dets, now: float) -> List[Alarm]:
        """Slow path for whole-frame detections."""
        alarms: List[Alarm] = []
        seen = set()
        for d in scene_dets:
            seen.add(d.cls_id)
            n = self._scene_hits.get(d.cls_id, 0) + 1
            self._scene_hits[d.cls_id] = n
            last = self._scene_alarmed.get(d.cls_id)
            if n >= self.cfg.scene_confirm_hits and (
                last is None or now - last >= self.cfg.scene_cooldown_s
            ):
                self._scene_alarmed[d.cls_id] = now
                alarms.append(
                    Alarm(
                        cls_name=f"{d.cls_name}:scene",
                        box=d.xyxy,
                        peak_score=d.score,
                        hits=n,
                        frames_to_confirm=n,
                        track_id=-1 - d.cls_id,
                        timestamp=now,
                    )
                )
        for cid in list(self._scene_hits):
            if cid not in seen:
                self._scene_hits[cid] = 0
        return alarms

    def update(
        self, img_rgb: np.ndarray, now: Optional[float] = None
    ) -> Tuple[List[Detection], List[Alarm]]:
        """Process one frame. Returns (raw detections, newly confirmed alarms)."""
        now = time.time() if now is None else now
        self._frame += 1
        all_dets = self.det.detect(img_rgb)
        dets = [d for d in all_dets if d.kind == "object"]
        alarms: List[Alarm] = self._handle_scene(
            [d for d in all_dets if d.kind == "scene"], now
        )

        for t in self.tracks:
            t.age += 1

        if dets:
            # Snapshot the track list: new tracks created inside this loop must
            # not shift the column indices of the IoU matrix.
            existing = list(self.tracks)
            new_tracks: List[_Track] = []
            det_boxes = np.array([d.xyxy for d in dets], dtype=np.float32)
            trk_boxes = (
                np.array([t.box for t in existing], dtype=np.float32)
                if existing else np.zeros((0, 4), np.float32)
            )
            ious = _iou_matrix(det_boxes, trk_boxes)

            used_tracks: set = set()
            for di, d in enumerate(dets):
                best_ti, best_iou = -1, self.cfg.match_iou
                for ti, t in enumerate(existing):
                    if ti in used_tracks or t.cls_id != d.cls_id:
                        continue
                    if ious[di, ti] > best_iou:
                        best_ti, best_iou = ti, ious[di, ti]

                if best_ti >= 0:
                    t = existing[best_ti]
                    used_tracks.add(best_ti)
                    t.box = np.array(d.xyxy, dtype=np.float32)
                    t.hits += 1
                    t.age = 0
                    t.peak_score = max(t.peak_score, d.score)
                    # Window expired without enough hits -> restart counting.
                    span = self._frame - t.first_frame + 1
                    if span > self.cfg.confirm_window and t.alarmed_at is None:
                        t.first_frame = self._frame
                        t.hits = 1
                    elif (
                        t.hits >= self.cfg.confirm_hits
                        and (t.alarmed_at is None
                             or now - t.alarmed_at >= self.cfg.cooldown_s)
                    ):
                        t.alarmed_at = now
                        alarms.append(
                            Alarm(
                                cls_name=self.det.names[t.cls_id],
                                box=tuple(float(v) for v in t.box),
                                peak_score=t.peak_score,
                                hits=t.hits,
                                frames_to_confirm=span,
                                track_id=t.id,
                                timestamp=now,
                            )
                        )
                else:
                    new_tracks.append(_Track(self._next_id, d, self._frame))
                    self._next_id += 1

            self.tracks.extend(new_tracks)

        self.tracks = [t for t in self.tracks if t.age <= self.cfg.max_age]
        return all_dets, alarms


# --------------------------------------------------------------------------
# Drawing (optional convenience)
# --------------------------------------------------------------------------

_COLORS = {"fire": (255, 64, 0), "smoke": (120, 160, 200)}


def draw(img_rgb: np.ndarray, dets: Sequence[Detection]) -> np.ndarray:
    """Annotate a copy of the frame. Input and output are both RGB."""
    if cv2 is None:
        raise RuntimeError("draw() requires opencv")
    out = img_rgb.copy()
    for d in dets:
        c = _COLORS.get(d.cls_name, (0, 255, 0))
        p1, p2 = (int(d.x1), int(d.y1)), (int(d.x2), int(d.y2))
        cv2.rectangle(out, p1, p2, c, 2)
        label = f"{d.cls_name} {d.score:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (p1[0], p1[1] - th - 6), (p1[0] + tw + 4, p1[1]), c, -1)
        cv2.putText(out, label, (p1[0] + 2, p1[1] - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                    cv2.LINE_AA)
    return out
