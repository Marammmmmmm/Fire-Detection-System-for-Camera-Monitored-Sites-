# Fire Detection System for Camera-Monitored Sites

**Object Detection Final Project — DOTPY Academy, Computer Vision Track**
YOLO11n · Transfer learning from COCO

A visual fire detector that finds fire in video from ordinary surveillance
cameras and raises a *confirmed* alarm, with a measured false-alarm rate rather
than a claimed one.

---

## Results

| | |
|---|---|
| **mAP@0.5** on held-out test split | **0.710** — above the dataset publisher's own 0.681 |
| Fire | precision 0.935, recall 0.864 at the calibrated threshold 0.43 |
| Smoke | precision 0.525, recall 0.525 — **not deployable**, disabled in the demo |
| False alarms on 70 s of event-free footage | **0 confirmed** (360/hour before temporal filtering) |
| Detection latency on the fire clip | **2.0 s** |
| Speed | 4.9 ms/frame inference on a Tesla T4 (~204 fps) |

Full analysis in **[`SRS_Documentation.pdf`](SRS_Documentation.pdf)**.

---

## Repository layout

```
model/
   run1_20ep.pt                shipped weights — 20 epochs      SHA-256 ff6a0d6b…
   run2_v2_50ep.pt             rejected weights — 50 epochs
   run1_20ep_20260817.onnx     exported model used by the demo  SHA-256 99ea3776…

notebooks/
   01_Training_and_Model_Comparison.ipynb    run once  (~4 h on a T4)
   02_Evaluation_Testing_and_Demo.ipynb      run anytime (~5 min)

src/
   fire_smoke_detector.py      inference wrapper — letterbox, NMS, confirmation
   calibrate_thresholds.py     threshold measurement on a labelled split
   run_video.py                video runner, reports alarms per hour
   server.py                   optional web console (not a required deliverable)
   web/index.html

reports/
   manifest_run1_20ep_20260817.json      weights hash ↔ metrics ↔ ONNX hash
   thresholds_run1_20ep_20260817.json    full calibration curve, per class
   model_selection_20260817.json         why run 2 was rejected

results/
   metrics/        F1 curve, confusion matrix, run 2 overfitting curves
   predictions/    unseen and external test images with boxes drawn
   demo/           console screenshots

SRS_Documentation.pdf
```

Both sets of weights are published, not just the shipped one, so the comparison
between the two training runs can be verified rather than taken on trust.

---

## Why there are two notebooks

Notebook 01 takes about four hours of GPU time. Re-running it to re-check a
metric wastes that time *and* produces a different set of weights each run, which
makes numbers impossible to attribute to a model. Training happens once, the
weights are saved, and everything downstream loads them.

| Notebook | Purpose | Run time |
|---|---|---|
| **01** | Downloads the dataset, fine-tunes YOLO11n twice, saves both sets of weights | ~4 h on a T4 — **run once** |
| **02** | Loads saved weights, evaluates, calibrates thresholds, tests on unseen images, runs the demo | ~5 min — **run anytime** |

Both notebooks keep their stored outputs, so they can be read without executing
anything.

---

## Running it

```bash
pip install ultralytics roboflow onnxruntime opencv-python
```

A GPU is needed only for notebook 01.

The dataset is public on Roboflow Universe but the download needs an API key.
Both notebooks read it from Colab Secrets rather than hardcoding it: click the
key icon in the Colab sidebar, add a secret named `ROBOFLOW_KEY`, and enable
notebook access.

To reproduce the headline figure: open notebook 02, run sections 1–3, point
`WEIGHTS` at `model/run1_20ep.pt` in section 5, and run it.

---

## Dataset

| | |
|---|---|
| Name | Fire-smoke Detection (`fire-smoke-detection-lk8z9`), version 1 |
| Source | [Roboflow Universe — detection-projet](https://universe.roboflow.com/detection-projet/fire-smoke-detection-lk8z9/dataset/1) |
| Licence | **CC BY 4.0** — free use with attribution |
| Classes | `fire`, `smoke` (`nc: 2`) |
| Split | 10,589 train / 1,017 validation / 521 test — **12,127 images** |
| Test instances | 479 fire, 99 smoke |

Already annotated in YOLO format; no manual annotation was performed.

**Known weaknesses**, analysed in the SRS §08:

1. Class imbalance of roughly 5:1 against smoke
2. Augmentation applied *before* the split — 386 of the 521 test filenames carry
   an `_aug` suffix and trace back to ~133 distinct source frames, so test
   metrics are optimistic by an unknown margin
3. **Zero background images** — every image contains an object, so the model was
   never shown an empty scene during training
4. Predominantly daylight imagery

---

## Model

| | |
|---|---|
| Architecture | YOLO11 nano — 2,582,542 parameters, 6.4 GFLOPs |
| Transfer learning | Initialised from `yolo11n.pt` (COCO-pretrained) |
| Run 1 — **shipped** | `epochs=20, imgsz=640, batch=32, optimizer=auto` |
| Run 2 — rejected | `epochs=50, imgsz=640, batch=16, optimizer=AdamW, lr0=0.001` |
| Export | ONNX opset 20, dynamic axes, BatchNorm fused, NMS external |

**The 20-epoch run beat the 50-epoch run** on the held-out test split — 0.710 vs
0.686 mAP@0.5, with smoke recall falling from 0.556 to 0.414. Validation metrics
had suggested the opposite; only the held-out split revealed it.

### Model provenance

Weights were downloaded and re-uploaded several times during the project until
the files were named `best.pt`, `best (1).pt` and `best (1) (1).pt` — at which
point the exported model could no longer be attributed to a training run.

Every evaluation and export now records a SHA-256 hash in a manifest, so the
chain is explicit:

```
weights ff6a0d6b…  →  mAP@0.5 0.710  →  ONNX 99ea3776…  →  threshold fire 0.20
```

An unplanned test confirmed the value of this. Re-running the export a day later
from the identical weights file produced identical metrics but a *different*
ONNX file — `99ea3776…` against `07d3fd22…`. The export is not bit-deterministic.
Tracking by filename would have treated the two as interchangeable.

See `reports/manifest_run1_20ep_20260817.json`.

---

## What the inference wrapper adds

`model.predict()` is enough for single images but not for video.
`src/fire_smoke_detector.py` adds four things:

| Problem | Fix |
|---|---|
| Resizing a 16:9 frame to a square distorts objects by 1.78×; training used letterbox | aspect-preserving letterbox, grey 114 padding |
| One confidence threshold for two classes that peak at very different values | per-class thresholds |
| The ONNX export carries no NMS (`nms=False`) | class-wise NMS |
| Each frame is scored independently, so every flicker became an alarm | temporal confirmation — 5 detections within 8 frames at the same location |

On 70 seconds of event-free office footage this is the difference between **360
alarms per hour** and **zero**.

---

## Known limitations

1. **Smoke detection does not work.** At its best operating point half the alarms
   are false and half the real smoke is missed; reaching 90% precision requires a
   threshold of 0.86, at which recall collapses to 0.020 — 2 of 99 instances.
   Since smoke normally precedes visible flame, the weak class is the one that
   would have provided early warning. It is disabled in the demo, and its metrics
   are still reported in full.
2. **Night-time failure.** One unseen night scene produced no detection at all.
   The training data is predominantly daylight, and for a fire alarm a night-time
   miss is the dangerous failure direction.
3. **False-alarm rate is indicative, not established.** Seventy seconds of one
   daylight scene is consistent with any true rate below roughly 50 per hour.
4. **Test metrics are optimistic** because of the augmentation leakage above.

Priorities for improvement: background images (1,500–2,500, no annotation
needed) · 800–1,000 more smoke instances · night footage · re-splitting at
source-clip level. Detailed in the SRS §14.

---

## Credits

Dataset: *Fire-smoke Detection* by **detection projet**, Roboflow Universe,
licensed CC BY 4.0.
Model: YOLO11 by **Ultralytics** (AGPL-3.0).
