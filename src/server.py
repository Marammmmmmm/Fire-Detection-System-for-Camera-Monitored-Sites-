"""
server.py — Fire Detection Unit (web console)
=============================================
FastAPI backend. Serves the operator panel at / and runs analysis jobs in a
background thread so the panel can poll progress.

    python server.py --model model.onnx --thresholds manifest.json

In Colab:
    !python server.py --model "$MODEL_PATH" --thresholds "$MANIFEST_FILE" --colab

Detection itself lives in fire_smoke_detector.py. Nothing here re-implements it.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np
import uvicorn
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from fire_smoke_detector import (
    AlarmConfig, AlarmPipeline, DetectorConfig, FireSmokeDetector, draw,
)

HERE = Path(__file__).resolve().parent
WEB = HERE / "web"

STATE: Dict[str, dict] = {}
CONFIG = {"model": "", "defaults": {"fire": 0.20, "smoke": 0.16}, "meta": {}}
_DET_CACHE: Dict[tuple, FireSmokeDetector] = {}
_LOCK = threading.Lock()

app = FastAPI(title="Fire Detection Unit")


# --------------------------------------------------------------------------

def hhmmss(seconds: float) -> str:
    t = int(timedelta(seconds=float(seconds)).total_seconds())
    return f"{t // 3600:02d}:{(t % 3600) // 60:02d}:{t % 60:02d}"


def get_detector(cf: float, cs: float, min_a: float, max_a: float):
    key = (round(cf, 3), round(cs, 3), round(min_a, 5), round(max_a, 3))
    with _LOCK:
        if key not in _DET_CACHE:
            _DET_CACHE[key] = FireSmokeDetector(CONFIG["model"], DetectorConfig(
                conf={"fire": cf, "smoke": cs},
                min_box_area_frac=min_a, max_box_area_frac=max_a))
        return _DET_CACHE[key]


def reencode_h264(src: str, dst: str) -> str:
    if shutil.which("ffmpeg") is None:
        return src
    try:
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", src,
                        "-c:v", "libx264", "-preset", "veryfast",
                        "-movflags", "+faststart", "-pix_fmt", "yuv420p", dst],
                       check=True, timeout=1800)
        return dst if os.path.exists(dst) and os.path.getsize(dst) else src
    except Exception:
        return src


# --------------------------------------------------------------------------

def run_job(job_id: str, video_path: str, opts: dict):
    job = STATE[job_id]
    workdir = Path(job["dir"])
    try:
        smoke_thr = opts["conf_smoke"] if opts["enable_smoke"] else 0.99
        min_a = 0.0015 if opts["guards"] else 0.0
        max_a = 0.80 if opts["guards"] else 1.01

        det = get_detector(opts["conf_fire"], smoke_thr, min_a, max_a)
        pipe = AlarmPipeline(det, AlarmConfig(
            confirm_hits=opts["confirm_hits"], confirm_window=opts["confirm_window"]))

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError("Cannot decode this file. Convert it to H.264 MP4.")

        src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        stride = max(1, int(opts["stride"]))
        eff_fps = src_fps / stride

        job.update(state="running", w=w, h=h, fps=round(src_fps, 1),
                   total=n_total, message="Scanning footage")

        writer = None
        raw_out = str(workdir / "annotated_raw.mp4")
        if opts["render"]:
            writer = cv2.VideoWriter(raw_out, cv2.VideoWriter_fourcc(*"mp4v"),
                                     max(1.0, eff_fps), (w, h))

        alarms_out, suppressed = [], []
        last_t: Dict[str, float] = {}
        counts = {"object": 0, "scene": 0, "tiny": 0}
        n_proc, idx, first_t = 0, 0, None
        t0 = time.time()

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % stride:
                idx += 1
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            dets, alarms = pipe.update(rgb)
            n_proc += 1
            for d in dets:
                counts[d.kind] += 1
            objs = [d for d in dets if d.kind == "object"]

            if writer is not None:
                ann = draw(rgb, objs) if objs else rgb.copy()
                if alarms:
                    cv2.rectangle(ann, (0, 0), (w - 1, h - 1), (200, 45, 35), 10)
                cv2.putText(ann, hhmmss(idx / src_fps), (14, 34),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
                            cv2.LINE_AA)
                writer.write(cv2.cvtColor(ann, cv2.COLOR_RGB2BGR))

            for al in alarms:
                t_src = idx / src_fps
                prev = last_t.get(al.cls_name)
                if prev is not None and (t_src - prev) < opts["dedupe_s"]:
                    suppressed.append({"time_s": round(t_src, 2),
                                       "class": al.cls_name,
                                       "peak_score": round(al.peak_score, 4),
                                       "reason": f"repeat within {opts['dedupe_s']}s"})
                    continue
                last_t[al.cls_name] = t_src
                if first_t is None:
                    first_t = t_src
                n = len(alarms_out) + 1
                shot = draw(rgb, objs) if objs else rgb
                cv2.imwrite(str(workdir / f"frame_{n:03d}.jpg"),
                            cv2.cvtColor(shot, cv2.COLOR_RGB2BGR),
                            [cv2.IMWRITE_JPEG_QUALITY, 88])
                rec = {"n": n, "time": hhmmss(t_src), "time_s": round(t_src, 2),
                       "class": al.cls_name, "peak_score": round(al.peak_score, 3),
                       "frame": idx, "frames_to_confirm": al.frames_to_confirm,
                       "track": al.track_id}
                alarms_out.append(rec)
                job["alarms"] = alarms_out

            idx += 1
            if n_total:
                job["progress"] = min(idx / n_total, 1.0)
            job["message"] = (f"{len(alarms_out)} alarm(s) · frame {idx}"
                              if alarms_out else f"Scanning · frame {idx}")

        cap.release()
        if writer is not None:
            writer.release()

        wall = time.time() - t0
        video_s = idx / src_fps if src_fps else 0.0
        per_hour = len(alarms_out) / video_s * 3600 if video_s else 0.0
        unfiltered = counts["object"] / video_s * 3600 if video_s else 0.0

        video_rel = None
        if opts["render"] and os.path.exists(raw_out):
            final = reencode_h264(raw_out, str(workdir / "annotated.mp4"))
            video_rel = os.path.basename(final)

        report = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "model": os.path.basename(CONFIG["model"]),
            "model_sha256": CONFIG["meta"].get("onnx_sha256"),
            "source": job["filename"],
            "resolution": f"{w}x{h}", "source_fps": round(src_fps, 2),
            "stride": stride, "analysed_fps": round(eff_fps, 2),
            "frames_analysed": n_proc, "video_seconds": round(video_s, 1),
            "settings": {"conf_fire": opts["conf_fire"],
                         "conf_smoke": smoke_thr if opts["enable_smoke"] else "disabled",
                         "confirm": f"{opts['confirm_hits']} of {opts['confirm_window']}",
                         "size_guards": opts["guards"],
                         "dedupe_seconds": opts["dedupe_s"]},
            "raw_detections": counts,
            "confirmed_alarms": len(alarms_out),
            "duplicate_alarms_suppressed": suppressed,
            "alarms_per_hour": round(per_hour, 2),
            "alarms_per_hour_without_confirmation": round(unfiltered, 1),
            "first_alarm_s": round(first_t, 2) if first_t is not None else None,
            "processing_fps": round(n_proc / max(wall, 1e-9), 1),
            "alarms": alarms_out,
        }
        (workdir / "report.json").write_text(json.dumps(report, indent=2))
        with open(workdir / "alarms.csv", "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["#", "time", "class", "peak_score", "frame",
                         "frames_to_confirm"])
            for a in alarms_out:
                wr.writerow([a["n"], a["time"], a["class"], a["peak_score"],
                             a["frame"], a["frames_to_confirm"]])

        job.update(state="done", progress=1.0, report=report, video=video_rel,
                   message=f"{len(alarms_out)} alarm(s) confirmed")

    except Exception as e:
        job.update(state="error", message=str(e))


# --------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def index():
    return (WEB / "index.html").read_text()


@app.get("/api/config")
def config():
    return {"model": os.path.basename(CONFIG["model"]),
            "defaults": CONFIG["defaults"],
            "meta": CONFIG["meta"]}


@app.post("/api/jobs")
async def create_job(background: BackgroundTasks,
                     file: UploadFile = File(...),
                     conf_fire: float = Form(0.20),
                     enable_smoke: bool = Form(False),
                     conf_smoke: float = Form(0.16),
                     stride: int = Form(5),
                     confirm_hits: int = Form(5),
                     confirm_window: int = Form(8),
                     guards: bool = Form(True),
                     dedupe_s: float = Form(10.0),
                     render: bool = Form(True)):
    job_id = uuid.uuid4().hex[:12]
    workdir = Path(tempfile.mkdtemp(prefix=f"fdu_{job_id}_"))
    dest = workdir / (file.filename or "upload.mp4")
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)

    STATE[job_id] = {"id": job_id, "state": "queued", "progress": 0.0,
                     "message": "Queued", "dir": str(workdir),
                     "filename": file.filename or "upload.mp4",
                     "alarms": [], "report": None, "video": None}
    opts = dict(conf_fire=conf_fire, enable_smoke=enable_smoke,
                conf_smoke=conf_smoke, stride=stride, confirm_hits=confirm_hits,
                confirm_window=confirm_window, guards=guards,
                dedupe_s=dedupe_s, render=render)
    background.add_task(run_job, job_id, str(dest), opts)
    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    j = STATE.get(job_id)
    if not j:
        raise HTTPException(404, "No such job")
    return JSONResponse({k: v for k, v in j.items() if k != "dir"})


@app.get("/api/jobs/{job_id}/frame/{n}")
def alarm_frame(job_id: str, n: int):
    j = STATE.get(job_id)
    if not j:
        raise HTTPException(404, "No such job")
    p = Path(j["dir"]) / f"frame_{n:03d}.jpg"
    if not p.exists():
        raise HTTPException(404, "No such frame")
    return FileResponse(p, media_type="image/jpeg")


@app.get("/api/jobs/{job_id}/video")
def annotated(job_id: str):
    j = STATE.get(job_id)
    if not j or not j.get("video"):
        raise HTTPException(404, "No rendered video for this job")
    return FileResponse(Path(j["dir"]) / j["video"], media_type="video/mp4")


@app.get("/api/jobs/{job_id}/download/{what}")
def download(job_id: str, what: str):
    j = STATE.get(job_id)
    if not j:
        raise HTTPException(404, "No such job")
    names = {"report": "report.json", "csv": "alarms.csv"}
    if what not in names:
        raise HTTPException(400, "Unknown file")
    p = Path(j["dir"]) / names[what]
    if not p.exists():
        raise HTTPException(404, "Not ready")
    return FileResponse(p, filename=names[what], media_type="application/octet-stream")


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--thresholds", default=None)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--colab", action="store_true",
                    help="open the panel inside the Colab notebook")
    a = ap.parse_args()

    CONFIG["model"] = a.model
    if a.thresholds and os.path.exists(a.thresholds):
        try:
            m = json.load(open(a.thresholds))
            t = m.get("thresholds_measured") or {}
            CONFIG["defaults"].update({k: float(v) for k, v in t.items()})
            CONFIG["meta"] = {
                "run_tag": m.get("run_tag"),
                "weights_sha256": (m.get("weights_sha256") or "")[:12],
                "onnx_sha256": (m.get("onnx_sha256") or "")[:12],
                "mAP50": (m.get("metrics_all") or {}).get("mAP50"),
                "evaluated_at": m.get("evaluated_at"),
                "per_class": m.get("metrics_per_class"),
            }
            print("manifest loaded:", CONFIG["meta"].get("run_tag"))
        except Exception as e:
            print("could not read manifest:", e)

    det = FireSmokeDetector(a.model, DetectorConfig())
    print("model  :", a.model)
    print("classes:", det.names)
    print(f"panel  : http://localhost:{a.port}")

    if a.colab:
        try:
            from google.colab import output
            threading.Thread(
                target=lambda: uvicorn.run(app, host="127.0.0.1", port=a.port,
                                           log_level="warning"),
                daemon=True).start()
            time.sleep(2.5)
            output.serve_kernel_port_as_window(a.port)
            print("panel opened in a new tab — keep this cell running")
            while True:
                time.sleep(3600)
        except ImportError:
            print("not running in Colab; starting normally")

    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
