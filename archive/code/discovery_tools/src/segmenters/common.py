"""Shared helpers for the segmenters: device, media I/O, JSON output."""
from __future__ import annotations

import gc
import json
import subprocess
import time
from pathlib import Path

import numpy as np


def get_device(pref: str = "auto") -> str:
    if pref != "auto":
        return pref
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def free_gpu():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def probe(video: Path) -> dict:
    import cv2
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return {"fps": fps, "frames": n, "duration": n / fps if fps else 0.0, "width": w, "height": h}


def extract_audio(video: Path, out_wav: Path, sr: int = 16000) -> Path:
    """Mono 16 kHz WAV via ffmpeg (cached)."""
    if not out_wav.exists():
        out_wav.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(video), "-vn", "-ac", "1",
                        "-ar", str(sr), "-c:a", "pcm_s16le", str(out_wav)], check=True)
    return out_wav


def load_audio(wav: Path) -> tuple[np.ndarray, int]:
    import soundfile as sf
    audio, sr = sf.read(str(wav), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio, sr


def iter_frames(video: Path, every_s: float, max_side: int | None = None, with_prev: bool = False):
    """Sequential decode; yields (t, frame_bgr[, prev_frame_bgr]) roughly every `every_s` seconds.
    with_prev=True also yields the frame immediately before (for optical flow)."""
    import cv2
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, int(round(every_s * fps)))
    idx, prev = 0, None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if max_side:
            h, w = frame.shape[:2]
            s = max_side / max(h, w)
            if s < 1:
                frame = cv2.resize(frame, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        if idx % step == 0 and (not with_prev or prev is not None):
            yield (idx / fps, frame, prev) if with_prev else (idx / fps, frame)
        prev = frame
        idx += 1
    cap.release()


def grab_frame(video: Path, t: float, max_side: int | None = None):
    import cv2
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
    ok, frame = cap.read()
    cap.release()
    if ok and max_side:
        h, w = frame.shape[:2]
        s = max_side / max(h, w)
        if s < 1:
            frame = cv2.resize(frame, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return frame if ok else None


def r3(x):
    return round(float(x), 3)


def write_result(out_dir: Path, name: str, model: str, params: dict, started: float, **payload) -> Path:
    """Every segmenter writes the same envelope: who/what/how long + its own raw payload.
    `boundaries` (if present) is a list of {"t", "conf", "kind", "detail"} with NO post-processing."""
    out_dir.mkdir(parents=True, exist_ok=True)
    doc = {"segmenter": name, "model": model, "params": params,
           "runtime_s": round(time.time() - started, 1), **payload}
    if "boundaries" in doc:
        doc["n_boundaries"] = len(doc["boundaries"])
    path = out_dir / f"{name}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=1)
    return path


def read_result(out_dir: Path, name: str) -> dict | None:
    p = out_dir / f"{name}.json"
    if not p.exists():
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)
