"""Camera motion: dense optical flow (OpenCV Farneback, CPU) between consecutive frames, 4 samples/s.

A global similarity model (translation tx, ty + zoom s + rotation r) is least-squares fitted to the
flow field; the residual is what the camera model can't explain (subject motion). Units: fraction of
frame width per frame. Each sample gets a label (static / pan / tilt / zoom / handheld), smoothed by
majority over ~1 s; every label change is reported as a raw boundary. Samples next to a shot cut
(from shots.json, if present) are excluded.
"""
from __future__ import annotations

import time
from collections import Counter
from pathlib import Path

import numpy as np

from .common import iter_frames, r3, read_result, write_result

NAME = "camera_motion"


def global_motion(prev_gray, gray, step: int = 8):
    import cv2
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    h, w = gray.shape
    ys, xs = np.mgrid[step // 2:h:step, step // 2:w:step]
    u, v = flow[ys, xs, 0].ravel() / w, flow[ys, xs, 1].ravel() / w
    x, y = (xs.ravel() - w / 2) / w, (ys.ravel() - h / 2) / w
    n = len(x)
    A = np.zeros((2 * n, 4))
    A[:n, 0], A[:n, 2], A[:n, 3] = 1, x, -y
    A[n:, 1], A[n:, 2], A[n:, 3] = 1, y, x
    b = np.concatenate([u, v])
    p, *_ = np.linalg.lstsq(A, b, rcond=None)
    resid = float(np.median(np.abs(A @ p - b)))
    return {"tx": float(p[0]), "ty": float(p[1]), "zoom": float(p[2]), "rot": float(p[3]), "resid": resid}


def label(m: dict, static_thr: float, zoom_thr: float) -> str:
    tmag = np.hypot(m["tx"], m["ty"])
    if abs(m["zoom"]) >= zoom_thr and abs(m["zoom"]) > tmag:
        return "zoom"
    if tmag < static_thr:
        return "static"
    return "pan" if abs(m["tx"]) >= abs(m["ty"]) else "tilt"


def smooth_labels(samples: list[dict], win: int, jitter_thr: float) -> list[str]:
    out = []
    for i in range(len(samples)):
        w = samples[max(0, i - win // 2):i + win // 2 + 1]
        lab = Counter(s["raw_label"] for s in w).most_common(1)[0][0]
        # direction keeps flipping with little net movement -> handheld shake
        tx = np.array([s["tx"] for s in w])
        ty = np.array([s["ty"] for s in w])
        if lab in ("pan", "tilt") and len(w) >= 3 and \
                np.hypot(tx.mean(), ty.mean()) < 0.5 * np.hypot(tx.std(), ty.std()) and \
                np.hypot(tx.std(), ty.std()) > jitter_thr:
            lab = "handheld"
        out.append(lab)
    return out


def run(video: Path, out_dir: Path, info: dict, every_s: float = 0.25, static_thr: float = 0.0015,
        zoom_thr: float = 0.002, jitter_thr: float = 0.002, smooth_win: int = 5, **_) -> Path:
    import cv2
    t0 = time.time()
    shots = read_result(out_dir, "shots")
    cuts = [b["t"] for b in shots["boundaries"]] if shots else []
    samples = []
    for t, frame, prev in iter_frames(video, every_s, max_side=320, with_prev=True):
        if any(abs(t - c) < 0.3 for c in cuts):
            continue
        g0 = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
        g1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        m = global_motion(g0, g1)
        m["raw_label"] = label(m, static_thr, zoom_thr)
        samples.append({"t": r3(t), **{k: (round(v, 5) if isinstance(v, float) else v) for k, v in m.items()}})
    labs = smooth_labels(samples, smooth_win, jitter_thr)
    for s, l in zip(samples, labs):
        s["label"] = l
    runs, bounds = [], []
    for s in samples:
        if runs and runs[-1]["label"] == s["label"] and s["t"] - runs[-1]["end"] <= 2 * every_s + 0.31:
            runs[-1]["end"] = s["t"]
        else:
            if runs and runs[-1]["label"] != s["label"]:
                bounds.append({"t": s["t"], "kind": "camera_change", "from": runs[-1]["label"], "to": s["label"],
                               "conf": 1.0, "prev_run_s": r3(runs[-1]["end"] - runs[-1]["start"])})
            runs.append({"label": s["label"], "start": s["t"], "end": s["t"]})
    share = Counter(labs)
    return write_result(out_dir, NAME, "OpenCV Farneback optical flow + global similarity fit",
                        {"every_s": every_s, "static_thr": static_thr, "zoom_thr": zoom_thr,
                         "jitter_thr": jitter_thr, "smooth_win": smooth_win, "units": "fraction of width per frame"},
                        t0, stats={"n_samples": len(samples),
                                   "label_share": {k: r3(v / max(len(labs), 1)) for k, v in share.items()}},
                        boundaries=bounds, runs=runs, samples=samples)
