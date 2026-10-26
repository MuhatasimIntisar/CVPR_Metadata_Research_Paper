"""Frame sampling at exact timestamps (OpenCV seek), resized for the VLM."""
from __future__ import annotations

from pathlib import Path

import numpy as np


def spread(a: float, b: float, n: int, edge: float = 0.5) -> list[float]:
    """n timestamps evenly inside [a+edge, b-edge] (at least the midpoint)."""
    lo, hi = a + min(edge, (b - a) / 4), b - min(edge, (b - a) / 4)
    if n <= 1 or hi <= lo:
        return [round((a + b) / 2, 2)]
    return [round(float(x), 2) for x in np.linspace(lo, hi, n)]


def grab(video: Path, times: list[float], max_side: int):
    import cv2
    from PIL import Image
    cap = cv2.VideoCapture(str(video))
    out = []
    for t in sorted(times):
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, fr = cap.read()
        if not ok:
            continue
        h, w = fr.shape[:2]
        s = max_side / max(h, w)
        if s < 1:
            fr = cv2.resize(fr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        out.append((t, Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))))
    cap.release()
    return out
