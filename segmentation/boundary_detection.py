"""One boundary-selection procedure shared by M2-M5: normalise -> smooth -> scipy find_peaks ->
minimum-segment constraint. Uses no reference information (not even the number of boundaries)."""
from __future__ import annotations

import math

import numpy as np

from .fusion import normalise


def smooth(x: np.ndarray, k: int) -> np.ndarray:
    if k <= 1:
        return x
    ok = np.isfinite(x)
    v = np.where(ok, x, 0.0)
    kern = np.ones(k)
    num = np.convolve(v, kern, mode="same")
    den = np.convolve(ok.astype(float), kern, mode="same")
    out = np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)
    out[~ok] = np.nan  # keep "no evidence" points as no evidence
    return out


def detect(grid: np.ndarray, score: np.ndarray, cfg, duration: float) -> tuple[list[float], np.ndarray]:
    """Return (boundary times, the normalised+smoothed curve the peaks were picked from)."""
    from scipy.signal import find_peaks
    s = smooth(normalise(score, cfg.normalise, cfg.score_clip), cfg.smooth_window)
    if not np.isfinite(s).any():
        return [], s
    filled = np.where(np.isfinite(s), s, np.nanmin(s))
    distance = max(1, math.ceil(cfg.min_segment_s / cfg.grid_s))
    peaks, _ = find_peaks(filled, prominence=cfg.peak_prominence, height=cfg.peak_height, distance=distance)
    out = [float(grid[p]) for p in peaks
           if np.isfinite(s[p]) and cfg.min_segment_s <= grid[p] <= duration - cfg.min_segment_s]
    return out, s
