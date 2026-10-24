"""Late fusion of per-modality change scores (no training, no embedding concatenation).

1. each modality's curve is normalised on its own (per video),
2. curves are combined point by point on the common grid, counting only modalities that have
   evidence at that time:
       mean:  B(t) = sum_m w_m a_m(t) d_m(t) / sum_m w_m a_m(t)
       max:   B(t) = max_m over available d_m(t)
   a_m(t) = 1 if d_m(t) is finite else 0. Missing modalities never contribute a zero.
"""
from __future__ import annotations

import numpy as np


def normalise(d: np.ndarray, method: str = "robust_z", clip: float | None = 5.0) -> np.ndarray:
    """Per-video normalisation. z-type scores are clipped to +/-clip so one artefact (e.g. a near-flat
    curve with a tiny spread) cannot dominate fusion or peak picking."""
    out = np.full_like(d, np.nan, dtype=float)
    ok = np.isfinite(d)
    if ok.sum() < 2:
        return out
    x = d[ok]
    if method == "robust_z":
        med = np.median(x)
        mad = 1.4826 * np.median(np.abs(x - med))
        scale = mad if mad > 1e-9 else (x.std() or 1.0)
        out[ok] = (x - med) / scale
    elif method == "z":
        out[ok] = (x - x.mean()) / (x.std() or 1.0)
    elif method == "minmax":
        rng = x.max() - x.min()
        out[ok] = (x - x.min()) / (rng or 1.0)
    else:
        raise ValueError(method)
    if clip is not None and method in ("robust_z", "z"):
        out[ok] = np.clip(out[ok], -clip, clip)
    return out


def fuse(curves: dict, weights: dict, mode: str = "mean", norm: str = "robust_z",
         clip: float | None = 5.0) -> tuple[np.ndarray, np.ndarray]:
    """curves: modality -> raw change score (NaN = unavailable). Returns (fused, n_available)."""
    names = list(curves)
    D = np.vstack([normalise(curves[m], norm, clip) for m in names])
    A = np.isfinite(D)
    n_avail = A.sum(0)
    if mode == "mean":
        w = np.array([weights.get(m, 1.0) for m in names])[:, None]
        num = np.where(A, w * D, 0.0).sum(0)
        den = (w * A).sum(0)
        fused = np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)
    elif mode == "max":
        fused = np.where(n_avail > 0, np.nanmax(np.where(A, D, -np.inf), axis=0), np.nan)
    else:
        raise ValueError(mode)
    return fused, n_avail
