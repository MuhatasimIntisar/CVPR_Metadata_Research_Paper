"""Boundary evaluation: one-to-one matching within a tolerance (Hungarian assignment, so one
prediction can never match two references), per-video and dataset-level micro-averaged P/R/F1."""
from __future__ import annotations

import numpy as np


def match(pred: list[float], ref: list[float], tol: float) -> list[tuple[int, int, float]]:
    """Maximum number of (pred, ref) pairs with |p - r| <= tol, ties broken by smallest total error."""
    if not pred or not ref:
        return []
    from scipy.optimize import linear_sum_assignment
    P, R = np.asarray(pred, float), np.asarray(ref, float)
    dist = np.abs(P[:, None] - R[None, :])
    big = 1e6
    cost = np.where(dist <= tol, dist, big)
    rows, cols = linear_sum_assignment(cost)
    return [(int(i), int(j), float(dist[i, j])) for i, j in zip(rows, cols) if dist[i, j] <= tol]


def prf(tp: int, n_pred: int, n_ref: int) -> dict:
    p = tp / n_pred if n_pred else 0.0
    r = tp / n_ref if n_ref else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"tp": tp, "n_pred": n_pred, "n_ref": n_ref, "precision": round(p, 4), "recall": round(r, 4),
            "f1": round(f, 4)}


def score_video(pred: list[float], ref: list[float], tol: float) -> dict:
    m = match(pred, ref, tol)
    out = prf(len(m), len(pred), len(ref))
    out["mean_abs_offset_s"] = round(float(np.mean([d for *_, d in m])), 2) if m else None
    return out


def max_matches_1d(pred, ref, tol) -> int:
    """Maximum one-to-one matches within tol on a line (greedy in time order is optimal here)."""
    pred, ref = sorted(pred), sorted(ref)
    i = j = n = 0
    while i < len(pred) and j < len(ref):
        if abs(pred[i] - ref[j]) <= tol:
            n, i, j = n + 1, i + 1, j + 1
        elif pred[i] < ref[j]:
            i += 1
        else:
            j += 1
    return n


def chance_tp(k: int, ref: list[float], tol: float, lo: float, hi: float, trials: int, rng) -> float:
    """Expected matches for k boundaries drawn uniformly in [lo, hi] (same count as the method)."""
    if k == 0 or not ref or hi <= lo:
        return 0.0
    return float(np.mean([max_matches_1d(rng.uniform(lo, hi, k), ref, tol) for _ in range(trials)]))


def micro(rows: list[dict]) -> dict:
    return prf(sum(r["tp"] for r in rows), sum(r["n_pred"] for r in rows), sum(r["n_ref"] for r in rows))
