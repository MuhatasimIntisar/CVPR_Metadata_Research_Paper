"""Shared tools for the Experiment A analysis and Experiment B (no model inference; cached embeddings only).

Key fact used throughout: scipy.signal.find_peaks applies `distance` before `prominence`, and a peak's
prominence does not depend on which other peaks are kept. So for a fixed curve and minimum segment,
detect(prominence = p) returns exactly the candidates (peaks found with that distance) whose prominence
is >= p. Sweeping p therefore traces the method's full precision-recall curve, and ranking candidates by
prominence measures how well the curve *ranks* true boundaries independently of how many are selected.
"""
from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np

from .boundary_detection import smooth
from .change_scores import make_grid, modality_curves
from .config import Config, discover, reference_boundaries
from .evaluation import max_matches_1d
from .fusion import fuse, normalise
from .run_experiment import load_caches

MOD_OF = {"clip": "visual", "clap": "audio", "sbert": "text"}


class Data:
    """Cached embeddings, durations and references for every video in a data folder."""

    def __init__(self, data_dir: Path, base: Config | None = None):
        self.dir = Path(data_dir)
        self.base = base or Config()
        self.dirs = {k: self.dir / k for k in ("cache", "predictions", "evaluation", "plots")}
        self.val = {r["video_id"]: r for r in json.loads((self.dir / "evaluation" / "validation.json").read_text(encoding="utf-8"))}
        self.items = {it.video_id: it for it in discover(self.dir)}
        self.refs = {v: reference_boundaries(it) for v, it in self.items.items()}
        self._caches = {}
        self._curves = {}

    def videos(self, dataset):
        return [v for v, it in self.items.items() if it.dataset == dataset]

    def dataset(self, v):
        return self.items[v].dataset

    def dur(self, v):
        return self.val[v]["analysis_duration_s"]

    def caches(self, v):
        if v not in self._caches:
            self._caches[v] = load_caches(self.items[v], self.dirs)[0]
        return self._caches[v]

    def raw_curve(self, v, method: str, ctx: float):
        """method: clip | clap | sbert | fusion_mean | fusion_max (fusion uses every available modality)."""
        key = (v, method, ctx)
        if key not in self._curves:
            grid = make_grid(self.dur(v), self.base.grid_s)
            c = self.caches(v)
            if method in MOD_OF:
                m = MOD_OF[method]
                curves = modality_curves({m: c[m]}, grid, self.base, {m: ctx}) if m in c else {}
                score = curves.get(m)
            else:
                curves = modality_curves(c, grid, self.base, {m: ctx for m in c})
                score = fuse(curves, self.base.fusion_weights, method.split("_")[1], self.base.normalise,
                             self.base.score_clip)[0] if curves else None
            self._curves[key] = (grid, score)
        return self._curves[key]


def detection_curve(score, smooth_k: int, base: Config):
    """Exactly what boundary_detection.detect() picks peaks from."""
    return smooth(normalise(score, base.normalise, base.score_clip), smooth_k)


def candidates(grid, s, min_seg: float, dur: float, grid_s: float):
    """All peaks under the min-segment spacing and edge rule, with their prominences."""
    from scipy.signal import find_peaks
    if s is None or not np.isfinite(s).any():
        return np.array([]), np.array([])
    filled = np.where(np.isfinite(s), s, np.nanmin(s))
    peaks, props = find_peaks(filled, distance=max(1, math.ceil(min_seg / grid_s)), prominence=0)
    keep = [i for i, p in enumerate(peaks) if np.isfinite(s[p]) and min_seg <= grid[p] <= dur - min_seg]
    return grid[peaks[keep]], props["prominences"][keep]


def select(times, proms, prominence):
    return sorted(float(t) for t, p in zip(times, proms) if p >= prominence)


def pr_points(per_video: list, tol: float):
    """per_video: [(cand_times, cand_proms, refs)]. Micro P/R at every distinct prominence threshold
    (descending), plus the all-candidates point. Returns list of dicts and average precision."""
    thr = sorted({float(p) for _, pr, _ in per_video for p in pr}, reverse=True)
    n_ref = sum(len(r) for *_, r in per_video)
    pts = []
    for t in thr:
        tp = npred = 0
        for times, proms, refs in per_video:
            sel = select(times, proms, t)
            npred += len(sel)
            tp += max_matches_1d(sel, refs, tol)
        p = tp / npred if npred else 0.0
        r = tp / n_ref if n_ref else 0.0
        pts.append({"prominence": t, "tp": tp, "n_pred": npred, "n_ref": n_ref, "precision": p, "recall": r,
                    "f1": 2 * p * r / (p + r) if p + r else 0.0})
    ap, prev_r = 0.0, 0.0
    for q in pts:  # step-wise AP over the ranked candidate list
        if q["recall"] > prev_r:
            ap += (q["recall"] - prev_r) * q["precision"]
            prev_r = q["recall"]
    return pts, ap


def max_reachable(refs, allowed_times, min_seg, tol, grid_s):
    """Upper bound on matched references for ANY prediction set the detector could output:
    predictions on allowed grid times (finite score, inside [min_seg, dur - min_seg]), spaced at
    least ceil(min_seg / grid_s) grid steps apart, one-to-one within tol. Greedy earliest placement
    is optimal for this interval problem."""
    allowed = np.asarray(sorted(allowed_times))
    gap = max(1, math.ceil(min_seg / grid_s)) * grid_s - 1e-6
    last, n = -np.inf, 0
    for r in sorted(refs):
        ok = allowed[(allowed >= r - tol - 1e-6) & (allowed <= r + tol + 1e-6) & (allowed >= last + gap)]
        if len(ok):
            last, n = ok[0], n + 1
    return n


def eligibility(data: Data, v, method, ctx, min_seg, tol):
    """Per-reference reasons a boundary cannot be predicted, plus the joint upper bound."""
    grid, score = data.raw_curve(v, method, ctx)
    dur = data.dur(v)
    finite = np.isfinite(score) if score is not None else np.zeros(len(grid), bool)
    in_window = (grid >= min_seg) & (grid <= dur - min_seg)
    allowed = grid[finite & in_window]
    rows = []
    for r in data.refs[v]:
        near = np.abs(grid - r) <= tol
        rows.append({"ref_s": r,
                     "naive_reachable": bool(min_seg <= r <= dur - min_seg),
                     "edge_ok": bool((near & in_window).any()),
                     "evidence_ok": bool((near & finite).any()),
                     "single_ok": bool((near & finite & in_window).any())})
    return rows, max_reachable(data.refs[v], allowed, min_seg, tol, data.base.grid_s), (
        float(grid[finite][0]) if finite.any() else None, float(grid[finite][-1]) if finite.any() else None)


def with_params(base: Config, **kw) -> Config:
    return replace(base, **kw)
