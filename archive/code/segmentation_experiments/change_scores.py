"""Per-modality change scores on a common temporal grid.

For a grid time t, compare the context just before t with the context just after t:
    d(t) = 1 - cos( unit(mean_w(emb in [t - C, t))), unit(mean_w(emb in [t, t + C))) )
where mean_w is an optional weighted mean (transcript chunks are weighted by word count).
A side without enough evidence gives NaN = "no evidence", never 0:
  - frames / audio windows: fewer than min_coverage x the samples the context should hold;
  - transcript: fewer than min_words words on that side.

Parameters are independent: the embedding chunk length (set at encoding), the context C (here),
the grid step and the minimum segment duration (boundary detection).
"""
from __future__ import annotations

import numpy as np


def make_grid(duration: float, step: float) -> np.ndarray:
    return np.arange(0.0, duration + 1e-6, step)


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def embedding_change(times: np.ndarray, emb: np.ndarray, grid: np.ndarray, context_s: float,
                     sample_step_s: float | None = None, min_coverage: float = 0.5,
                     weights: np.ndarray | None = None, min_weight: float | None = None) -> np.ndarray:
    """Count-based availability (sample_step_s + min_coverage) or weight-based (weights + min_weight)."""
    out = np.full(len(grid), np.nan)
    if len(times) == 0:
        return out
    order = np.argsort(times)
    times, emb = times[order], emb[order]
    w = None if weights is None else np.asarray(weights, float)[order]
    if w is None:
        need = max(1, int(np.floor(min_coverage * context_s / sample_step_s)))
    for i, t in enumerate(grid):
        l0, l1 = np.searchsorted(times, [t - context_s, t])
        r0, r1 = np.searchsorted(times, [t, t + context_s])
        if w is None:
            if l1 - l0 < need or r1 - r0 < need:
                continue
            a, b = emb[l0:l1].mean(0), emb[r0:r1].mean(0)
        else:
            wl, wr = w[l0:l1], w[r0:r1]
            if wl.sum() < (min_weight or 1) or wr.sum() < (min_weight or 1):
                continue
            a, b = wl @ emb[l0:l1] / wl.sum(), wr @ emb[r0:r1] / wr.sum()
        out[i] = 1.0 - float(_unit(a) @ _unit(b))
    return out


def text_change(arrays: dict, grid: np.ndarray, context_s: float, min_words: int) -> np.ndarray:
    """Transcript chunks (centre times, embeddings, word counts) aggregated over the context window."""
    return embedding_change(arrays["times"], arrays["emb"], grid, context_s,
                            weights=arrays["n_words"], min_weight=min_words)


def modality_curves(caches: dict, grid: np.ndarray, cfg, contexts: dict | None = None) -> dict:
    """caches: modality -> arrays from the embedding cache. contexts overrides per-modality context."""
    ctx = {"visual": cfg.visual_context_s, "audio": cfg.audio_context_s, "text": cfg.text_context_s}
    ctx.update(contexts or {})
    out = {}
    if "visual" in caches:
        out["visual"] = embedding_change(caches["visual"]["times"], caches["visual"]["emb"], grid, ctx["visual"],
                                         1.0 / cfg.visual_fps, cfg.min_context_coverage)
    if "audio" in caches:
        out["audio"] = embedding_change(caches["audio"]["times"], caches["audio"]["emb"], grid, ctx["audio"],
                                        cfg.audio_hop_s, cfg.min_context_coverage)
    if "text" in caches:
        out["text"] = text_change(caches["text"], grid, ctx["text"], cfg.text_min_words)
    return {k: v for k, v in out.items() if np.isfinite(v).any()}
