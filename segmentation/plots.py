"""Diagnostic timelines (matplotlib, saved as PNG)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

COLORS = {"ref": "#111111", "M1_pyscenedetect": "#9a6700", "M2_clip": "#2563eb", "M3_clap": "#16a34a",
          "M4_sbert": "#9333ea", "M5_fusion_mean": "#dc2626", "M5_fusion_max": "#ea580c"}


def _mmss(x, _pos=None):
    return f"{int(x // 60)}:{int(x % 60):02d}"


def boundary_timeline(path, title, duration, ref, preds: dict, labels: dict, curve=None, grid=None):
    """Plot 1: one row per method (ticks at predicted boundaries), reference as vertical lines."""
    rows = list(preds)
    fig, ax = plt.subplots(figsize=(12, 1.0 + 0.55 * (len(rows) + (1 if curve is not None else 0))))
    for r in ref:
        ax.axvline(r, color=COLORS["ref"], lw=1.2, alpha=0.6, zorder=0)
    for i, m in enumerate(rows):
        y = len(rows) - i
        ax.scatter(preds[m], [y] * len(preds[m]), marker="|", s=320, lw=2, color=COLORS.get(m, "#555"))
    ticks = [len(rows) - i for i in range(len(rows))]
    names = [labels.get(m, m) for m in rows]
    if curve is not None:
        c = np.where(np.isfinite(curve), curve, np.nan)
        lo, hi = np.nanmin(c), np.nanmax(c)
        ax.plot(grid, 0.1 + 0.8 * (c - lo) / ((hi - lo) or 1), color=COLORS["M5_fusion_mean"], lw=1, alpha=0.8)
        ticks.append(0.5)
        names.append("fused score")
    ax.set_yticks(ticks, names)
    ax.set_xlim(0, duration)
    ax.set_ylim(0, len(rows) + 0.8)
    ax.xaxis.set_major_formatter(_mmss)
    ax.set_xlabel("time (m:ss) - vertical lines = human reference boundaries")
    ax.set_title(title, loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def score_panels(path, title, duration, ref, grid, curves: dict, labels: dict, fused, fused_pred, colors=None):
    """Plot 2: per-modality normalised change scores + fused score + predicted boundaries."""
    names = list(curves) + ["fused"]
    fig, axes = plt.subplots(len(names), 1, figsize=(12, 1.6 * len(names)), sharex=True)
    for ax, n in zip(axes, names):
        y = fused if n == "fused" else curves[n]
        for r in ref:
            ax.axvline(r, color=COLORS["ref"], lw=1.2, alpha=0.55)
        ax.plot(grid, y, lw=1, color=(colors or {}).get(n, COLORS["M5_fusion_mean"] if n == "fused" else "#444"))
        if not np.isfinite(y).any():
            ax.text(0.5, 0.5, "no evidence", transform=ax.transAxes, ha="center", va="center", color="#888")
        if n == "fused":
            for p in fused_pred:
                ax.axvline(p, color=COLORS["M5_fusion_mean"], lw=1.5, ls="--")
        ax.set_ylabel(labels.get(n, n), rotation=0, ha="right", va="center", fontsize=9)
        ax.grid(alpha=0.2)
    axes[-1].xaxis.set_major_formatter(_mmss)
    axes[-1].set_xlim(0, duration)
    axes[-1].set_xlabel("time (m:ss) - black lines = human reference boundaries, red dashed = fused predictions")
    axes[0].set_title(title, loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
