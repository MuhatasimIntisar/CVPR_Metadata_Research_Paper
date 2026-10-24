"""Experiment A: temporal context and minimum segment duration.

    python -m segmentation.experiment_a                      # default grids below, both datasets
    python -m segmentation.experiment_a --datasets ami --ami-context 30 60 90 --ami-min-seg 60 90 120

Context window C is applied to every modality (CLIP frames, CLAP windows, transcript chunks): at each
grid time the cached embeddings in [t - C, t) and [t, t + C) are aggregated (mean; word-count weighted
for transcript chunks), L2-normalised and compared by cosine distance. No CLIP / CLAP / SBERT inference
is run; the embedding chunk length, grid step, context and minimum segment are independent.

Outputs (evaluation/experiment_A/):
    predictions.jsonl   boundaries per dataset x config x video x method, with parameters and runtime
    per_video.csv       P / R / F1 per video, tolerance, config, method (+ chance F1, reachable refs)
    summary.csv         micro-averaged per dataset x config x method x tolerance (all / dev / test)
    summary.md          readable tables: F1 over context x min segment, per method
    heatmap_<dataset>.png
    params.json         grids and the full base configuration

Runtime = change scores + fusion + peak picking for that method (cached neural inference excluded).
Reference boundaries are used only for scoring. The number of predictions is never set from them.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from segmentation.boundary_detection import detect  # noqa: E402
from segmentation.change_scores import make_grid, modality_curves  # noqa: E402
from segmentation.config import DEFAULT_DATA_DIR, Config, discover, out_dirs, reference_boundaries  # noqa: E402
from segmentation.evaluation import chance_tp, micro, score_video  # noqa: E402
from segmentation.fusion import fuse  # noqa: E402
from segmentation.methods import METHODS  # noqa: E402
from segmentation.run_experiment import load_caches  # noqa: E402

DEFAULT_GRIDS = {"moviechat": {"context": [5.0, 10.0, 20.0], "min_seg": [15.0, 30.0, 45.0]},
                 "ami": {"context": [30.0, 60.0, 90.0], "min_seg": [60.0, 90.0, 120.0]}}
SWEEP_METHODS = [m for m in METHODS if m.id != "M1_pyscenedetect"]


def config_id(ctx, ms):
    return f"ctx{ctx:g}_min{ms:g}"


def run(data_dir: Path, base: Config, grids: dict, datasets: list, trials: int):
    dirs = out_dirs(data_dir)
    out = dirs["evaluation"] / "experiment_A"
    out.mkdir(exist_ok=True)
    vpath = dirs["evaluation"] / "validation.json"
    if not vpath.exists():
        sys.exit("Run `python -m segmentation.run_experiment --steps validate` first.")
    val = {r["video_id"]: r for r in json.loads(vpath.read_text(encoding="utf-8"))}
    rng = np.random.default_rng(0)
    preds, rows = [], []

    for ds in datasets:
        items = [it for it in discover(data_dir) if it.dataset == ds]
        g = grids[ds]
        for it in items:
            dur = val[it.video_id]["analysis_duration_s"]
            ref = reference_boundaries(it)
            caches, metas = load_caches(it, dirs)
            if "text" not in caches and metas == {}:
                print(f"[warn] {it.video_id}: no cached embeddings; run run_experiment --steps encode")
            grid = make_grid(dur, base.grid_s)
            for ctx in g["context"]:
                # one curve per modality per context, timed separately
                curves, ctimes = {}, {}
                for mod in caches:
                    t0 = time.perf_counter()
                    c = modality_curves({mod: caches[mod]}, grid, base, {mod: ctx})
                    ctimes[mod] = time.perf_counter() - t0
                    curves.update(c)
                for ms in g["min_seg"]:
                    cfg = replace(base, min_segment_s=ms, visual_context_s=ctx, audio_context_s=ctx, text_context_s=ctx)
                    cid = config_id(ctx, ms)
                    reachable = sum(ms <= r <= dur - ms for r in ref)
                    for meth in SWEEP_METHODS:
                        avail = [m for m in meth.modalities if m in curves]
                        missing = [m for m in meth.modalities if m not in curves]
                        rec = {"dataset": ds, "config": cid, "context_s": ctx, "min_segment_s": ms,
                               "video_id": it.video_id, "method": meth.id, "label": meth.label, "group": meth.group}
                        if (meth.require_all and missing) or not avail:
                            preds.append({**rec, "status": "not_available", "missing_modalities": missing})
                            continue
                        t0 = time.perf_counter()
                        if meth.fusion is None:
                            score = curves[avail[0]]
                        else:
                            score, _ = fuse({k: curves[k] for k in avail}, cfg.fusion_weights, meth.fusion,
                                            cfg.normalise, cfg.score_clip)
                        bounds, _ = detect(grid, score, cfg, dur)
                        rt = time.perf_counter() - t0 + sum(ctimes[k] for k in avail)
                        preds.append({**rec, "status": "ok", "modalities_used": avail, "missing_modalities": missing,
                                      "boundaries_seconds": [round(b, 2) for b in bounds], "runtime_s": round(rt, 4),
                                      "params": {"grid_s": cfg.grid_s, "context_s": ctx, "min_segment_s": ms,
                                                 "text_chunk_s": cfg.text_chunk_s, "text_chunk_hop_s": cfg.text_chunk_hop_s,
                                                 "visual_fps": cfg.visual_fps, "audio_window_s": cfg.audio_window_s,
                                                 "audio_hop_s": cfg.audio_hop_s, "peak_prominence": cfg.peak_prominence,
                                                 "smooth_window": cfg.smooth_window, "normalise": cfg.normalise,
                                                 "score_clip": cfg.score_clip, "fusion_weights": cfg.fusion_weights}})
                        for tol in base.tolerances[ds]:
                            s = score_video(bounds, ref, tol)
                            ctp = chance_tp(len(bounds), ref, tol, ms, dur - ms, trials, rng)
                            rows.append({**rec, "tolerance_s": tol, **s, "chance_tp": round(ctp, 3),
                                         "chance_f1": prf_f1(ctp, len(bounds), len(ref)),
                                         "reachable_refs": reachable, "runtime_s": round(rt, 4),
                                         "split": "dev" if it.video_id in base.dev_videos else "test"})
            print(f"[A] {ds:9} {it.video_id:8} {len(g['context']) * len(g['min_seg'])} configs x "
                  f"{len(SWEEP_METHODS)} methods  (modalities: {', '.join(sorted(caches)) or 'none'})", flush=True)

    with open(out / "predictions.jsonl", "w", encoding="utf-8") as f:
        for p in preds:
            f.write(json.dumps(p) + "\n")
    write_csv(out / "per_video.csv", rows)
    summary = summarise(rows, base)
    write_csv(out / "summary.csv", summary)
    (out / "params.json").write_text(json.dumps({"grids": grids, "datasets": datasets, "chance_trials": trials,
                                                 "base_config": base.to_dict()}, indent=1), encoding="utf-8")
    md = markdown(summary, grids, datasets, base)
    (out / "summary.md").write_text(md, encoding="utf-8")
    heatmaps(summary, grids, datasets, base, out)
    print("\n" + md)
    print(f"\nWrote {out}")


def prf_f1(tp, n_pred, n_ref):
    p = tp / n_pred if n_pred else 0.0
    r = tp / n_ref if n_ref else 0.0
    return round(2 * p * r / (p + r), 4) if p + r else 0.0


def summarise(rows, base):
    out = []
    keys = sorted({(r["dataset"], r["config"], r["context_s"], r["min_segment_s"], r["method"], r["label"], r["group"],
                    r["tolerance_s"]) for r in rows})
    splits = [("all", None)] + ([("dev", "dev"), ("test", "test")] if base.dev_videos else [])
    for ds, cid, ctx, ms, mid, label, group, tol in keys:
        sel = [r for r in rows if (r["dataset"], r["config"], r["method"], r["tolerance_s"]) == (ds, cid, mid, tol)]
        for split, want in splits:
            s = [r for r in sel if want is None or r["split"] == want]
            if not s:
                continue
            m = micro(s)
            ctp = sum(r["chance_tp"] for r in s)
            out.append({"dataset": ds, "split": split, "config": cid, "context_s": ctx, "min_segment_s": ms,
                        "method": mid, "label": label, "group": group, "tolerance_s": tol, "n_videos": len(s), **m,
                        "chance_f1": prf_f1(ctp, m["n_pred"], m["n_ref"]),
                        "f1_minus_chance": round(m["f1"] - prf_f1(ctp, m["n_pred"], m["n_ref"]), 4),
                        "reachable_refs": sum(r["reachable_refs"] for r in s),
                        "runtime_s_total": round(sum(r["runtime_s"] for r in s), 3)})
    return out


def markdown(summary, grids, datasets, base):
    L = []
    for ds in datasets:
        g = grids[ds]
        L.append(f"## {'MovieChat-SSeg (scene boundaries)' if ds == 'moviechat' else 'AMI (topic boundaries, oracle transcript)'}\n")
        for tol in base.tolerances[ds]:
            L.append(f"### F1 @±{tol:g}s  (cell: F1 [chance] - predictions / references)\n")
            L.append("| Method | context \\ min seg | " + " | ".join(f"{m:g}s" for m in g["min_seg"]) + " |")
            L.append("|---|---|" + "---|" * len(g["min_seg"]))
            for meth in SWEEP_METHODS:
                first = True
                for ctx in g["context"]:
                    cells = []
                    for ms in g["min_seg"]:
                        r = next((s for s in summary if s["dataset"] == ds and s["split"] == "all" and s["method"] == meth.id
                                  and s["context_s"] == ctx and s["min_segment_s"] == ms and s["tolerance_s"] == tol), None)
                        cells.append("N/A" if r is None else f"**{r['f1']:.2f}** [{r['chance_f1']:.2f}] - {r['n_pred']}/{r['n_ref']}")
                    if all(c == "N/A" for c in cells):
                        if first:
                            L.append(f"| {meth.label} | - | " + " | ".join(cells) + " |")
                        break
                    L.append(f"| {meth.label if first else ''} | {ctx:g}s | " + " | ".join(cells) + " |")
                    first = False
            L.append("")
        reach = [s for s in summary if s["dataset"] == ds and s["split"] == "all" and s["method"] == "M2_clip"
                 and s["tolerance_s"] == base.tolerances[ds][0] and s["context_s"] == g["context"][0]]
        if reach:
            L.append("Reference boundaries reachable under each minimum segment (a boundary closer than the minimum "
                     "to the start/end cannot be predicted): " +
                     ", ".join(f"{r['min_segment_s']:g}s: {r['reachable_refs']}/{r['n_ref']}" for r in reach) + "\n")
    return "\n".join(L)


def heatmaps(summary, grids, datasets, base, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for ds in datasets:
        g, tol = grids[ds], max(base.tolerances[ds])
        meths = [m for m in SWEEP_METHODS if any(s["dataset"] == ds and s["method"] == m.id for s in summary)]
        if not meths:
            continue
        fig, axes = plt.subplots(1, len(meths), figsize=(2.6 * len(meths), 2.9), squeeze=False)
        for ax, meth in zip(axes[0], meths):
            M = np.full((len(g["context"]), len(g["min_seg"])), np.nan)
            for i, ctx in enumerate(g["context"]):
                for j, ms in enumerate(g["min_seg"]):
                    r = next((s for s in summary if s["dataset"] == ds and s["split"] == "all" and s["method"] == meth.id
                              and s["context_s"] == ctx and s["min_segment_s"] == ms and s["tolerance_s"] == tol), None)
                    if r:
                        M[i, j] = r["f1"]
            ax.imshow(M, vmin=0, vmax=1, cmap="viridis")
            for i in range(M.shape[0]):
                for j in range(M.shape[1]):
                    if np.isfinite(M[i, j]):
                        ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                                color="white" if M[i, j] < 0.6 else "black", fontsize=9)
            ax.set_xticks(range(len(g["min_seg"])), [f"{m:g}" for m in g["min_seg"]])
            ax.set_yticks(range(len(g["context"])), [f"{c:g}" for c in g["context"]])
            ax.set_xlabel("min segment (s)", fontsize=8)
            ax.set_title(meth.label.replace("Ablation: ", "").replace("Fusion: CLIP+CLAP+SBERT", "Fusion"), fontsize=8)
        axes[0][0].set_ylabel("context (s)", fontsize=8)
        fig.suptitle(f"{ds}: F1 @±{tol:g}s", fontsize=10)
        fig.tight_layout()
        fig.savefig(out / f"heatmap_{ds}.png", dpi=150)
        plt.close(fig)


def write_csv(path, rows):
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in r.items()})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    ap.add_argument("--config", type=Path, help="JSON overriding base Config fields")
    ap.add_argument("--datasets", nargs="+", default=["moviechat", "ami"], choices=["moviechat", "ami"])
    ap.add_argument("--moviechat-context", nargs="+", type=float, default=DEFAULT_GRIDS["moviechat"]["context"])
    ap.add_argument("--moviechat-min-seg", nargs="+", type=float, default=DEFAULT_GRIDS["moviechat"]["min_seg"])
    ap.add_argument("--ami-context", nargs="+", type=float, default=DEFAULT_GRIDS["ami"]["context"])
    ap.add_argument("--ami-min-seg", nargs="+", type=float, default=DEFAULT_GRIDS["ami"]["min_seg"])
    ap.add_argument("--chance-trials", type=int, default=500, help="random-boundary draws for the chance F1")
    args = ap.parse_args()
    grids = {"moviechat": {"context": args.moviechat_context, "min_seg": args.moviechat_min_seg},
             "ami": {"context": args.ami_context, "min_seg": args.ami_min_seg}}
    run(args.data_dir, Config.from_file(args.config), grids, args.datasets, args.chance_trials)


if __name__ == "__main__":
    main()
