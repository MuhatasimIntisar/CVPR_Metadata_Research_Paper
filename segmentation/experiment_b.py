"""Experiment B: boundary selection (peak prominence x temporal smoothing), precision-recall curves, and
leave-one-out generalisation. Cached embeddings only; the models are not run.

    python -m segmentation.experiment_b
    python -m segmentation.experiment_b --prominence 0.5 1 1.5 2 3 --smoothing 1 3 5

Starting configurations (diagnostic, chosen in the exploratory Experiment A):
    MovieChat: CLIP, context 20 s, minimum segment 30 s     (evaluated at ±2 / ±5 s; selection uses ±5 s)
    AMI:       Sentence-BERT, context 60 s, minimum segment 90 s   (±5 / ±10 s; selection uses ±10 s)
Fusion (mean of all available modalities) is run alongside as a comparison.

Held-out evaluation: leave one video (MovieChat) / meeting (AMI) out. Context, minimum segment,
smoothing and prominence are chosen on the training fold only (max micro-F1 at the selection
tolerance; ties -> fewer training predictions, then higher prominence, lower smoothing, shorter context,
shorter minimum segment), then applied unchanged to the held-out recording. Held-out predictions are
pooled into a micro-F1. Prominence is one global value per fold, never per recording, and the number of
reference boundaries is never used to set how many boundaries to predict.

Outputs (evaluation/experiment_B/; Experiment A outputs are not touched):
    experiment_b_threshold_sweep.csv    every operating point: micro and per video, with prediction counts
    experiment_b_precision_recall.png
    experiment_b_heldout.csv            per fold: selected parameters, training F1, held-out scores; + pooled rows
    experiment_b_comparison.csv / .md   original vs exploratory vs held-out F1
    experiment_b_params.json
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from segmentation.analysis_tools import Data, candidates, detection_curve, pr_points, select  # noqa: E402
from segmentation.config import DEFAULT_DATA_DIR  # noqa: E402
from segmentation.evaluation import max_matches_1d  # noqa: E402

START = {"moviechat": {"method": "clip", "context_s": 20.0, "min_segment_s": 30.0},
         "ami": {"method": "sbert", "context_s": 60.0, "min_segment_s": 90.0}}
ORIGINAL = {"moviechat": {"context_s": 8.0, "min_segment_s": 15.0},       # main run defaults (visual)
            "ami": {"context_s": 20.0, "min_segment_s": 15.0}}            # main run defaults (text)
SEL_TOL = {"moviechat": 5.0, "ami": 10.0}
TOLS = {"moviechat": [2.0, 5.0], "ami": [5.0, 10.0]}
A_GRID = {"moviechat": {"context": [5.0, 10.0, 20.0], "min_seg": [15.0, 30.0, 45.0]},
          "ami": {"context": [30.0, 60.0, 90.0], "min_seg": [60.0, 90.0, 120.0]}}
PROMINENCE = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
SMOOTHING = [1, 3, 5, 7]  # moving-average length in grid points (grid 2 s -> 2/6/10/14 s)
ORIG_PROM, ORIG_SMOOTH = 1.0, 3
LABEL = {"clip": "CLIP-only", "sbert": "Sentence-BERT", "fusion_mean": "Fusion (mean)", "clap": "CLAP-only"}


class Cands:
    """Candidate peaks (times, prominences) cached per video x method x context x min segment x smoothing."""

    def __init__(self, D: Data):
        self.D, self.c = D, {}

    def get(self, v, method, ctx, ms, k):
        key = (v, method, ctx, ms, k)
        if key not in self.c:
            g, sc = self.D.raw_curve(v, method, ctx)
            self.c[key] = (np.array([]), np.array([])) if sc is None else \
                candidates(g, detection_curve(sc, k, self.D.base), ms, self.D.dur(v), self.D.base.grid_s)
        return self.c[key]


def score(C, D, videos, method, ctx, ms, k, prom, tol):
    tp = npred = nref = 0
    per = {}
    for v in videos:
        t, p = C.get(v, method, ctx, ms, k)
        sel = select(t, p, prom)
        m = max_matches_1d(sel, D.refs[v], tol)
        per[v] = (m, len(sel), len(D.refs[v]), sel)
        tp, npred, nref = tp + m, npred + len(sel), nref + len(D.refs[v])
    return prf(tp, npred, nref), per


def prf(tp, npred, nref):
    p = tp / npred if npred else 0.0
    r = tp / nref if nref else 0.0
    return {"tp": tp, "n_pred": npred, "n_ref": nref, "precision": round(p, 4), "recall": round(r, 4),
            "f1": round(2 * p * r / (p + r), 4) if p + r else 0.0}


def chance_f1(D, per_counts: dict, tol: float, trials: int = 500) -> float:
    """per_counts: video -> (n_pred, min_seg). Expected micro-F1 of the same number of random boundaries."""
    from segmentation.evaluation import chance_tp
    rng = np.random.default_rng(0)
    tp = sum(chance_tp(k, D.refs[v], tol, ms, D.dur(v) - ms, trials, rng) for v, (k, ms) in per_counts.items())
    npred = sum(k for k, _ in per_counts.values())
    nref = sum(len(D.refs[v]) for v in per_counts)
    p, r = (tp / npred if npred else 0.0), (tp / nref if nref else 0.0)
    return round(2 * p * r / (p + r), 4) if p + r else 0.0


def available(C, D, videos, method, ctx):
    return [v for v in videos if D.raw_curve(v, method, ctx)[1] is not None]


def write_csv(path, rows):
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in r.items()})


# ---------------------------------------------------------------- 1. threshold sweep
def threshold_sweep(D, C, proms, smooths, methods):
    rows = []
    for ds, st in START.items():
        for method in methods[ds]:
            vids = available(C, D, D.videos(ds), method, st["context_s"])
            for k in smooths:
                for prom in proms:
                    for tol in TOLS[ds]:
                        m, per = score(C, D, vids, method, st["context_s"], st["min_segment_s"], k, prom, tol)
                        base = {"dataset": ds, "method": method, "context_s": st["context_s"],
                                "min_segment_s": st["min_segment_s"], "smoothing": k, "prominence": prom,
                                "tolerance_s": tol, "is_original_threshold": k == ORIG_SMOOTH and prom == ORIG_PROM}
                        rows.append({**base, "video_id": "ALL (micro)", **m})
                        for v, (tp, n, nr, sel) in per.items():
                            rows.append({**base, "video_id": v, **prf(tp, n, nr), "boundaries_seconds": [round(x, 2) for x in sel]})
    return rows


# ---------------------------------------------------------------- 2. precision-recall figure
def pr_figure(D, C, sweep, smooths, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, ds in zip(axes, ("moviechat", "ami")):
        st, tol = START[ds], SEL_TOL[ds]
        method = st["method"]
        vids = D.videos(ds)
        for i, k in enumerate(smooths):
            col = plt.get_cmap("viridis")(i / max(1, len(smooths) - 1))
            pv = [(*C.get(v, method, st["context_s"], st["min_segment_s"], k), D.refs[v]) for v in vids]
            pts, ap = pr_points(pv, tol)
            ax.plot([q["recall"] for q in pts], [q["precision"] for q in pts], "-", color=col, lw=1.2, alpha=0.8,
                    label=f"smoothing {k} ({2 * k}s), AP {ap:.2f}")
            grid_pts = [r for r in sweep if r["dataset"] == ds and r["method"] == method and r["smoothing"] == k
                        and r["tolerance_s"] == tol and r["video_id"] == "ALL (micro)"]
            ax.scatter([r["recall"] for r in grid_pts], [r["precision"] for r in grid_pts], color=col, s=22, zorder=3)
            for r in grid_pts:
                ax.annotate(f"{r['prominence']:g}:{r['n_pred']}", (r["recall"], r["precision"]), fontsize=6,
                            xytext=(3, 2), textcoords="offset points", color=col)
        orig = next(r for r in sweep if r["dataset"] == ds and r["method"] == method and r["is_original_threshold"]
                    and r["tolerance_s"] == tol and r["video_id"] == "ALL (micro)")
        best = max((r for r in sweep if r["dataset"] == ds and r["method"] == method and r["tolerance_s"] == tol
                    and r["video_id"] == "ALL (micro)"), key=lambda r: (r["f1"], -r["n_pred"]))
        ax.scatter([orig["recall"]], [orig["precision"]], marker="*", s=260, color="#dc2626", zorder=5,
                   label=f"original threshold (prom 1.0, smooth 3): F1 {orig['f1']:.2f}, {orig['n_pred']} preds")
        ax.scatter([best["recall"]], [best["precision"]], marker="D", s=90, facecolor="none", edgecolor="k", lw=1.5,
                   zorder=5, label=f"best grid point (exploratory): F1 {best['f1']:.2f}, {best['n_pred']} preds "
                                   f"(prom {best['prominence']:g}, smooth {best['smoothing']})")
        for fval in (0.2, 0.4, 0.6, 0.8):
            rr = np.linspace(fval / 2 + 1e-3, 1, 200)
            pp = fval * rr / (2 * rr - fval)
            ok = (pp > 0) & (pp <= 1)
            ax.plot(rr[ok], pp[ok], color="#ccc", lw=0.7, zorder=0)
            ax.text(rr[ok][-1], pp[ok][-1], f"F1={fval}", fontsize=7, color="#999")
        n_ref = orig["n_ref"]
        ax.set_title(f"{'MovieChat - CLIP' if ds == 'moviechat' else 'AMI - Sentence-BERT'}: context {st['context_s']:g}s, "
                     f"min segment {st['min_segment_s']:g}s, ±{tol:g}s, {n_ref} references", fontsize=10)
        ax.set_xlabel("recall")
        ax.set_ylabel("precision")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7, loc="lower left")
    fig.suptitle("Experiment B precision-recall: lines = every prominence threshold; dots = sweep grid, "
                 "labelled prominence:predictions", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- 3. leave-one-out
def candidates_grid(ds, proms, smooths):
    g = A_GRID[ds]
    return [(c, m, k, p) for c in g["context"] for m in g["min_seg"] for k in smooths for p in proms]


def loo(D, C, proms, smooths, methods, selection="full"):
    """selection='full': context, min seg, smoothing and prominence chosen on the training fold.
    selection='context_only': context and min seg chosen; smoothing/prominence fixed at the original values."""
    if selection == "context_only":
        proms, smooths = [ORIG_PROM], [ORIG_SMOOTH]
    rows = []
    for ds in ("moviechat", "ami"):
        for method in methods[ds]:
            vids = [v for v in D.videos(ds) if D.raw_curve(v, method, A_GRID[ds]["context"][0])[1] is not None]
            pooled = {tol: [0, 0, 0] for tol in TOLS[ds]}
            for held in vids:
                train = [v for v in vids if v != held]
                best, best_key = None, None
                for ctx, ms, k, prom in candidates_grid(ds, proms, smooths):
                    m, _ = score(C, D, train, method, ctx, ms, k, prom, SEL_TOL[ds])
                    key = (m["f1"], -m["n_pred"], prom, -k, -ctx, -ms)
                    if best_key is None or key > best_key:
                        best, best_key = (ctx, ms, k, prom, m), key
                ctx, ms, k, prom, tm = best
                rec = {"dataset": ds, "method": method, "selection": selection, "held_out": held, "train_videos": "+".join(train),
                       "selected_context_s": ctx, "selected_min_segment_s": ms, "selected_smoothing": k,
                       "selected_prominence": prom, "train_f1": tm["f1"], "train_n_pred": tm["n_pred"],
                       "train_n_ref": tm["n_ref"], "selection_tolerance_s": SEL_TOL[ds]}
                for tol in TOLS[ds]:
                    hm, per = score(C, D, [held], method, ctx, ms, k, prom, tol)
                    rec.update({f"heldout_{x}_tol{tol:g}": hm[x] for x in ("tp", "n_pred", "n_ref", "precision", "recall", "f1")})
                    pooled[tol][0] += hm["tp"]
                    pooled[tol][1] += hm["n_pred"]
                    pooled[tol][2] += hm["n_ref"]
                    rec[f"heldout_boundaries_tol{tol:g}"] = [round(x, 2) for x in per[held][3]]
                rows.append(rec)
            summ = {"dataset": ds, "method": method, "selection": selection, "held_out": "POOLED (micro over folds)", "train_videos": "",
                    "selected_context_s": "", "selected_min_segment_s": "", "selected_smoothing": "",
                    "selected_prominence": "", "train_f1": "", "selection_tolerance_s": SEL_TOL[ds]}
            for tol in TOLS[ds]:
                summ.update({f"heldout_{x}_tol{tol:g}": v for x, v in prf(*pooled[tol]).items()})
            rows.append(summ)
            print(f"[loo ] {ds:9} {method:12} {selection:12} pooled held-out F1 @±{SEL_TOL[ds]:g}s = "
                  f"{summ[f'heldout_f1_tol{SEL_TOL[ds]:g}']:.3f}", flush=True)
    return rows


# ---------------------------------------------------------------- 4. comparison
def comparison(D, C, sweep, held, proms, smooths, methods, data_dir):
    main = {}
    p = data_dir / "evaluation" / "dataset_summary.csv"
    if p.exists():
        for r in csv.DictReader(open(p, encoding="utf-8")):
            main[(r["dataset"], r["method"], float(r["tolerance_s"]))] = r
    main_id = {"clip": "M2_clip", "sbert": "M4_sbert", "fusion_mean": "M5_fusion_mean"}
    rows = []
    for ds in ("moviechat", "ami"):
        for method in methods[ds]:
            vids = [v for v in D.videos(ds) if D.raw_curve(v, method, A_GRID[ds]["context"][0])[1] is not None]

            def add(stage, note, params, scores, counts=None):
                row = {"dataset": ds, "method": method, "stage": stage, "params": params, "note": note}
                for tol in TOLS[ds]:
                    s = scores[tol]
                    row.update({f"f1_tol{tol:g}": s["f1"], f"precision_tol{tol:g}": s["precision"],
                                f"recall_tol{tol:g}": s["recall"], f"n_pred_tol{tol:g}": s["n_pred"],
                                f"chance_f1_tol{tol:g}": chance_f1(D, counts, tol) if counts else None})
                row["n_ref"] = scores[TOLS[ds][0]]["n_ref"]
                rows.append(row)

            def counts_for(ctx, ms, k, prom):
                _, per = score(C, D, vids, method, ctx, ms, k, prom, SEL_TOL[ds])
                return {v: (per[v][1], ms) for v in per}

            mr = {tol: main.get((ds, main_id.get(method, ""), tol)) for tol in TOLS[ds]}
            if all(mr.values()):
                add("1 original (main run)",
                    "first run; SBERT used 20 s left/right windows (pre-chunk encoder)" if method == "sbert" else "first run",
                    "defaults", {t: {k: (float(mr[t][k]) if k not in ("n_pred", "n_ref") else int(mr[t][k]))
                                     for k in ("f1", "precision", "recall", "n_pred", "n_ref")} for t in TOLS[ds]})
            o = ORIGINAL[ds]
            add("2 original parameters, current encoder",
                "first-run min seg/prominence/smoothing; one context for all modalities (first run used 8/10/20 s "
                "for visual/audio/text)" if method.startswith("fusion") else "context/min seg/prominence/smoothing of the first run",
                f"ctx {o['context_s']:g}, min {o['min_segment_s']:g}, prom {ORIG_PROM}, smooth {ORIG_SMOOTH}",
                {t: score(C, D, vids, method, o["context_s"], o["min_segment_s"], ORIG_SMOOTH, ORIG_PROM, t)[0] for t in TOLS[ds]},
                counts_for(o["context_s"], o["min_segment_s"], ORIG_SMOOTH, ORIG_PROM))
            # exploratory A: best context x min seg at the original threshold, selected on all videos
            g = A_GRID[ds]
            bestA = max(((c, m) for c in g["context"] for m in g["min_seg"]),
                        key=lambda cm: (score(C, D, vids, method, cm[0], cm[1], ORIG_SMOOTH, ORIG_PROM, SEL_TOL[ds])[0]["f1"]))
            add("3 exploratory: Experiment A best", "selected on all videos (evaluation labels) - not held out",
                f"ctx {bestA[0]:g}, min {bestA[1]:g}, prom {ORIG_PROM}, smooth {ORIG_SMOOTH}",
                {t: score(C, D, vids, method, bestA[0], bestA[1], ORIG_SMOOTH, ORIG_PROM, t)[0] for t in TOLS[ds]},
                counts_for(bestA[0], bestA[1], ORIG_SMOOTH, ORIG_PROM))
            # exploratory A+B: best over the joint grid on all videos
            bestAB = max(candidates_grid(ds, proms, smooths),
                         key=lambda x: (score(C, D, vids, method, *x, SEL_TOL[ds])[0]["f1"],
                                        -score(C, D, vids, method, *x, SEL_TOL[ds])[0]["n_pred"]))
            add("4 exploratory: Experiment A+B best", "context, min seg, smoothing, prominence selected on all videos - not held out",
                f"ctx {bestAB[0]:g}, min {bestAB[1]:g}, smooth {bestAB[2]}, prom {bestAB[3]:g}",
                {t: score(C, D, vids, method, *bestAB, t)[0] for t in TOLS[ds]}, counts_for(*bestAB))
            for sel, stage, note in (("full", "5 held-out: leave-one-out", "context, min seg, smoothing, prominence selected on training folds only"),
                                     ("context_only", "6 held-out diagnostic: context/min seg only",
                                      "context and min seg selected on training folds; prominence 1.0 and smoothing 3 fixed")):
                fold = [r for r in held if r["dataset"] == ds and r["method"] == method and r["selection"] == sel]
                pooled = next(r for r in fold if r["held_out"].startswith("POOLED"))
                held_counts = {r["held_out"]: (r[f"heldout_n_pred_tol{SEL_TOL[ds]:g}"], r["selected_min_segment_s"])
                               for r in fold if not r["held_out"].startswith("POOLED")}
                picks = "; ".join(f"{r['held_out']}: ctx {r['selected_context_s']:g}/min {r['selected_min_segment_s']:g}"
                                  + (f"/sm {r['selected_smoothing']}/prom {r['selected_prominence']:g}" if sel == "full" else "")
                                  for r in fold if not r["held_out"].startswith("POOLED"))
                add(stage, note, picks,
                    {t: {"f1": pooled[f"heldout_f1_tol{t:g}"], "precision": pooled[f"heldout_precision_tol{t:g}"],
                         "recall": pooled[f"heldout_recall_tol{t:g}"], "n_pred": pooled[f"heldout_n_pred_tol{t:g}"],
                         "n_ref": pooled[f"heldout_n_ref_tol{t:g}"]} for t in TOLS[ds]}, held_counts)
    return rows


def comparison_md(rows) -> str:
    L = ["# Experiment B: original vs exploratory vs held-out F1", "",
         "Exploratory rows select parameters on the same videos they are scored on (optimistic). "
         "Held-out rows select on the other videos/meeting only. AMI leave-one-meeting-out trains on a single meeting.", ""]
    for ds in ("moviechat", "ami"):
        t0, t1 = TOLS[ds]
        L += [f"## {'MovieChat-SSeg (scene boundaries)' if ds == 'moviechat' else 'AMI (topic boundaries, oracle transcript)'}", "",
              f"| Method | Stage | Parameters | F1 @±{t0:g}s | F1 @±{t1:g}s | chance F1 @±{t1:g}s | P / R @±{t1:g}s | Predictions / refs |",
              "|---|---|---|---|---|---|---|---|"]
        for r in rows:
            if r["dataset"] != ds:
                continue
            L.append(f"| {LABEL.get(r['method'], r['method'])} | {r['stage']} | {r['params']} | {r[f'f1_tol{t0:g}']:.3f} | "
                     f"**{r[f'f1_tol{t1:g}']:.3f}** | {'-' if r[f'chance_f1_tol{t1:g}'] is None else format(r[f'chance_f1_tol{t1:g}'], '.3f')} | "
                     f"{r[f'precision_tol{t1:g}']:.2f} / {r[f'recall_tol{t1:g}']:.2f} | "
                     f"{r[f'n_pred_tol{t1:g}']} / {r['n_ref']} |")
        L.append("")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    ap.add_argument("--prominence", nargs="+", type=float, default=PROMINENCE)
    ap.add_argument("--smoothing", nargs="+", type=int, default=SMOOTHING)
    ap.add_argument("--no-fusion", action="store_true", help="skip the fusion comparison")
    args = ap.parse_args()
    D = Data(args.data_dir)
    C = Cands(D)
    out = D.dir / "evaluation" / "experiment_B"
    out.mkdir(parents=True, exist_ok=True)
    methods = {ds: [START[ds]["method"]] + ([] if args.no_fusion else ["fusion_mean"]) for ds in START}

    sweep = threshold_sweep(D, C, args.prominence, args.smoothing, methods)
    write_csv(out / "experiment_b_threshold_sweep.csv", sweep)
    pr_figure(D, C, sweep, args.smoothing, out / "experiment_b_precision_recall.png")
    held = loo(D, C, args.prominence, args.smoothing, methods, "full") + \
        loo(D, C, args.prominence, args.smoothing, methods, "context_only")
    write_csv(out / "experiment_b_heldout.csv", held)
    comp = comparison(D, C, sweep, held, args.prominence, args.smoothing, methods, D.dir)
    write_csv(out / "experiment_b_comparison.csv", comp)
    md = comparison_md(comp)
    (out / "experiment_b_comparison.md").write_text(md, encoding="utf-8")
    (out / "experiment_b_params.json").write_text(json.dumps(
        {"start": START, "original": ORIGINAL, "prominence_grid": args.prominence, "smoothing_grid": args.smoothing,
         "selection_tolerance": SEL_TOL, "tolerances": TOLS, "experiment_A_grid": A_GRID, "methods": methods,
         "tie_break": "train F1, then fewer train predictions, higher prominence, lower smoothing, shorter context, shorter min segment",
         "base_config": D.base.to_dict()}, indent=1), encoding="utf-8")
    print("\n" + md)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
