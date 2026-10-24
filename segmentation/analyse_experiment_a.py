"""Experiment A analysis: boundary-eligibility audit, trends, and ranking quality vs boundary count.
Uses the existing Experiment A outputs and cached embeddings only (no inference, no re-sweep of
predictions; Experiment A files are read, never written).

    python -m segmentation.analyse_experiment_a

Writes to evaluation/experiment_B/:
    experiment_a_eligibility.csv       per dataset x method x context x min segment x tolerance
    experiment_a_reference_audit.csv   every reference boundary in the two best configurations
    experiment_a_ranking.csv           AP (threshold-free ranking quality), F1 at the original threshold,
                                       oracle-threshold F1 and candidate ceiling, incl. diagnostic contexts
    experiment_a_sensitivity.png
    experiment_a_tables.md             the tables used in experiment_a_analysis.md
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from segmentation.analysis_tools import Data, candidates, detection_curve, eligibility, pr_points  # noqa: E402
from segmentation.config import DEFAULT_DATA_DIR  # noqa: E402

A_GRID = {"moviechat": {"context": [5, 10, 20], "min_seg": [15, 30, 45]},
          "ami": {"context": [30, 60, 90], "min_seg": [60, 90, 120]}}
# diagnostic extensions (not part of Experiment A): original settings and the counterfactual cells
# needed to separate context from minimum segment
DIAG_GRID = {"moviechat": {"context": [5, 8, 10, 20, 40], "min_seg": [15, 30, 45]},
             "ami": {"context": [20, 30, 45, 60, 90, 120], "min_seg": [15, 30, 60, 90, 120]}}
METHODS = {"moviechat": ["clip", "clap", "fusion_mean", "fusion_max"],
           "ami": ["clip", "clap", "sbert", "fusion_mean", "fusion_max"]}
PRIMARY = {"moviechat": ("clip", 5.0), "ami": ("sbert", 10.0)}
TOLS = {"moviechat": [2.0, 5.0], "ami": [5.0, 10.0]}
BEST = {"moviechat": ("clip", 20, 30, 5.0), "ami": ("sbert", 60, 90, 10.0)}


def write_csv(path, rows):
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def ranking_stats(D, ds, method, ctx, ms, tol, smooth_k=3, prom=1.0):
    pv = []
    for v in D.videos(ds):
        g, sc = D.raw_curve(v, method, ctx)
        if sc is None:
            continue
        t, p = candidates(g, detection_curve(sc, smooth_k, D.base), ms, D.dur(v), D.base.grid_s)
        pv.append((t, p, D.refs[v]))
    if not pv:
        return None
    pts, ap = pr_points(pv, tol)
    at = [q for q in pts if q["prominence"] >= prom]
    cur = at[-1] if at else {"f1": 0.0, "n_pred": 0, "precision": 0.0, "recall": 0.0}
    best = max(pts, key=lambda q: (q["f1"], -q["n_pred"])) if pts else cur
    allc = pts[-1] if pts else cur
    return {"ap": round(ap, 4), "f1_at_original_threshold": round(cur["f1"], 4), "n_pred_at_original_threshold": cur["n_pred"],
            "precision_at_original_threshold": round(cur["precision"], 4), "recall_at_original_threshold": round(cur["recall"], 4),
            "oracle_threshold_f1": round(best["f1"], 4), "oracle_threshold_prominence": round(best["prominence"], 3),
            "oracle_threshold_n_pred": best["n_pred"], "candidate_ceiling_recall": round(allc["recall"], 4),
            "n_candidates": allc["n_pred"], "n_ref": sum(len(r) for *_, r in pv)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    args = ap.parse_args()
    D = Data(args.data_dir)
    out = D.dir / "evaluation" / "experiment_B"
    out.mkdir(parents=True, exist_ok=True)

    # ---------------- 1. eligibility audit
    elig = []
    for ds in ("moviechat", "ami"):
        for method in METHODS[ds]:
            for ctx in A_GRID[ds]["context"]:
                for ms in A_GRID[ds]["min_seg"]:
                    for tol in TOLS[ds]:
                        tot = {"naive": 0, "edge": 0, "evidence": 0, "single": 0, "joint": 0, "n_ref": 0}
                        spans = {}
                        for v in D.videos(ds):
                            rows, joint, span = eligibility(D, v, method, ctx, ms, tol)
                            tot["n_ref"] += len(rows)
                            tot["naive"] += sum(r["naive_reachable"] for r in rows)
                            tot["edge"] += sum(r["edge_ok"] for r in rows)
                            tot["evidence"] += sum(r["evidence_ok"] for r in rows)
                            tot["single"] += sum(r["single_ok"] for r in rows)
                            tot["joint"] += joint
                            spans[v] = span
                        elig.append({"dataset": ds, "method": method, "context_s": ctx, "min_segment_s": ms,
                                     "tolerance_s": tol, "n_ref": tot["n_ref"],
                                     "reachable_refs_experiment_A": tot["naive"],
                                     "eligible_edge_with_tolerance": tot["edge"],
                                     "eligible_evidence_available": tot["evidence"],
                                     "eligible_individually": tot["single"],
                                     "max_matchable_jointly": tot["joint"],
                                     "max_recall": round(tot["joint"] / tot["n_ref"], 4),
                                     "evidence_span_s": json.dumps(spans)})
    write_csv(out / "experiment_a_eligibility.csv", elig)

    audit = []
    for ds, (method, ctx, ms, tol) in BEST.items():
        for v in D.videos(ds):
            rows, joint, span = eligibility(D, v, method, ctx, ms, tol)
            dur = D.dur(v)
            refs = D.refs[v]
            for i, r in enumerate(rows):
                prev_gap = refs[i] - refs[i - 1] if i else None
                next_gap = refs[i + 1] - refs[i] if i + 1 < len(refs) else None
                reason = []
                if not r["naive_reachable"]:
                    reason.append(f"within {ms:g}s of start/end (Experiment A counted it unreachable)")
                if not r["edge_ok"]:
                    reason.append("no allowed prediction position within tolerance")
                if not r["evidence_ok"]:
                    reason.append("no score (insufficient context/evidence) within tolerance")
                if (prev_gap is not None and prev_gap < ms - 2 * tol) or (next_gap is not None and next_gap < ms - 2 * tol):
                    reason.append(f"neighbouring reference closer than min segment - 2 x tolerance ({ms - 2 * tol:g}s)")
                audit.append({"dataset": ds, "video_id": v, "method": method, "context_s": ctx, "min_segment_s": ms,
                              "tolerance_s": tol, "ref_s": r["ref_s"], "duration_s": round(dur, 2),
                              "gap_prev_s": None if prev_gap is None else round(prev_gap, 2),
                              "gap_next_s": None if next_gap is None else round(next_gap, 2), **r,
                              "evidence_span_s": json.dumps(span), "video_max_matchable": joint,
                              "notes": "; ".join(reason)})
    write_csv(out / "experiment_a_reference_audit.csv", audit)

    # ---------------- 2. ranking quality vs count (A grid + diagnostic cells)
    rank = []
    for ds in ("moviechat", "ami"):
        for method in METHODS[ds]:
            grid = DIAG_GRID[ds] if method in ("clip", "sbert", "fusion_mean") else A_GRID[ds]
            for ctx in grid["context"]:
                for ms in grid["min_seg"]:
                    for tol in TOLS[ds]:
                        st = ranking_stats(D, ds, method, ctx, ms, tol)
                        if st is None:
                            continue
                        rank.append({"dataset": ds, "method": method, "context_s": ctx, "min_segment_s": ms,
                                     "tolerance_s": tol,
                                     "in_experiment_A_grid": ctx in A_GRID[ds]["context"] and ms in A_GRID[ds]["min_seg"],
                                     **st})
        print(f"[rank] {ds} done", flush=True)
    write_csv(out / "experiment_a_ranking.csv", rank)

    # ---------------- 3. figure
    figure(rank, out / "experiment_a_sensitivity.png")
    # ---------------- 4. tables
    (out / "experiment_a_tables.md").write_text(tables(elig, audit, rank), encoding="utf-8")
    print(f"Wrote {out}")


def _sel(rank, ds, method, tol, **kw):
    return [r for r in rank if r["dataset"] == ds and r["method"] == method and r["tolerance_s"] == tol
            and all(r[k] == v for k, v in kw.items())]


def figure(rank, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 4, figsize=(17, 7.5))
    for row, ds in enumerate(("moviechat", "ami")):
        method, tol = PRIMARY[ds]
        g = DIAG_GRID[ds]
        cmap = plt.get_cmap("viridis")
        mss = g["min_seg"]
        ctxs = g["context"]
        a, b, c, d = axes[row]
        for i, ms in enumerate(mss):
            col = cmap(i / max(1, len(mss) - 1))
            pts = sorted(_sel(rank, ds, method, tol, min_segment_s=ms), key=lambda r: r["context_s"])
            x = [p["context_s"] for p in pts]
            a.plot(x, [p["f1_at_original_threshold"] for p in pts], "o-", color=col, label=f"min seg {ms}s")
            d.plot(x, [p["ap"] for p in pts], "o-", color=col, label=f"min seg {ms}s")
        for i, ctx in enumerate(ctxs):
            col = plt.get_cmap("plasma")(i / max(1, len(ctxs) - 1))
            pts = sorted(_sel(rank, ds, method, tol, context_s=ctx), key=lambda r: r["min_segment_s"])
            b.plot([p["min_segment_s"] for p in pts], [p["n_pred_at_original_threshold"] for p in pts], "o-", color=col,
                   label=f"context {ctx}s")
            c.scatter([p["recall_at_original_threshold"] for p in pts], [p["precision_at_original_threshold"] for p in pts],
                      color=col, s=[18 + 0.6 * p["min_segment_s"] for p in pts], label=f"context {ctx}s", alpha=0.85)
        n_ref = _sel(rank, ds, method, tol)[0]["n_ref"]
        b.axhline(n_ref, color="k", ls="--", lw=1)
        b.text(b.get_xlim()[0], n_ref, f" references = {n_ref}", va="bottom", fontsize=8)
        for fval in (0.2, 0.4, 0.6):
            rr = np.linspace(fval / 2 + 1e-3, 1, 100)
            pp = fval * rr / (2 * rr - fval)
            ok = (pp > 0) & (pp <= 1)
            c.plot(rr[ok], pp[ok], color="#bbb", lw=0.8)
            c.text(rr[ok][-1], pp[ok][-1], f"F1={fval}", fontsize=7, color="#888")
        name = "MovieChat - CLIP" if ds == "moviechat" else "AMI - Sentence-BERT"
        a.set_title(f"{name}: F1 @±{tol:g}s vs context", fontsize=10)
        b.set_title("Predicted boundaries vs min segment", fontsize=10)
        c.set_title("Precision vs recall (size = min segment)", fontsize=10)
        d.set_title("Ranking quality (AP) vs context", fontsize=10)
        for ax, xl, yl in ((a, "context (s)", "F1"), (b, "min segment (s)", "predictions"),
                           (c, "recall", "precision"), (d, "context (s)", "average precision")):
            ax.set_xlabel(xl)
            ax.set_ylabel(yl)
            ax.grid(alpha=0.25)
        for ax in (a, d):
            for ctx in g["context"]:
                if ctx not in A_GRID[ds]["context"]:
                    ax.axvline(ctx, color="#ddd", lw=6, zorder=0)
        a.legend(fontsize=7)
        b.legend(fontsize=7)
        c.set_xlim(0, 1)
        c.set_ylim(0, 1)
    fig.suptitle("Experiment A sensitivity (grey columns = diagnostic contexts outside the Experiment A grid; "
                 "F1 at the original threshold, prominence 1.0, smoothing 3)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def tables(elig, audit, rank) -> str:
    L = ["# Experiment A analysis tables (generated)", ""]
    L += ["## Boundary eligibility (primary methods, primary tolerance)", "",
          "| Dataset | Method | Context | Min seg | Refs | A `reachable_refs` | Edge-eligible (tol-aware) | Evidence available | Max matchable jointly |",
          "|---|---|---|---|---|---|---|---|---|"]
    for ds, (method, tol) in PRIMARY.items():
        for r in elig:
            if r["dataset"] == ds and r["method"] == method and r["tolerance_s"] == tol:
                L.append(f"| {ds} | {method} | {r['context_s']} | {r['min_segment_s']} | {r['n_ref']} | "
                         f"{r['reachable_refs_experiment_A']} | {r['eligible_edge_with_tolerance']} | "
                         f"{r['eligible_evidence_available']} | {r['max_matchable_jointly']} ({r['max_recall']:.2f}) |")
    L += ["", "## Reference boundaries with a structural disadvantage in the best configurations", "",
          "| Dataset | Video | Ref (s) | Gap prev / next (s) | Notes |", "|---|---|---|---|---|"]
    for a in audit:
        if a["notes"]:
            L.append(f"| {a['dataset']} | {a['video_id']} | {a['ref_s']} | {a['gap_prev_s']} / {a['gap_next_s']} | {a['notes']} |")
    L += ["", "## Ranking quality vs boundary count (primary methods, primary tolerance)", "",
          "AP = average precision of candidates ranked by peak prominence (threshold-free). "
          "F1@1.0 = F1 at the original prominence threshold. Oracle F1 = best F1 over all thresholds "
          "(exploratory, selected on the evaluation labels). Ceiling = recall if every candidate were kept.", "",
          "| Dataset | Method | Context | Min seg | AP | F1@1.0 (n) | Oracle F1 (n, prom) | Ceiling recall (candidates) | In A grid |",
          "|---|---|---|---|---|---|---|---|---|"]
    for ds, (method, tol) in PRIMARY.items():
        for r in _sel(rank, ds, method, tol):
            L.append(f"| {ds} | {method} | {r['context_s']} | {r['min_segment_s']} | {r['ap']:.2f} | "
                     f"{r['f1_at_original_threshold']:.2f} ({r['n_pred_at_original_threshold']}) | "
                     f"{r['oracle_threshold_f1']:.2f} ({r['oracle_threshold_n_pred']}, {r['oracle_threshold_prominence']:.1f}) | "
                     f"{r['candidate_ceiling_recall']:.2f} ({r['n_candidates']}) | {'yes' if r['in_experiment_A_grid'] else 'diagnostic'} |")
    L += ["", "## Fusion and secondary modalities (primary tolerance, AP)", "",
          "| Dataset | Method | Context | Min seg | AP | F1@1.0 (n) |", "|---|---|---|---|---|---|"]
    for ds, (_, tol) in PRIMARY.items():
        for method in METHODS[ds]:
            if method == PRIMARY[ds][0]:
                continue
            for r in _sel(rank, ds, method, tol):
                if r["in_experiment_A_grid"]:
                    L.append(f"| {ds} | {method} | {r['context_s']} | {r['min_segment_s']} | {r['ap']:.2f} | "
                             f"{r['f1_at_original_threshold']:.2f} ({r['n_pred_at_original_threshold']}) |")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
