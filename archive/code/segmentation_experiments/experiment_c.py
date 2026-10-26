"""Experiment C: modality-specific context and reliability-aware fusion (cached embeddings only).

No pretrained model is run. Inputs: the embedding cache (CLIP, CLAP, Sentence-BERT chunks), the
reference boundaries (evaluation only) and one cheap signal-processing feature for the audio
activity cue: frame RMS loudness (0.5 s frames, ffmpeg decode, no model), cached in
cache/experiment_C_audio_energy/<id>_rms.csv.

Design
  * Every modality has its own context window (CONTEXT). Change scores are evaluated directly at the
    2 s grid times from the timestamped embeddings (no resampling of embeddings, so nothing is
    interpolated across gaps): a side with too little evidence gives NaN, never 0.
  * Boundary selection is fixed per dataset (FIXED_SEL) and identical for every variant.
  * Reliability weights are computed from the input only (sample coverage, transcript words,
    within-window embedding dispersion, audio loudness). References are never used.
  * Held-out: leave-one-recording-out; any choice (method, prominence) is made on training folds only.

Outputs -> <data>/evaluation/experiment_C/ (Experiments A/B are not touched).
Usage: python -m segmentation.experiment_c --data-dir data/AMI_ES2002_segmentation_reference
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from .analysis_tools import Data, candidates, pr_points
from .boundary_detection import detect, smooth
from .change_scores import embedding_change, make_grid, text_change
from .evaluation import chance_tp, match, prf
from .fusion import normalise

# ------------------------------------------------------------------ configuration
CONTEXT = {"moviechat": {"clip": 8.0, "clap": 10.0},
           "ami": {"clip": 10.0, "clap": 10.0, "sbert": 60.0}}
FIXED_SEL = {"moviechat": {"min_segment_s": 15.0, "peak_prominence": 1.0, "smooth_window": 3},
             "ami": {"min_segment_s": 60.0, "peak_prominence": 1.0, "smooth_window": 3}}
TOL = {"moviechat": {"primary": 5.0, "strict": 2.0}, "ami": {"primary": 10.0, "strict": 5.0}}
BASELINE = {"moviechat": "clip", "ami": "sbert"}            # strongest unimodal references (Exps A/B)
FIXED_W = {"moviechat": {"clip": 0.75, "clap": 0.25},       # a-priori dominant modality per footage type
           "ami": {"sbert": 0.75, "clip": 0.125, "clap": 0.125}}
PROM_SET = [0.5, 1.0, 1.5, 2.0]                             # only threshold set used in fold selection
SILENCE_DBFS = -50.0                                        # audio activity gate (frame RMS)
MOD = {"clip": "visual", "clap": "audio", "sbert": "text"}
LABEL = {"clip": "CLIP", "clap": "CLAP", "sbert": "SBERT"}
CHANCE_TRIALS = 2000


def spec(name, mods, mode="mean", weights="equal", rel="none", ctx="async", group="async"):
    return {"name": name, "mods": tuple(mods), "mode": mode, "weights": weights, "rel": rel, "ctx": ctx,
            "group": group}


ASYNC = {
    "moviechat": [
        spec("CLIP", ["clip"]), spec("CLAP", ["clap"]),
        spec("CLIP+CLAP mean", ["clip", "clap"]),
        spec("CLIP+CLAP max", ["clip", "clap"], mode="max"),
        spec("CLIP+CLAP 0.75/0.25", ["clip", "clap"], weights="fixed"),
        spec("CLIP+CLAP mean, shared ctx 8s", ["clip", "clap"], ctx=8.0, group="shared_control"),
        spec("CLIP+CLAP mean, shared ctx 10s", ["clip", "clap"], ctx=10.0, group="shared_control"),
        spec("CLIP+CLAP mean, shared ctx 20s", ["clip", "clap"], ctx=20.0, group="shared_control"),
    ],
    "ami": [
        spec("CLIP", ["clip"]), spec("CLAP", ["clap"]), spec("SBERT", ["sbert"]),
        spec("CLAP+SBERT mean", ["clap", "sbert"]),
        spec("CLIP+SBERT mean", ["clip", "sbert"]),
        spec("All three mean", ["clip", "clap", "sbert"]),
        spec("All three max", ["clip", "clap", "sbert"], mode="max"),
        spec("All three mean, shared ctx 10s", ["clip", "clap", "sbert"], ctx=10.0, group="shared_control"),
        spec("All three mean, shared ctx 60s", ["clip", "clap", "sbert"], ctx=60.0, group="shared_control"),
        spec("CLIP+SBERT mean, shared ctx 60s", ["clip", "sbert"], ctx=60.0, group="shared_control"),
    ],
}


def _rel_set(base_name, mods):
    return [
        spec(f"{base_name} | equal mean", mods, group="reliability"),
        spec(f"{base_name} | max", mods, mode="max", group="reliability"),
        spec(f"{base_name} | fixed weights", mods, weights="fixed", group="reliability"),
        spec(f"{base_name} | R1 coverage", mods, rel="coverage", group="reliability"),
        spec(f"{base_name} | R2 stability", mods, rel="stability", group="reliability"),
        spec(f"{base_name} | R3 coverage x stability", mods, rel="both", group="reliability"),
        spec(f"{base_name} | fixed weights x R3", mods, weights="fixed", rel="both", group="reliability"),
        spec(f"{base_name} | R3 time-shuffled (control)", mods, rel="both_shuffled", group="reliability_control"),
    ]


RELIABILITY = {
    "moviechat": _rel_set("CLIP+CLAP", ["clip", "clap"]),
    "ami": _rel_set("All three", ["clip", "clap", "sbert"]) + [
        spec("CLAP+SBERT | equal mean", ["clap", "sbert"], group="reliability"),
        spec("CLAP+SBERT | R1 coverage", ["clap", "sbert"], rel="coverage", group="reliability"),
        spec("CLAP+SBERT | R2 stability", ["clap", "sbert"], rel="stability", group="reliability"),
        spec("CLAP+SBERT | R3 coverage x stability", ["clap", "sbert"], rel="both", group="reliability"),
        spec("CLAP+SBERT | R1 coverage time-shuffled (control)", ["clap", "sbert"], rel="coverage_shuffled",
             group="reliability_control"),
        spec("CLAP+SBERT | R3 time-shuffled (control)", ["clap", "sbert"], rel="both_shuffled",
             group="reliability_control"),
    ],
}


# ------------------------------------------------------------------ evidence per modality
def side_stats(times, emb, grid, ctx, weights=None):
    """Per grid time: evidence on each side (sample count or word sum) and within-side dispersion
    (weighted mean of 1 - cos(x_i, side centroid)). Same windows as change_scores.embedding_change."""
    order = np.argsort(times)
    times, emb = times[order], emb[order]
    w = np.ones(len(times)) if weights is None else np.asarray(weights, float)[order]
    U = emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12)
    n = np.zeros((2, len(grid)))
    disp = np.full((2, len(grid)), np.nan)
    for i, t in enumerate(grid):
        for s, (a, b) in enumerate(((t - ctx, t), (t, t + ctx))):
            lo, hi = np.searchsorted(times, [a, b])
            ww = w[lo:hi]
            n[s, i] = ww.sum() if weights is not None else hi - lo
            if hi - lo >= 2 and ww.sum() > 0:
                c = ww @ U[lo:hi] / ww.sum()
                c = c / max(np.linalg.norm(c), 1e-12)
                disp[s, i] = float(ww @ (1 - U[lo:hi] @ c) / ww.sum())
    return n, disp


def load_rms(data_dir, vid):
    p = Path(data_dir) / "cache" / "experiment_C_audio_energy" / f"{vid}_rms.csv"
    if not p.exists():
        return None
    a = np.loadtxt(p, delimiter=",", skiprows=1)
    return a[:, 0], a[:, 1]


def activity(rms, grid, ctx):
    """min over the two sides of the share of 0.5 s frames louder than SILENCE_DBFS."""
    t, db = rms
    act = (db > SILENCE_DBFS).astype(float)
    out = np.full(len(grid), np.nan)
    for i, g in enumerate(grid):
        l0, l1 = np.searchsorted(t, [g - ctx, g])
        r0, r1 = np.searchsorted(t, [g, g + ctx])
        if l1 > l0 and r1 > r0:
            out[i] = min(act[l0:l1].mean(), act[r0:r1].mean())
    return out


class Evidence:
    """Raw change score and reliability terms for one (video, modality, context)."""

    def __init__(self, D: Data, vid, mod, ctx):
        self.grid = make_grid(D.dur(vid), D.base.grid_s)
        cache = D.caches(vid).get(MOD[mod])
        self.ok = cache is not None
        if not self.ok:
            return
        cfg = D.base
        if mod == "sbert":
            self.d = text_change(cache, self.grid, ctx, cfg.text_min_words)
            n, disp = side_stats(cache["times"], cache["emb"], self.grid, ctx, cache["n_words"])
            m = np.minimum(n[0], n[1])
            med = np.median(m[np.isfinite(self.d)]) if np.isfinite(self.d).any() else 1.0
            self.coverage = np.clip(m / max(med, 1e-9), 0, 1)          # words on the poorer side vs typical
        else:
            step = 1.0 / cfg.visual_fps if mod == "clip" else cfg.audio_hop_s
            self.d = embedding_change(cache["times"], cache["emb"], self.grid, ctx, step, cfg.min_context_coverage)
            n, disp = side_stats(cache["times"], cache["emb"], self.grid, ctx)
            self.coverage = np.clip(np.minimum(n[0], n[1]) / (ctx / step), 0, 1)
            if mod == "clap":
                rms = load_rms(D.dir, vid)
                self.activity = activity(rms, self.grid, ctx) if rms is not None else np.ones(len(self.grid))
                self.coverage = self.coverage * np.nan_to_num(self.activity, nan=0.0)
        with np.errstate(all="ignore"):
            import warnings
            warnings.simplefilter("ignore", RuntimeWarning)
            s = np.nanmean(disp, axis=0)
        med_s = np.nanmedian(s[np.isfinite(self.d)]) if np.isfinite(self.d).any() else 1.0
        self.stability = 1.0 / (1.0 + s / max(med_s, 1e-9))          # 0.5 at the video's median dispersion
        self.dispersion = s

    def weight(self, rel, shift=0):
        if rel == "none":
            r = np.ones(len(self.grid))
        elif rel == "coverage":
            r = self.coverage
        elif rel == "stability":
            r = self.stability
        elif rel == "both":
            r = self.coverage * self.stability
        else:
            raise ValueError(rel)
        r = np.nan_to_num(r, nan=0.0)
        return np.roll(r, shift) if shift else r


class Runner:
    def __init__(self, data_dir: Path):
        self.D = Data(data_dir)
        self._ev = {}

    def ev(self, vid, mod, ctx):
        k = (vid, mod, ctx)
        if k not in self._ev:
            self._ev[k] = Evidence(self.D, vid, mod, ctx)
        return self._ev[k]

    def contexts(self, ds, sp):
        return {m: (CONTEXT[ds][m] if sp["ctx"] == "async" else float(sp["ctx"])) for m in sp["mods"]}

    def score(self, vid, sp):
        """Fused (or unimodal) raw score on the grid; NaN where no modality has evidence."""
        ds = self.D.dataset(vid)
        ctx = self.contexts(ds, sp)
        evs = {m: self.ev(vid, m, ctx[m]) for m in sp["mods"]}
        evs = {m: e for m, e in evs.items() if e.ok and np.isfinite(e.d).any()}
        if not evs:
            return None, None
        grid = next(iter(evs.values())).grid
        if len(sp["mods"]) == 1:
            return grid, next(iter(evs.values())).d
        names = list(evs)
        Z = np.vstack([normalise(evs[m].d, self.D.base.normalise, self.D.base.score_clip) for m in names])
        A = np.isfinite(Z)
        if sp["mode"] == "max":
            fused = np.where(A.any(0), np.nanmax(np.where(A, Z, -np.inf), axis=0), np.nan)
            return grid, fused
        fw = FIXED_W[ds] if sp["weights"] == "fixed" else {}
        shift = len(grid) // 2 if sp["rel"].endswith("shuffled") else 0
        W = np.vstack([fw.get(m, 1.0) * evs[m].weight(sp["rel"].replace("_shuffled", ""), shift) for m in names])
        W = np.where(A, W, 0.0)                                       # missing modality -> no vote, not a zero
        den = W.sum(0)
        num = np.where(A, W * Z, 0.0).sum(0)
        fused = np.where(den > 1e-12, num / np.where(den > 1e-12, den, 1), np.nan)
        return grid, fused

    def predict(self, vid, sp, prom=None):
        ds = self.D.dataset(vid)
        grid, s = self.score(vid, sp)
        if s is None:
            return None
        sel = dict(FIXED_SEL[ds])
        if prom is not None:
            sel["peak_prominence"] = prom
        cfg = replace(self.D.base, **sel)
        pred, _ = detect(grid, s, cfg, self.D.dur(vid))
        return pred

    def evaluate(self, vids, sp, prom=None, tol_kind="primary", rng=None):
        rows = []
        for v in vids:
            ds = self.D.dataset(v)
            pred = self.predict(v, sp, prom)
            if pred is None:
                continue
            ref = self.D.refs[v]
            tol = TOL[ds][tol_kind]
            tp = len(match(pred, ref, tol))
            row = {"video_id": v, "tp": tp, "fp": len(pred) - tp, "fn": len(ref) - tp, "n_pred": len(pred),
                   "n_ref": len(ref), "pred": pred}
            if rng is not None:
                ms = FIXED_SEL[ds]["min_segment_s"]
                row["chance_tp"] = chance_tp(len(pred), ref, tol, ms, self.D.dur(v) - ms, CHANCE_TRIALS, rng)
            rows.append(row)
        return rows


def average_precision(R: Runner, ds, sp, tol_kind="primary"):
    """Threshold-free ranking quality under the fixed min-segment and smoothing (Experiment B method):
    candidates ranked by prominence, step-wise AP over the dataset (micro)."""
    per = []
    for v in R.D.videos(ds):
        grid, s = R.score(v, sp)
        if s is None:
            continue
        curve = smooth(normalise(s, R.D.base.normalise, R.D.base.score_clip), FIXED_SEL[ds]["smooth_window"])
        t, p = candidates(grid, curve, FIXED_SEL[ds]["min_segment_s"], R.D.dur(v), R.D.base.grid_s)
        per.append((t, p, R.D.refs[v]))
    return round(pr_points(per, TOL[ds][tol_kind])[1], 4)


def micro(rows):
    tp = sum(r["tp"] for r in rows)
    out = prf(tp, sum(r["n_pred"] for r in rows), sum(r["n_ref"] for r in rows))
    out.update(fp=out["n_pred"] - tp, fn=out["n_ref"] - tp)
    if rows and "chance_tp" in rows[0]:
        out["chance_f1"] = round(prf(sum(r["chance_tp"] for r in rows), out["n_pred"], out["n_ref"])["f1"], 4)
    return out


# ------------------------------------------------------------------ experiments
def run_table(R: Runner, specs_by_ds, experiment):
    table, per_video = [], []
    for ds, specs in specs_by_ds.items():
        vids = R.D.videos(ds)
        for sp in specs:
            rec = {"dataset": ds, "experiment": experiment, "method": sp["name"], "group": sp["group"],
                   "modalities": "+".join(LABEL[m] for m in sp["mods"]),
                   "contexts_s": "; ".join(f"{LABEL[m]} {c:g}" for m, c in R.contexts(ds, sp).items()),
                   "mode": sp["mode"], "weights": sp["weights"] if len(sp["mods"]) > 1 else "",
                   "reliability": sp["rel"] if len(sp["mods"]) > 1 else "",
                   **{f"sel_{k}": v for k, v in FIXED_SEL[ds].items()}}
            for kind in ("primary", "strict"):
                rows = R.evaluate(vids, sp, tol_kind=kind, rng=np.random.default_rng(0))
                m = micro(rows)
                sfx = "" if kind == "primary" else "_strict"
                rec[f"tol{sfx}_s"] = TOL[ds][kind]
                for k in ("tp", "fp", "fn", "n_pred", "n_ref", "precision", "recall", "f1", "chance_f1"):
                    rec[f"{k}{sfx}"] = m[k]
                rec[f"ap{sfx}"] = average_precision(R, ds, sp, kind)
                for r in rows:
                    pr = prf(r["tp"], r["n_pred"], r["n_ref"])
                    per_video.append({"dataset": ds, "experiment": experiment, "method": sp["name"], "protocol": "fixed",
                                      "video_id": r["video_id"], "tolerance": kind, "tol_s": TOL[ds][kind],
                                      "tp": r["tp"], "fp": r["fp"], "fn": r["fn"], "n_pred": r["n_pred"],
                                      "n_ref": r["n_ref"], "precision": pr["precision"], "recall": pr["recall"],
                                      "f1": pr["f1"], "selected": "",
                                      "predictions_s": " ".join(f"{x:g}" for x in r["pred"])})
            table.append(rec)
            print(f"[{experiment[:5]}] {ds:9} {sp['name']:42} F1 {rec['f1']:.3f} (n_pred {rec['n_pred']:3d}) "
                  f"strict {rec['f1_strict']:.3f} AP {rec['ap']:.3f}", flush=True)
    return table, per_video


def loo(R: Runner, ds, candidates, proms, protocol):
    """Pick (method, prominence) by training-fold micro F1 at the primary tolerance.
    Tie-break: fewer predictions, then earlier candidate (simpler), then prominence closer to 1.0."""
    vids = R.D.videos(ds)
    folds, pooled = [], {"primary": [], "strict": []}
    for held in vids:
        train = [v for v in vids if v != held]
        best, key_best = None, None
        for ci, sp in enumerate(candidates):
            for prom in proms:
                m = micro(R.evaluate(train, sp, prom))
                key = (m["f1"], -m["n_pred"], -ci, -abs(prom - 1.0))
                if key_best is None or key > key_best:
                    best, key_best = (sp, prom, m), key
        sp, prom, tm = best
        rec = {"dataset": ds, "protocol": protocol, "held_out": held, "selected_method": sp["name"],
               "selected_prominence": prom, "train_f1": tm["f1"], "train_n_pred": tm["n_pred"]}
        for kind in ("primary", "strict"):
            rows = R.evaluate([held], sp, prom, kind)
            pooled[kind] += rows
            r = rows[0]
            rec.update({f"{kind}_{k}": r[k] for k in ("tp", "fp", "fn", "n_pred", "n_ref")})
            rec[f"{kind}_f1"] = prf(r["tp"], r["n_pred"], r["n_ref"])["f1"]
            if kind == "primary":
                rec["predictions_s"] = " ".join(f"{x:g}" for x in r["pred"])
        folds.append(rec)
    return folds, {k: micro(v) for k, v in pooled.items()}


def heldout(R: Runner, async_rows):
    out, per_video = [], []
    for ds in ("moviechat", "ami"):
        A = [s for s in ASYNC[ds] if s["group"] == "async"]
        fusion_async = [s for s in A if len(s["mods"]) > 1]
        rel = [s for s in RELIABILITY[ds] if s["group"] == "reliability"]
        base = [s for s in A if s["mods"] == (BASELINE[ds],)]
        unimodal = [s for s in A if len(s["mods"]) == 1]
        uni_first = base + [s for s in unimodal if s not in base]

        # fixed configurations (no ground-truth selection at all)
        for sp in A + [s for s in RELIABILITY[ds] if s["group"].startswith("reliability")]:
            r_p = micro(R.evaluate(R.D.videos(ds), sp, tol_kind="primary"))
            r_s = micro(R.evaluate(R.D.videos(ds), sp, tol_kind="strict"))
            out.append(_held_row(ds, "fixed (no selection)", sp["name"], "-", r_p, r_s))

        protocols = [
            ("fold-selected: baseline prominence", base, PROM_SET),
            ("fold-selected: unimodal method + prominence", uni_first, PROM_SET),
            ("fold-selected: async fusion method (prominence 1.0)", fusion_async, [1.0]),
            ("fold-selected: reliability fusion method (prominence 1.0)", rel, [1.0]),
            ("fold-selected: any fusion method + prominence", fusion_async + rel, PROM_SET),
            ("fold-selected: any method (unimodal or fusion) + prominence", uni_first + fusion_async + rel, PROM_SET),
        ]
        for name, cands, proms in protocols:
            folds, pooled = loo(R, ds, cands, proms, name)
            sel = "; ".join(f"{f['held_out']}: {f['selected_method']} @p{f['selected_prominence']:g}" for f in folds)
            out.append(_held_row(ds, name, "selected per fold", sel, pooled["primary"], pooled["strict"]))
            for f in folds:
                for kind in ("primary", "strict"):
                    per_video.append({"dataset": ds, "experiment": "heldout", "method": f["selected_method"],
                                      "protocol": name, "video_id": f["held_out"], "tolerance": kind,
                                      "tol_s": TOL[ds][kind], "tp": f[f"{kind}_tp"], "fp": f[f"{kind}_fp"],
                                      "fn": f[f"{kind}_fn"], "n_pred": f[f"{kind}_n_pred"], "n_ref": f[f"{kind}_n_ref"],
                                      **{k: v for k, v in prf(f[f"{kind}_tp"], f[f"{kind}_n_pred"], f[f"{kind}_n_ref"]).items()
                                         if k in ("precision", "recall")},
                                      "f1": f[f"{kind}_f1"], "selected": f"p={f['selected_prominence']:g}, train F1 {f['train_f1']:.3f}",
                                      "predictions_s": f["predictions_s"] if kind == "primary" else ""})
            print(f"[held ] {ds:9} {name:60} F1 {pooled['primary']['f1']:.3f} | {sel}", flush=True)
    return out, per_video


def _held_row(ds, protocol, method, selections, p, s):
    return {"dataset": ds, "protocol": protocol, "method": method, "selections_per_fold": selections,
            "tol_s": TOL[ds]["primary"], "tp": p["tp"], "fp": p["fp"], "fn": p["fn"], "n_pred": p["n_pred"],
            "n_ref": p["n_ref"], "precision": p["precision"], "recall": p["recall"], "f1": p["f1"],
            "tol_strict_s": TOL[ds]["strict"], "f1_strict": s["f1"], "n_pred_strict": s["n_pred"], "tp_strict": s["tp"]}


# ------------------------------------------------------------------ failure analysis
def _peaks_any(R, vid, mod, ctx, thr=1.0):
    """All local maxima (no spacing rule) of a modality's detection curve with prominence >= thr."""
    from scipy.signal import find_peaks
    e = R.ev(vid, mod, ctx)
    if not e.ok:
        return np.array([]), None
    s = smooth(normalise(e.d, R.D.base.normalise, R.D.base.score_clip), FIXED_SEL[R.D.dataset(vid)]["smooth_window"])
    if not np.isfinite(s).any():
        return np.array([]), s
    filled = np.where(np.isfinite(s), s, np.nanmin(s))
    p, _ = find_peaks(filled, prominence=thr)
    return e.grid[p], s


def failure_cases(R: Runner, vid):
    ds = R.D.dataset(vid)
    tol = TOL[ds]["primary"]
    ref = R.D.refs[vid]
    mods = [m for m in CONTEXT[ds] if R.ev(vid, m, CONTEXT[ds][m]).ok]
    sp_uni = {m: spec(LABEL[m], [m]) for m in mods}
    pred = {m: R.predict(vid, sp_uni[m]) for m in mods}
    fus_sp = spec("CLIP+CLAP mean" if ds == "moviechat" else "All three mean", mods)
    pred["fusion"] = R.predict(vid, fus_sp)
    hit = {k: {ref[j] for _, j, _ in match(p, ref, tol)} for k, p in pred.items()}
    tp_pred = {k: {p[i] for i, _, _ in match(p, ref, tol)} for k, p in pred.items()}
    peaks = {m: _peaks_any(R, vid, m, CONTEXT[ds][m])[0] for m in mods}
    curves = {m: _peaks_any(R, vid, m, CONTEXT[ds][m])[1] for m in mods}
    grid = R.ev(vid, mods[0], CONTEXT[ds][mods[0]]).grid
    best = BASELINE[ds] if BASELINE[ds] in mods else mods[0]
    out = []
    # 1. strongest unimodal finds it, fusion loses it
    for r in sorted(hit[best] - hit["fusion"]):
        out.append((1, r, f"reference {r:.1f}s matched by {LABEL[best]} alone, missed by {fus_sp['name']}"))
    # 6. (complementarity check) fusion recovers a reference the strongest unimodal misses
    for r in sorted(hit["fusion"] - hit[best]):
        out.append((6, r, f"reference {r:.1f}s recovered by {fus_sp['name']}, missed by {LABEL[best]} alone"))
    # 2. audio-only false positives
    if "clap" in pred:
        for p in sorted(set(pred["clap"]) - tp_pred["clap"]):
            out.append((2, p, f"CLAP predicts {p:.0f}s with no reference within ±{tol:g}s"))
    # 3. transcript catches a shift that CLIP misses
    if "sbert" in pred and "clip" in pred:
        for r in sorted(hit["sbert"] - hit["clip"]):
            out.append((3, r, f"reference {r:.1f}s matched by SBERT, missed by CLIP"))
    # 4. no prominent peak in any modality
    for r in ref:
        if not any(len(peaks[m]) and np.min(np.abs(peaks[m] - r)) <= tol for m in mods):
            out.append((4, r, f"reference {r:.1f}s: no peak with prominence >= 1 within ±{tol:g}s in any modality"))
    # 5. fusion false positive created by several moderate signals aligning
    for p in sorted(set(pred["fusion"]) - tp_pred["fusion"]):
        i = int(np.argmin(np.abs(grid - p)))
        lo, hi = max(0, i - int(tol // 2)), i + int(tol // 2) + 1
        moderate = [m for m in mods if curves[m] is not None and np.nanmax(curves[m][lo:hi], initial=-np.inf) >= 0.5]
        own = [m for m in mods if len(pred[m]) and np.min(np.abs(np.asarray(pred[m]) - p)) <= tol]
        if len(moderate) >= 2 and not own:
            out.append((5, p, f"fused prediction {p:.0f}s: {', '.join(LABEL[m] for m in moderate)} all mildly raised "
                              f"(z >= 0.5), none predicts a boundary there alone; no reference within ±{tol:g}s"))
    return out, pred, hit


# ------------------------------------------------------------------ plotting
def diagnostics_plot(R: Runner, vid, path, rel_name):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ds = R.D.dataset(vid)
    tol = TOL[ds]["primary"]
    ref = R.D.refs[vid]
    mods = [m for m in CONTEXT[ds] if R.ev(vid, m, CONTEXT[ds][m]).ok]
    fus_name = "CLIP+CLAP mean" if ds == "moviechat" else "All three mean"
    rel_sp = next(s for s in RELIABILITY[ds] if s["name"] == rel_name)
    rows = [(LABEL[m] + f" ({CONTEXT[ds][m]:g}s)", spec(LABEL[m], [m]), m) for m in mods]
    rows += [(fus_name.replace(" mean", "\nequal-weight mean"), spec(fus_name, mods), "fusion"),
             ("R3 reliability-\nweighted mean", rel_sp, "rel")]
    col = {"clip": "#2563eb", "clap": "#16a34a", "sbert": "#9333ea", "fusion": "#dc2626", "rel": "#ea580c"}
    fig, axes = plt.subplots(len(rows) + 1, 1, figsize=(13, 1.55 * (len(rows) + 1) + 0.6), sharex=True)
    for ax, (lab, sp, key) in zip(axes, rows):
        grid, s = R.score(vid, sp)
        curve = smooth(normalise(s, R.D.base.normalise, R.D.base.score_clip), FIXED_SEL[ds]["smooth_window"])
        pred = R.predict(vid, sp)
        tp_idx = {i for i, _, _ in match(pred, ref, tol)}
        for r in ref:
            ax.axvspan(r - tol, r + tol, color="#111", alpha=0.07, lw=0)
            ax.axvline(r, color="#111", lw=1.1, alpha=0.7)
        ax.plot(grid, curve, color=col[key], lw=1)
        for i, p in enumerate(pred):
            ax.axvline(p, color=col[key], lw=1.4, ls="-" if i in tp_idx else ":", alpha=0.9)
        f = prf(len(tp_idx), len(pred), len(ref))
        ax.set_ylabel(lab, rotation=0, ha="right", va="center", fontsize=9)
        ax.text(1.002, 0.5, f"F1 {f['f1']:.2f}\n{len(tp_idx)}/{len(pred)} pred", transform=ax.transAxes,
                fontsize=8, va="center")
        ax.set_ylim(-3, 5.5)
        ax.grid(alpha=0.2)
    ax = axes[-1]
    for m in mods:
        e = R.ev(vid, m, CONTEXT[ds][m])
        ax.plot(e.grid, e.weight("both"), color=col[m], lw=1, label=f"{LABEL[m]} reliability (coverage x stability)")
    for r in ref:
        ax.axvline(r, color="#111", lw=1.1, alpha=0.7)
    ax.set_ylabel("reliability\nweight", rotation=0, ha="right", va="center", fontsize=9)
    ax.legend(fontsize=7, loc="upper right", ncol=len(mods))
    ax.set_ylim(0, 1.05)
    ax.xaxis.set_major_formatter(lambda x, _p=None: f"{int(x // 60)}:{int(x % 60):02d}")
    ax.set_xlim(0, R.D.dur(vid))
    ax.set_xlabel(f"time (m:ss). Black lines + grey bands = reference boundaries ±{tol:g}s. "
                  "Coloured solid = matched prediction, dotted = false positive. Curves: normalised, smoothed change score.")
    axes[0].set_title(f"{'MovieChat' if ds == 'moviechat' else 'AMI'} {vid}: per-modality and fused change scores "
                      f"(fixed selection: min seg {FIXED_SEL[ds]['min_segment_s']:g}s, prominence 1.0, smoothing 3)",
                      loc="left", fontsize=10)
    fig.tight_layout(rect=(0.02, 0, 0.97, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------ io
def write_csv(path, rows):
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/AMI_ES2002_segmentation_reference")
    a = ap.parse_args()
    data_dir = Path(a.data_dir)
    out = data_dir / "evaluation" / "experiment_C"
    out.mkdir(parents=True, exist_ok=True)
    R = Runner(data_dir)

    async_rows, pv1 = run_table(R, ASYNC, "asynchronous_context")
    write_csv(out / "asynchronous_context_results.csv", async_rows)
    rel_rows, pv2 = run_table(R, RELIABILITY, "reliability_fusion")
    write_csv(out / "reliability_fusion_results.csv", rel_rows)
    held_rows, pv3 = heldout(R, async_rows)
    write_csv(out / "heldout_comparison.csv", held_rows)
    write_csv(out / "per_video_results.csv", pv1 + pv2 + pv3)

    fc = []
    for vid in R.D.items:
        cases, _, _ = failure_cases(R, vid)
        fc += [{"dataset": R.D.dataset(vid), "video_id": vid, "category": c, "time_s": round(t, 2), "detail": d}
               for c, t, d in cases]
    write_csv(out / "failure_cases.csv", fc)
    diagnostics_plot(R, "236", out / "moviechat_fusion_diagnostics.png", "CLIP+CLAP | R3 coverage x stability")
    diagnostics_plot(R, "ES2002a", out / "ami_fusion_diagnostics.png", "All three | R3 coverage x stability")
    params = {"contexts_s": CONTEXT, "fixed_selection": FIXED_SEL, "tolerances_s": TOL, "baseline": BASELINE,
              "fixed_weights": FIXED_W, "prominence_set_for_fold_selection": PROM_SET,
              "silence_dbfs": SILENCE_DBFS, "chance_trials": CHANCE_TRIALS, "base_config": R.D.base.to_dict()}
    (out / "experiment_c_params.json").write_text(json.dumps(params, indent=1), encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
