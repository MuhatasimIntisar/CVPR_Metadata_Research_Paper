"""Training-free multimodal segmentation pilot: PySceneDetect vs CLIP vs CLAP vs Sentence-BERT vs late
fusion, evaluated against MovieChat-SSeg scene boundaries and AMI topic boundaries (separately).

Run from the repo root:
    python -m segmentation.run_experiment                          # validate .. report (default steps)
    python -m segmentation.run_experiment --steps validate         # check inputs/timestamps only
    python -m segmentation.run_experiment --steps detect evaluate plot report   # re-tune w/o inference
    python -m segmentation.run_experiment --steps transcribe       # optional Whisper for MovieChat
    python -m segmentation.run_experiment --config my_params.json  # override Config fields

Steps: validate, transcribe (optional), scenedetect, encode, detect, evaluate, plot, report.
Expensive steps (scenedetect, encode, transcribe) are cached; detect/evaluate/plot/report reuse the
cache and take seconds. Source files are never modified; outputs go to cache/, predictions/,
evaluation/ and plots/ inside the data folder.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from segmentation import audio_encoder, text_encoder, visual_encoder  # noqa: E402
from segmentation.boundary_detection import detect  # noqa: E402
from segmentation.change_scores import make_grid, modality_curves  # noqa: E402
from segmentation.config import DEFAULT_DATA_DIR, Config, discover, out_dirs, reference_boundaries  # noqa: E402
from segmentation.evaluation import micro, score_video  # noqa: E402
from segmentation.fusion import fuse, normalise  # noqa: E402
from segmentation.media_utils import cache_paths, extract_audio, ffprobe, get_device, load_cache  # noqa: E402
from segmentation.methods import BY_ID, METHODS  # noqa: E402

DEFAULT_STEPS = ["validate", "scenedetect", "encode", "detect", "evaluate", "plot", "report"]
ALL_STEPS = ["validate", "transcribe", "scenedetect", "encode", "detect", "evaluate", "plot", "report"]


def jdump(path: Path, obj):
    path.write_text(json.dumps(obj, indent=1, ensure_ascii=False), encoding="utf-8")


def jload(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------- 1. validate
def validate(items, cfg, dirs):
    report = []
    print(f"{'video':10} {'set':9} {'video_s':>8} {'audio_s':>8} {'fps':>5} {'res':>9} {'refs':>4} "
          f"{'ref range':>16} {'transcript':>22}  warnings")
    for it in items:
        w = []
        vi = ffprobe(it.video) if it.video else None
        ai = ffprobe(it.audio) if it.audio else None
        v_dur = vi["video"]["duration"] if vi and vi.get("video") else None
        if it.audio:
            a_dur = (ai.get("audio") or {}).get("duration") if ai else None
        else:
            a_dur = ((vi or {}).get("audio") or {}).get("duration") if vi else None
            if vi and not vi.get("audio"):
                w.append("video has no audio stream (CLAP unavailable)")
        ref = reference_boundaries(it)
        tr = None
        if it.transcript:
            words, _ = text_encoder.load_words(it, dirs)
            tr = {"words": len(words), "span": [words[0]["start"], words[-1]["end"]],
                  "speakers": sorted({x["speaker"] for x in words if x["speaker"]})}
        elif (dirs["cache"] / f"{it.video_id}_asr_words.json").exists():
            tr = {"source": "whisper_asr"}
        durs = [d for d in (v_dur, a_dur) if d]
        analysis = max(durs) if durs else 0.0
        if v_dur and a_dur and abs(v_dur - a_dur) > 1.0:
            w.append(f"video/audio length differ by {a_dur - v_dur:+.1f}s (assumed common start t=0; "
                     f"visual evidence ends at {v_dur:.1f}s)")
        if ref and ref[-1] > min(durs or [0]):
            w.append(f"reference boundary at {ref[-1]:.1f}s beyond shortest stream")
        if tr and tr.get("span") and a_dur and tr["span"][1] > a_dur + 1:
            w.append("transcript extends past audio end")
        if it.dataset == "moviechat" and vi and vi.get("video"):
            v = vi["video"]
            if abs(v["fps"] - 25.0) > 0.01 or (v.get("frames") and v["frames"] != 12000):
                w.append(f"expected 25 fps / 12000 frames, got {v['fps']} fps / {v.get('frames')}")
        rec = {"video_id": it.video_id, "dataset": it.dataset, "files": it.to_dict(), "video_duration_s": v_dur,
               "audio_duration_s": a_dur, "analysis_duration_s": analysis,
               "fps": (vi or {}).get("video", {}).get("fps") if vi else None,
               "resolution": f"{vi['video']['width']}x{vi['video']['height']}" if vi and vi.get("video") else None,
               "n_reference_boundaries": len(ref), "reference_range_s": [ref[0], ref[-1]] if ref else None,
               "transcript": tr, "warnings": w}
        report.append(rec)
        trs = (f"{tr['words']}w {tr['span'][0]:.0f}-{tr['span'][1]:.0f}s" if tr and "words" in tr
               else (tr or {}).get("source", "none"))
        print(f"{it.video_id:10} {it.dataset:9} {v_dur or 0:8.1f} {a_dur or 0:8.1f} {rec['fps'] or 0:5.1f} "
              f"{rec['resolution'] or '-':>9} {len(ref):4} {str([round(x) for x in rec['reference_range_s'] or []]):>16} "
              f"{trs:>22}  {'; '.join(w) or 'ok'}")
    jdump(dirs["evaluation"] / "validation.json", report)
    return {r["video_id"]: r for r in report}


def durations(dirs):
    p = dirs["evaluation"] / "validation.json"
    if not p.exists():
        sys.exit("Run --steps validate first.")
    return {r["video_id"]: r for r in jload(p)}


# ---------------------------------------------------------------- 2. transcribe (optional)
def transcribe(items, cfg, dirs, device, force):
    from src.segmenters import asr  # reuse the segmented Whisper ASR from the main pipeline
    for it in items:
        if it.transcript is not None:
            continue
        out = dirs["cache"] / f"{it.video_id}_asr_words.json"
        if out.exists() and not force:
            print(f"[skip] {it.video_id}: Whisper transcript cached")
            continue
        print(f"[asr ] {it.video_id} ...", flush=True)
        wav = extract_audio(it.video, dirs["cache"] / f"{it.video_id}_audio16k.wav", 16000)
        work = dirs["cache"] / f"asr_{it.video_id}"
        work.mkdir(exist_ok=True)
        info = ffprobe(it.video)
        asr.run(it.video, work, {"duration": info["duration"]}, wav=wav, device=device)
        doc = jload(work / "asr.json")
        jdump(out, [{"start": w["start"], "end": w["end"], "text": w["w"]} for w in doc["words"]])
        jdump(dirs["cache"] / f"{it.video_id}_asr_meta.json",
              {"model": doc["model"], "params": doc["params"], "stats": doc["stats"], "runtime_s": doc["runtime_s"]})
        print(f"       {doc['stats']['n_words']} words, language {doc['stats']['language']}")


# ---------------------------------------------------------------- 3. scenedetect (M1)
def run_scenedetect(items, cfg, dirs, force):
    from segmentation import scenedetect_baseline
    for it in items:
        out = dirs["predictions"] / f"{it.video_id}__M1_pyscenedetect.json"
        if out.exists() and not force:
            print(f"[skip] {it.video_id}: PySceneDetect cached")
            continue
        r = scenedetect_baseline.run(it.video, cfg.psd_threshold, cfg.psd_min_scene_len)
        jdump(out, {"video_id": it.video_id, "dataset": it.dataset, "method": "M1_pyscenedetect",
                    "label": BY_ID["M1_pyscenedetect"].label, "status": "ok", **r})
        print(f"[psd ] {it.video_id}: {len(r['boundaries_seconds'])} cuts in {r['runtime_s']}s")


# ---------------------------------------------------------------- 4. encode
def encode(items, cfg, dirs, device, force, modalities, val):
    for it in items:
        dur = val[it.video_id]["analysis_duration_s"]
        for name, fn in (("visual", lambda: visual_encoder.encode(it, cfg, dirs, device, force)),
                         ("audio", lambda: audio_encoder.encode(it, cfg, dirs, device, force)),
                         ("text", lambda: text_encoder.encode(it, cfg, dirs, device, dur, force))):
            if name not in modalities:
                continue
            cached = all(p.exists() for p in cache_paths(dirs["cache"], it.video_id,
                                                         {"visual": visual_encoder.NAME, "audio": audio_encoder.NAME, "text": text_encoder.NAME}[name]))
            t0 = time.time()
            _, meta = fn()
            took = time.time() - t0
            state = meta.get("status")
            extra = (f"n={meta.get('n')} ({'cache' if cached and not force else f'{took:.0f}s'})"
                     if state == "ok" else meta.get("reason"))
            print(f"[enc ] {it.video_id:8} {name:6} {state:12} {extra}", flush=True)


# ---------------------------------------------------------------- 5. detect (M2-M5 + ablations)
def load_caches(it, dirs):
    """Cached embeddings + metadata per modality (no inference)."""
    caches, metas = {}, {}
    for mod, name in (("visual", visual_encoder.NAME), ("audio", audio_encoder.NAME), ("text", text_encoder.NAME)):
        a, m = load_cache(dirs["cache"], it.video_id, name)
        if a is not None:
            caches[mod], metas[mod] = a, m
    asr_meta = dirs["cache"] / f"{it.video_id}_asr_meta.json"
    if "text" in metas and metas["text"]["params"].get("transcript_source") == "whisper_asr" and asr_meta.exists():
        metas["text"] = dict(metas["text"], runtime_s=metas["text"]["runtime_s"] + jload(asr_meta)["runtime_s"])
    return caches, metas


def curves_for(it, cfg, dirs, dur):
    grid = make_grid(dur, cfg.grid_s)
    caches, metas = load_caches(it, dirs)
    return grid, modality_curves(caches, grid, cfg), metas


def run_detect(items, cfg, dirs, val):
    for it in items:
        dur = val[it.video_id]["analysis_duration_s"]
        grid, curves, metas = curves_for(it, cfg, dirs, dur)
        np.savez_compressed(dirs["predictions"] / f"{it.video_id}__curves.npz", grid=grid,
                            **{k: v for k, v in curves.items()})
        for meth in METHODS:
            if meth.id == "M1_pyscenedetect":
                continue
            avail = [mm for mm in meth.modalities if mm in curves]
            missing = [mm for mm in meth.modalities if mm not in curves]
            rec = {"video_id": it.video_id, "dataset": it.dataset, "method": meth.id, "label": meth.label,
                   "group": meth.group}
            if (meth.require_all and missing) or not avail:
                rec.update(status="not_available", missing_modalities=missing, boundaries_seconds=[])
            else:
                t0 = time.time()
                if meth.fusion is None:
                    score = curves[avail[0]]
                else:
                    score, _ = fuse({k: curves[k] for k in avail}, cfg.fusion_weights, meth.fusion, cfg.normalise,
                                    cfg.score_clip)
                pred, _ = detect(grid, score, cfg, dur)
                runtime = sum(metas[k]["runtime_s"] for k in avail) + (time.time() - t0)
                rec.update(status="ok", modalities_used=avail, missing_modalities=missing,
                           boundaries_seconds=[round(p, 2) for p in pred], runtime_s=round(runtime, 2),
                           realtime_factor=round(runtime / dur, 4),
                           transcript_source=metas.get("text", {}).get("params", {}).get("transcript_source"))
            jdump(dirs["predictions"] / f"{it.video_id}__{meth.id}.json", rec)
        ok = sorted(curves)
        print(f"[det ] {it.video_id:8} modalities with evidence: {', '.join(ok) or 'none'}")
    jdump(dirs["evaluation"] / "config_used.json", cfg.to_dict())


# ---------------------------------------------------------------- 6. evaluate
def run_evaluate(items, cfg, dirs, val):
    per_video, summary = [], []
    for ds in sorted({it.dataset for it in items}):
        ds_items = [it for it in items if it.dataset == ds]
        for meth in METHODS:
            for tol in cfg.tolerances[ds]:
                rows = []
                for it in ds_items:
                    p = dirs["predictions"] / f"{it.video_id}__{meth.id}.json"
                    if not p.exists():
                        continue
                    pr = jload(p)
                    dur = val[it.video_id]["analysis_duration_s"]
                    base = {"dataset": ds, "video_id": it.video_id, "method": meth.id, "label": meth.label,
                            "group": meth.group, "tolerance_s": tol, "status": pr["status"],
                            "split": "dev" if it.video_id in cfg.dev_videos else "test"}
                    if pr["status"] != "ok":
                        per_video.append({**base, "missing_modalities": "+".join(pr.get("missing_modalities", []))})
                        continue
                    s = score_video(pr["boundaries_seconds"], reference_boundaries(it), tol)
                    rt = pr.get("runtime_s")
                    row = {**base, **s, "runtime_s": rt, "realtime_factor": round(rt / dur, 4) if rt else None,
                           "modalities_used": "+".join(pr.get("modalities_used", [])) or "video",
                           "transcript_source": pr.get("transcript_source")}
                    rows.append(row)
                    per_video.append(row)
                if rows:
                    m = micro(rows)
                    summary.append({"dataset": ds, "method": meth.id, "label": meth.label, "group": meth.group,
                                    "tolerance_s": tol, "n_videos": len(rows), **m,
                                    "runtime_s_total": round(sum(r["runtime_s"] or 0 for r in rows), 1),
                                    "realtime_factor_mean": round(float(np.mean([r["realtime_factor"] or 0 for r in rows])), 4),
                                    "modalities_used": sorted({r["modalities_used"] for r in rows})})
                else:
                    summary.append({"dataset": ds, "method": meth.id, "label": meth.label, "group": meth.group,
                                    "tolerance_s": tol, "n_videos": 0, "status": "not_available"})
    write_csv(dirs["evaluation"] / "per_video.csv", per_video)
    write_csv(dirs["evaluation"] / "dataset_summary.csv", summary)
    md = summary_markdown(summary, per_video, cfg)
    (dirs["evaluation"] / "results.md").write_text(md, encoding="utf-8")
    print(md)
    return summary, per_video


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
            w.writerow({k: ("; ".join(map(str, v)) if isinstance(v, list) else v) for k, v in r.items()})


def summary_markdown(summary, per_video, cfg) -> str:
    out = []
    names = {"moviechat": "MovieChat-SSeg (semantic scene boundaries)", "ami": "AMI (topic boundaries, oracle transcript)"}
    for ds in sorted({s["dataset"] for s in summary}):
        tols = cfg.tolerances[ds]
        out.append(f"### {names.get(ds, ds)}\n")
        out.append("| Method | " + " | ".join(f"P / R / F1 @±{t:g}s" for t in tols) + " | Pred / Ref | Runtime (s) | RTF |")
        out.append("|---|" + "---|" * len(tols) + "---|---|---|")
        for meth in METHODS:
            rows = [s for s in summary if s["dataset"] == ds and s["method"] == meth.id]
            if not rows or rows[0].get("n_videos", 0) == 0:
                out.append(f"| {meth.label} | " + " | ".join("N/A" for _ in tols) + " | N/A | N/A | N/A |")
                continue
            cells = []
            for t in tols:
                r = next(x for x in rows if x["tolerance_s"] == t)
                cells.append(f"{r['precision']:.2f} / {r['recall']:.2f} / **{r['f1']:.2f}**")
            r0 = rows[0]
            out.append(f"| {meth.label} | " + " | ".join(cells) +
                       f" | {r0['n_pred']} / {r0['n_ref']} | {r0['runtime_s_total']} | {r0['realtime_factor_mean']} |")
        out.append("")
        out.append(f"Per video (F1 @±{tols[-1]:g}s):\n")
        vids = sorted({p["video_id"] for p in per_video if p["dataset"] == ds})
        out.append("| Method | " + " | ".join(vids) + " |")
        out.append("|---|" + "---|" * len(vids))
        for meth in METHODS:
            cells = []
            for v in vids:
                r = next((p for p in per_video if p["dataset"] == ds and p["video_id"] == v and p["method"] == meth.id
                          and p["tolerance_s"] == tols[-1]), None)
                cells.append("N/A" if not r or r["status"] != "ok" else f"{r['f1']:.2f} ({r['n_pred']})")
            out.append(f"| {meth.label} | " + " | ".join(cells) + " |")
        out.append("\nCells: F1 (number of predicted boundaries). N/A = a required modality is unavailable; not scored.\n")
    return "\n".join(out)


# ---------------------------------------------------------------- 7. plot
def run_plot(items, cfg, dirs, val):
    from segmentation import plots
    labels = {m.id: m.label for m in METHODS}
    labels.update(visual="CLIP (visual)", audio="CLAP (audio)", text="SBERT (speech)", fused="Fused (mean)")
    for it in items:
        dur = val[it.video_id]["analysis_duration_s"]
        ref = reference_boundaries(it)
        preds = {}
        for mid in ("M1_pyscenedetect", "M2_clip", "M3_clap", "M4_sbert", "M5_fusion_mean", "M5_fusion_max"):
            p = dirs["predictions"] / f"{it.video_id}__{mid}.json"
            if p.exists():
                pr = jload(p)
                if pr["status"] == "ok":
                    preds[mid] = pr["boundaries_seconds"]
        cz = dirs["predictions"] / f"{it.video_id}__curves.npz"
        if not cz.exists():
            continue
        z = np.load(cz)
        grid = z["grid"]
        curves = {k: z[k] for k in z.files if k != "grid"}
        fused = None
        if curves:
            fused, _ = fuse(curves, cfg.fusion_weights, "mean", cfg.normalise, cfg.score_clip)
        plots.boundary_timeline(dirs["plots"] / f"{it.video_id}_boundaries.png",
                                f"{it.video_id} ({it.dataset}) - predicted vs human boundaries", dur, ref, preds,
                                labels, fused, grid)
        normed = {k: normalise(v, cfg.normalise, cfg.score_clip) for k, v in curves.items()}
        for k in ("visual", "audio", "text"):
            normed.setdefault(k, np.full(len(grid), np.nan))
        plots.score_panels(dirs["plots"] / f"{it.video_id}_scores.png",
                           f"{it.video_id} ({it.dataset}) - normalised change scores", dur, ref, grid,
                           {k: normed[k] for k in ("visual", "audio", "text")}, labels,
                           normalise(fused, cfg.normalise, cfg.score_clip) if fused is not None else np.full(len(grid), np.nan),
                           preds.get("M5_fusion_mean", []),
                           colors={"visual": "#2563eb", "audio": "#16a34a", "text": "#9333ea"})
        print(f"[plot] {it.video_id}: {it.video_id}_boundaries.png, {it.video_id}_scores.png")


# ---------------------------------------------------------------- 8. report
def run_report(items, cfg, dirs, val, data_dir):
    from segmentation.report import write_report
    path = write_report(items, cfg, dirs, val, data_dir)
    print(f"[rep ] {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    ap.add_argument("--steps", nargs="+", choices=ALL_STEPS + ["all"], default=DEFAULT_STEPS)
    ap.add_argument("--videos", nargs="+", help="only these video ids")
    ap.add_argument("--modalities", nargs="+", default=["visual", "audio", "text"], choices=["visual", "audio", "text"])
    ap.add_argument("--config", type=Path, help="JSON file overriding Config fields")
    ap.add_argument("--device", default=None)
    ap.add_argument("--force", action="store_true", help="recompute cached steps")
    args = ap.parse_args()

    cfg = Config.from_file(args.config)
    if args.device:
        cfg.device = args.device
    steps = DEFAULT_STEPS if "all" in args.steps else args.steps
    dirs = out_dirs(args.data_dir)
    all_items = discover(args.data_dir)
    items = [i for i in all_items if not args.videos or i.video_id in args.videos]
    if not items:
        sys.exit(f"No videos found in {args.data_dir}")
    device = get_device(cfg.device)
    print(f"{len(items)} videos ({', '.join(i.video_id for i in items)}), device={device}, steps={steps}\n")
    with open(dirs["evaluation"] / "runs_log.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"time": datetime.now().isoformat(timespec="seconds"), "steps": steps,
                            "videos": [i.video_id for i in items], "config": cfg.to_dict()}) + "\n")

    need_val = "validate" in steps or not (dirs["evaluation"] / "validation.json").exists()
    val = validate(all_items, cfg, dirs) if need_val else durations(dirs)  # cheap; always covers all videos
    if "transcribe" in steps:
        transcribe(items, cfg, dirs, device, args.force)
    if "scenedetect" in steps:
        run_scenedetect(items, cfg, dirs, args.force)
    if "encode" in steps:
        encode(items, cfg, dirs, device, args.force, args.modalities, val)
    if "detect" in steps:
        run_detect(items, cfg, dirs, val)
    if "evaluate" in steps:  # evaluate/report always cover every video that has predictions
        run_evaluate(all_items, cfg, dirs, val)
    if "plot" in steps:
        run_plot(items, cfg, dirs, val)
    if "report" in steps:
        run_report(all_items, cfg, dirs, val, args.data_dir)


if __name__ == "__main__":
    main()
