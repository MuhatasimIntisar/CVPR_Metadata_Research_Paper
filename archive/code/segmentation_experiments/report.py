"""Short Markdown experiment summary (evaluation/report.md). Observations are generated from the
numbers only; interpretation is left to a 'Discussion' section to be written by hand."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from .media_utils import load_cache
from .methods import METHODS


def _read_csv(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def write_report(items, cfg, dirs, val, data_dir: Path) -> Path:
    ev = dirs["evaluation"]
    summary = _read_csv(ev / "dataset_summary.csv") if (ev / "dataset_summary.csv").exists() else []
    results_md = (ev / "results.md").read_text(encoding="utf-8") if (ev / "results.md").exists() else "_Run --steps evaluate._"
    L = ["# Training-free multimodal segmentation: pilot results", ""]
    L += ["## Objective",
          "Test whether lightweight pretrained visual (CLIP), acoustic (CLAP) and linguistic (Sentence-BERT) "
          "representations, combined by late fusion with no training, find useful temporal boundaries in both "
          "edited footage and visually static, speech-heavy meetings, compared with PySceneDetect and "
          "single-modality baselines. This is a feasibility test on a handful of videos, not a benchmark.", ""]

    # models
    L += ["## Models and parameters", "", "| Component | Model / setting | Revision |", "|---|---|---|"]
    revs = {}
    for it in items:
        for name in ("clip", "clap", "text_chunks"):
            _, m = load_cache(dirs["cache"], it.video_id, name)
            if m and m.get("model_revision"):
                revs[name] = m["model_revision"][:10]
    L += [f"| M1 PySceneDetect | ContentDetector, threshold {cfg.psd_threshold}, min scene {cfg.psd_min_scene_len} frames (defaults) | - |",
          f"| M2 visual | `{cfg.clip_model}`, {cfg.visual_fps:g} fps, context ±{cfg.visual_context_s:g}s | {revs.get('clip', '-')} |",
          f"| M3 audio | `{cfg.clap_model}` (LAION-CLAP HTSAT), {cfg.audio_window_s:g}s windows / {cfg.audio_hop_s:g}s hop, {cfg.audio_sr} Hz mono, context ±{cfg.audio_context_s:g}s | {revs.get('clap', '-')} |",
          f"| M4 speech | `{cfg.text_model}`, {cfg.text_chunk_s:g}s chunks every {cfg.text_chunk_hop_s:g}s, context ±{cfg.text_context_s:g}s, ≥{cfg.text_min_words} words per side | - |",
          f"| M5 fusion | robust-z per modality, availability-aware {'/'.join(['mean', 'max'])}, weights {cfg.fusion_weights} | - |",
          f"| Boundary detection (M2-M5) | {cfg.grid_s:g}s grid, {cfg.normalise} clipped ±{cfg.score_clip:g}, moving average {cfg.smooth_window}, "
          f"find_peaks prominence {cfg.peak_prominence}, min segment {cfg.min_segment_s:g}s | - |",
          f"| Tolerances | MovieChat ±{', ±'.join(f'{t:g}' for t in cfg.tolerances['moviechat'])}s; AMI ±{', ±'.join(f'{t:g}' for t in cfg.tolerances['ami'])}s (provisional) | - |",
          "", f"No parameter was tuned on reference boundaries{'' if not cfg.dev_videos else ' outside the dev subset ' + str(cfg.dev_videos)}. "
          "Every run's configuration is logged in `evaluation/runs_log.jsonl`.", ""]

    # data
    L += ["## Data", "", "| Video | Dataset | Video (s) | Audio (s) | Resolution | Reference boundaries | Transcript | Notes |",
          "|---|---|---|---|---|---|---|---|"]
    for it in items:
        v = val.get(it.video_id, {})
        tr = v.get("transcript") or {}
        trs = (f"manual, {tr['words']} words" if "words" in tr else ("Whisper ASR" if tr.get("source") else "none"))
        L.append(f"| {it.video_id} | {'MovieChat-SSeg' if it.dataset == 'moviechat' else 'AMI'} | "
                 f"{(v.get('video_duration_s') or 0):.0f} | {(v.get('audio_duration_s') or 0):.0f} | {v.get('resolution')} | "
                 f"{v.get('n_reference_boundaries')} | {trs} | {'; '.join(v.get('warnings', [])) or ''} |")
    L += ["", "MovieChat references are semantic scene boundaries; AMI references are top-level conversational "
          "topic boundaries. The two are evaluated separately and never pooled. AMI results use the manual "
          "(oracle) transcript and are not end-to-end ASR results.", ""]

    L += ["## Results", "", "Micro-averaged over videos within each dataset (matched-boundary counts summed).", "",
          results_md, ""]

    # observations
    L += ["## Observations (generated from the numbers)", ""]
    for ds in sorted({s["dataset"] for s in summary}):
        tol = max(cfg.tolerances[ds])
        rows = [s for s in summary if s["dataset"] == ds and _f(s.get("tolerance_s")) == tol and s.get("f1")]
        if not rows:
            continue
        best = max(rows, key=lambda s: _f(s["f1"]))
        psd = next((s for s in rows if s["method"] == "M1_pyscenedetect"), None)
        fus = next((s for s in rows if s["method"] == "M5_fusion_mean"), None)
        name = "MovieChat" if ds == "moviechat" else "AMI"
        L.append(f"- **{name} @±{tol:g}s:** best F1 {float(best['f1']):.2f} from {best['label']}.")
        if psd and fus:
            L.append(f"  - PySceneDetect F1 {float(psd['f1']):.2f} with {psd['n_pred']} predictions for {psd['n_ref']} "
                     f"references; mean fusion F1 {float(fus['f1']):.2f} with {fus['n_pred']} predictions.")
        for s in rows:
            n_pred, n_ref = int(s["n_pred"]), int(s["n_ref"])
            if n_ref and (n_pred > 3 * n_ref or n_pred < n_ref / 3):
                L.append(f"  - {s['label']}: {n_pred} predictions vs {n_ref} references "
                         f"({'over' if n_pred > n_ref else 'under'}-segmentation).")
    na = [s for s in summary if s.get("status") == "not_available"]
    if na:
        L.append("- Not scored (modality unavailable): " + ", ".join(sorted({f"{s['label']} on {s['dataset']}" for s in na})) + ".")
    L.append("")

    # cost
    L += ["## Computational cost", "", "| Modality | Mean real-time factor | Total time (s) |", "|---|---|---|"]
    for name, label in (("clip", "CLIP"), ("clap", "CLAP"), ("text_chunks", "Sentence-BERT")):
        rts, tot = [], 0.0
        for it in items:
            _, m = load_cache(dirs["cache"], it.video_id, name)
            if m and m.get("runtime_s") is not None:
                d = val.get(it.video_id, {}).get("analysis_duration_s") or 1
                rts.append(m["runtime_s"] / d)
                tot += m["runtime_s"]
        if rts:
            L.append(f"| {label} | {sum(rts) / len(rts):.4f} | {tot:.1f} |")
    L += ["", "Real-time factor = processing seconds per second of video (encoding; fusion and peak picking add "
          "milliseconds because they reuse cached embeddings).", ""]

    # plots
    L += ["## Example plots", ""]
    for ds, title in (("moviechat", "MovieChat"), ("ami", "AMI")):
        ex = next((it for it in items if it.dataset == ds and (dirs["plots"] / f"{it.video_id}_boundaries.png").exists()), None)
        if ex:
            L += [f"**{title} - {ex.video_id}: boundaries**", "", f"![{ex.video_id} boundaries](../plots/{ex.video_id}_boundaries.png)", "",
                  f"**{title} - {ex.video_id}: change scores**", "", f"![{ex.video_id} scores](../plots/{ex.video_id}_scores.png)", ""]
    L += ["## Caveats", "",
          "- 5 MovieChat videos and 2 AMI meetings: differences of one or two matched boundaries move F1 noticeably.",
          "- Fixed initial parameters, not tuned; tolerances are provisional and reported explicitly.",
          "- AMI video is shorter than its audio; streams are assumed to share t = 0 (see Data notes).",
          "- No claim of general superiority follows from this pilot.", "",
          "## Discussion", "", "_To be written after inspecting the plots: where each modality helps or fails, "
          "and whether one fixed procedure works on both edited and speech-dominated video._", ""]
    path = ev / "report.md"
    path.write_text("\n".join(L), encoding="utf-8")
    (ev / "report_inputs.json").write_text(json.dumps({"config": cfg.to_dict()}, indent=1), encoding="utf-8")
    return path


__all__ = ["write_report", "METHODS"]
