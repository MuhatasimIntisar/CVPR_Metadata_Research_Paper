"""Level 1: run (or reuse) every detector, keep raw outputs, then build fine-grained observations.

Per step, in order of preference:
  1. existing output in outputs/<id>/level1/raw/ (cached; skipped)
  2. compatible output from an earlier run (paths.legacy_segments_dir / legacy_level1_dir) -> copied in, with
     provenance recorded in level1/raw/provenance.json
  3. run the detector
"""
from __future__ import annotations

import json
import shutil
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ..utils.io import read_json, write_json
from .observations import build

SEGMENTER_FILES = {"silence": "silence.json", "shots": "shots.json", "camera_motion": "camera_motion.json",
                   "audio_events": "audio_events.json", "ocr": "ocr.json", "speakers": "speakers.json"}
L1_FILES = {"yolo": ["yolo_track.json", "yolo_pose.json"], "vad": ["vad.json"],
            "vision_language": ["v2/xclip_v2.npz", "v2/clip_v2.npz", "v2/frame_stats.npz", "v2/overlay_mask.npy",
                                "v2/detect_v2_meta.json"]}


CPU_STEPS = ("shots", "camera_motion")  # OpenCV-only steps, run alongside the GPU steps (camera_motion needs shots)


def outputs_of(step):
    if step in SEGMENTER_FILES:
        return [SEGMENTER_FILES[step]]
    if step == "transcript_words":
        return ["asr.json"]
    if step == "transcript_topics":
        return ["transcript_topics.npz"]
    return [Path(f).name for f in L1_FILES[step]]


def device_of(cfg):
    d = cfg["runtime"]["device"]
    if d != "auto":
        return d
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def legacy_sources(cfg, vid, step):
    p = cfg["paths"]
    if step in SEGMENTER_FILES and p.get("legacy_segments_dir"):
        return [Path(p["legacy_segments_dir"]) / vid / SEGMENTER_FILES[step]]
    if step in L1_FILES and p.get("legacy_level1_dir"):
        return [Path(p["legacy_level1_dir"]) / vid / f for f in L1_FILES[step]]
    return None


def run(cfg, manifest, entry, vdir: Path, log):
    vid = entry["video_id"]
    raw = vdir / "level1" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    src = read_json(vdir / "source" / "source_manifest.json")
    transcript = read_json(vdir / "source" / "transcript.json")
    if src is None or transcript is None:
        raise RuntimeError("run the ingest stage first")
    # resolve against the CURRENT corpus root (portable across machines), not the path stored at ingestion
    root = Path(cfg["paths"].get("corpus_root") or manifest["corpus_root"])
    video = root / entry["item_dir"] / src["inputs"]["video"]
    if not video.exists():
        raise FileNotFoundError(f"video not found: {video} (check paths.corpus_root in your config)")
    prov = read_json(raw / "provenance.json", {})
    device = device_of(cfg)
    wav = vdir / "level1" / "_cache" / "audio_16k.wav"
    from .adapters.segmenters.common import probe
    info = probe(video)
    lock = threading.Lock()

    def save(step, rec):
        with lock:  # both branches update provenance.json
            prov[step] = rec
            write_json(raw / "provenance.json", prov)

    def do_step(step):
        outs = outputs_of(step)
        if all((raw / o).exists() for o in outs):
            log.info(f"{step}: cached")
            return
        leg = legacy_sources(cfg, vid, step)
        if leg and all(p.exists() for p in leg):
            for p in leg:
                shutil.copy2(p, raw / p.name)
            log.info(f"{step}: reused earlier output ({leg[0].parent})")
            save(step, {"source": "reused earlier output", "files": [str(p) for p in leg]})
            return
        t0 = time.time()
        if step in ("silence", "audio_events", "speakers", "vad") and not wav.exists():
            wav.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(video), "-vn", "-ac", "1", "-ar", "16000",
                            "-c:a", "pcm_s16le", str(wav)], check=True)
        dev = "cpu" if step in CPU_STEPS else device
        log.info(f"{step}: running on {dev}")
        if step == "transcript_words":
            from .adapters.visual import transcript_words
            meta = transcript_words(raw, transcript)
        elif step == "transcript_topics":
            from .adapters.topics import run as run_topics
            meta = run_topics(raw, transcript, cfg, device)
        elif step == "yolo":
            from .adapters.visual import run_yolo
            meta = run_yolo(video, raw, cfg, device)
        elif step == "vision_language":
            from .adapters.visual import run_vision_language
            meta = run_vision_language(video, raw, cfg, device)
        elif step == "vad":
            from .adapters.visual import run_vad
            meta = run_vad(raw, wav)
        else:
            import importlib
            mod = importlib.import_module(f".adapters.segmenters.{step}", __package__)
            langs = cfg["level1"]["ocr"]["langs"]
            if langs == "auto":
                code = (read_json(vdir / "source" / "source_metadata.json") or {}).get("language_code") or "en"
                langs = "en" if code == "en" else f"{code},en"
            kw = {"wav": wav, "device": dev}
            if step == "ocr":
                kw.update(langs=langs, every_s=cfg["level1"]["ocr"]["every_s"])
            mod.run(video, raw, info, **kw)
            meta = {"runtime_s": round(time.time() - t0, 1)}
        save(step, {"source": "computed", "device": dev, **(meta or {})})
        log.info(f"{step}: done in {time.time() - t0:.0f} s")

    steps = list(cfg["level1"]["steps"])
    # the CPU-only video steps (shot cuts, then optical flow, which needs the cuts) run in a background thread while
    # the GPU/audio steps run here; OpenCV releases the GIL, so the two branches overlap
    cpu = [s for s in steps if s in CPU_STEPS]
    rest = [s for s in steps if s not in CPU_STEPS]
    t_all = time.time()
    with ThreadPoolExecutor(1) as ex:
        fut = ex.submit(lambda: [do_step(s) for s in cpu])
        for step in rest:
            do_step(step)
        fut.result()
    log.info(f"detectors done in {time.time() - t_all:.0f} s")
    t0 = time.time()
    n_obs, n_win = build(cfg, vdir, transcript)
    prov["observations"] = {"n_steps": n_obs, "n_windows": n_win, "runtime_s": round(time.time() - t0, 1)}
    write_json(raw / "provenance.json", prov)
    log.info(f"observations: {n_obs} steps of {cfg['level1']['step_s']} s, {n_win} windows of {cfg['level1']['window_s']} s")
    (vdir / "level1" / "README.txt").write_text(
        "raw/: unmodified detector outputs (see raw/provenance.json)\nobservations.jsonl: 2 s evidence steps\n"
        "window_summaries.jsonl: 15 s summaries (observation mechanism, not final intervals)\n", encoding="utf-8")
    return json.dumps(prov)
