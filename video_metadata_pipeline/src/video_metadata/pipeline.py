"""Stage orchestration with per-video error isolation, checkpoints and resume."""
from __future__ import annotations

import time
import traceback
from pathlib import Path

from . import PIPELINE_VERSION
from .config import video_dir
from .state import State
from .utils.io import fingerprint, read_json
from .utils.logging import stage_logger

STAGES = ["ingest", "level1", "correction", "level2", "review"]
AUTOMATIC = ["ingest", "level1", "correction", "level2"]


def corpus_root(cfg, manifest):
    """The config wins over the manifest, so a manifest built on one machine works on another."""
    return Path(cfg["paths"].get("corpus_root") or manifest["corpus_root"])


def run_stage(cfg, manifest, entry, stage, force=False):
    vid = entry["video_id"]
    vdir = video_dir(cfg, vid)
    st = State(vdir)
    log = stage_logger(vdir / "logs", vid, stage)
    fp = fingerprint({"stage": stage, "cfg": cfg.get(stage if stage != "ingest" else "ingestion"),
                      "pipeline": PIPELINE_VERSION})
    if not force and st.done(stage, fp):
        log.info("cached (done, same configuration) - skipping")
        return "cached"
    st.mark(stage, "running")
    t0 = time.time()
    try:
        if stage == "ingest":
            from .ingestion.manifest import ingest_video
            rep = ingest_video(cfg, entry, corpus_root(cfg, manifest), vdir / "source")
            for w in rep["warnings"]:
                log.warning(w)
            if rep["status"] == "failed":
                raise RuntimeError("; ".join(rep["issues"] + rep["warnings"]) or "ingestion failed")
        elif stage == "level1":
            from .level1.pipeline import run as run_l1
            run_l1(cfg, manifest, entry, vdir, log)
        elif stage == "correction":
            from .correction.moment_proposals import run as run_corr
            run_corr(cfg, entry, vdir, log)
        elif stage == "level2":
            from .level2.pipeline import run as run_l2
            run_l2(cfg, manifest, entry, vdir, log)
        else:
            raise NotImplementedError(f"stage '{stage}' is not implemented yet (review: checkpoint 3)")
        st.mark(stage, "done", fingerprint=fp, runtime_s=round(time.time() - t0, 1))
        st.invalidate_from(stage, STAGES)
        log.info(f"done in {time.time() - t0:.1f} s")
        return "done"
    except Exception as e:  # isolate failures per video
        st.mark(stage, "failed", error=f"{type(e).__name__}: {e}", runtime_s=round(time.time() - t0, 1))
        log.error(f"FAILED: {type(e).__name__}: {e}\n{traceback.format_exc(limit=4)}")
        return "failed"


def run(cfg, manifest_path, stages, video_ids=None, force=False):
    manifest = read_json(manifest_path)
    if manifest is None:
        raise SystemExit(f"manifest not found: {manifest_path} (run `ingest` first)")
    videos = [e for e in manifest["videos"] if e["status"] == "ok" and not (video_ids and e["video_id"] not in video_ids)]
    results = {e["video_id"]: {} for e in videos}
    # stage-major: every video finishes a stage before the next stage starts, so each stage's models are loaded once
    # per job (the Level-2 engine stays on the GPU for all videos and never competes with Level-1 detectors)
    for s in [x for x in STAGES if x in stages]:
        for e in videos:
            if "failed" in results[e["video_id"]].values():
                continue  # later stages of this video depend on the failed one; other videos continue
            results[e["video_id"]][s] = run_stage(cfg, manifest, e, s, force)
    return results


def status(cfg, manifest_path):
    manifest = read_json(manifest_path)
    rows = []
    for e in manifest["videos"]:
        st = State(video_dir(cfg, e["video_id"])).doc["stages"] if e["status"] == "ok" else {}
        rows.append((e["video_id"], e["status"], {s: st.get(s, {}).get("status", "-") for s in STAGES},
                     {s: st[s].get("error") for s in st if st[s].get("error")}))
    return rows
