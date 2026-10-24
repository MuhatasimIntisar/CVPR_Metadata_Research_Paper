"""Level 2 stage: semantic boundary decisions -> consolidated moments -> semantic metadata -> report/timeline.

Outputs (outputs/<id>/level2/): semantic_boundary_decisions.jsonl, consolidated_moments.json, semantic_metadata.json,
level2_report.md, cache/ (every raw VLM output); reports/semantic_timeline.png. Earlier stages are read-only here.
"""
from __future__ import annotations

import time
from collections import Counter
from pathlib import Path

from .. import PIPELINE_VERSION
from ..schemas import check_time_bounds
from ..utils.io import read_json, write_json, write_jsonl
from .metadata_generator import describe
from .semantic_merging import consolidate, decide
from .vlm_backend import CachedVLM


def run(cfg, manifest, entry, vdir: Path, log, backend=None):
    from ..level1.pipeline import device_of
    vid = entry["video_id"]
    cm = read_json(vdir / "correction" / "candidate_moments.json")
    if not cm:
        raise RuntimeError("no candidate moments; run the correction stage first")
    cands = cm["moments"]
    src = read_json(vdir / "source" / "source_metadata.json") or {}
    man = read_json(vdir / "source" / "source_manifest.json")
    transcript = read_json(vdir / "source" / "transcript.json") or {}
    root = Path(cfg["paths"].get("corpus_root") or manifest["corpus_root"])
    video = root / entry["item_dir"] / man["inputs"]["video"]
    vhash = man["hashes"][man["inputs"]["video"]]["digest"]
    out = vdir / "level2"
    out.mkdir(parents=True, exist_ok=True)
    vlm = CachedVLM(cfg, device_of(cfg), out / "cache", vhash, backend)
    t0 = time.time()
    decisions, audit = decide(vlm, cfg, video, cands, transcript, src, log)
    t_merge = time.time() - t0
    moments = consolidate(vid, cands, decisions)
    errs = check_time_bounds(moments, src.get("duration_s"))
    if errs:
        raise RuntimeError(f"consolidated moments are not time-consistent: {errs[:3]}")
    byid = {m["interval_id"]: m for m in cands}
    records = []
    for m in moments:
        sources = [byid[i] for i in m["source_intervals"]]
        obj, attempts, ftimes = describe(vlm, cfg, video, m, sources, transcript, src)
        rec = {"moment_id": m["moment_id"], "video_id": vid, "start_s": m["start_s"], "end_s": m["end_s"],
               "source_intervals": m["source_intervals"],
               "boundary_explanations": {"start": m["start_boundary"], "end": m["end_boundary"]},
               "generation_status": "ok" if obj else "invalid_output"}
        if obj:
            rec.update({k: obj.get(k) for k in ("description", "tags", "entities", "events", "relationships", "context", "attributes")})
        else:  # never invent fields: leave them empty and say why
            rec.update(description=None, tags=[], entities=[], events=[], relationships=[], context={}, attributes={},
                       generation_errors=attempts[-1]["errors"])
        rec.update({"evidence_provenance": {"frames_s": ftimes, "transcript": (transcript.get("primary") or {}).get("file"),
                                            "level1_intervals": m["source_intervals"], "video_context_used": cfg["level2"]["use_video_context"]},
                    "review_status": "automatic", "needs_review": m["needs_review"],
                    "pipeline_version": PIPELINE_VERSION, "model": cfg["level2"]["model_id"],
                    "prompt_version": cfg["level2"]["prompt_version"], "attempts": len(attempts)})
        records.append(rec)
    stats = {**vlm.stats, "merge_stage_s": round(t_merge, 1), "total_s": round(time.time() - t0, 1)}
    if vlm._backend is not None:
        stats["gpu_peak_gb"] = vlm._backend.gpu_memory_gb()
    write_jsonl(out / "semantic_boundary_decisions.jsonl", decisions)
    write_json(out / "merge_windows_audit.json", audit)
    write_json(out / "consolidated_moments.json", {"video_id": vid, "n_moments": len(moments), "moments": moments})
    write_json(out / "semantic_metadata.json", {"video_id": vid, "model": cfg["level2"]["model_id"], "stats": stats,
                                                "moments": records})
    report(vid, cands, decisions, moments, records, stats, out / "level2_report.md")
    try:
        plot(cands, decisions, moments, src.get("duration_s") or moments[-1]["end_s"], vdir / "reports" / "semantic_timeline.png")
    except Exception as e:  # noqa: BLE001
        log.warning(f"timeline plot failed: {e}")
    log.info(f"{len(cands)} candidate intervals -> {len(moments)} moments; decisions {dict(Counter(d['decision'] for d in decisions))}; "
             f"VLM calls {stats['calls']} (cache hits {stats['cache_hits']}), {stats['generate_s']:.0f} s")


def report(vid, cands, D, M, R, stats, path):
    L = [f"# Level 2 report: {vid}", "",
         f"- {len(cands)} candidate intervals -> **{len(M)} consolidated moments**",
         f"- boundary decisions: {dict(Counter(d['decision'] for d in D))}; window conflicts: {sum(d['conflict'] for d in D)}; "
         f"needing review: {sum(d['needs_review'] for d in D)}",
         f"- metadata: {dict(Counter(r['generation_status'] for r in R))}",
         f"- VLM: {stats.get('calls')} calls, {stats.get('cache_hits')} cache hits, {stats.get('frames')} frames, "
         f"{stats.get('generate_s', 0):.0f} s generation, model load {stats.get('model_load_s')} s, GPU peak {stats.get('gpu_peak_gb')} GB",
         "", "## Boundary decisions", "", "| boundary | t (s) | decision | by | explanation |", "|---|---:|---|---|---|"]
    for d in D:
        L.append(f"| {d['boundary']} | {d['t']:g} | {d['decision']} | {d['decided_by']} | {d['explanation'][:220]} |")
    L += ["", "## Moments", ""]
    for r in R:
        L += [f"### {r['moment_id']}: {r['start_s']:g}-{r['end_s']:g} s ({', '.join(r['source_intervals'])})"
              + (" [needs review]" if r["needs_review"] else ""),
              f"- description: {r['description']}", f"- tags: {', '.join(r['tags'] or [])}",
              f"- events: {[e.get('type') for e in r['events'] or []]}; context: {r['context']}",
              f"- start: {r['boundary_explanations']['start'].get('decision')} - {r['boundary_explanations']['start'].get('explanation', '')}",
              f"- end: {r['boundary_explanations']['end'].get('decision')} - {r['boundary_explanations']['end'].get('explanation', '')}", ""]
    path.write_text("\n".join(L), encoding="utf-8")


def plot(cands, D, M, dur, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(2, 1, figsize=(16, 3.6), sharex=True)
    col = {"moment": "tab:green", "ambiguous": "tab:orange", "unobservable": "tab:gray"}
    for m in cands:
        ax[0].axvspan(m["start_s"], m["end_s"], ymin=.2, ymax=.8, color=col.get(m["status"], "tab:blue"), ec="white", lw=.8)
    dc = {"merge": "tab:blue", "keep": "black", "uncertain": "red"}
    for d in D:
        ax[0].axvline(d["t"], color=dc[d["decision"]], lw=1.6 if d["decision"] != "merge" else .8,
                      ls="-" if d["decision"] != "merge" else ":")
    for k, m in enumerate(M):
        ax[1].axvspan(m["start_s"], m["end_s"], ymin=.2, ymax=.8, color=plt.get_cmap("tab20")(k % 20), ec="white", lw=1)
        ax[1].text((m["start_s"] + m["end_s"]) / 2, .5, m["moment_id"].split("_")[-1], ha="center", va="center", fontsize=6)
    ax[0].set_ylabel("candidate\nintervals", rotation=0, ha="right", fontsize=8)
    ax[1].set_ylabel("consolidated\nmoments", rotation=0, ha="right", fontsize=8)
    for a_ in ax:
        a_.set_yticks([])
    ax[-1].set_xlim(0, dur)
    ax[-1].xaxis.set_major_formatter(lambda v, _p=None: f"{int(v // 60)}:{int(v % 60):02d}")
    ax[-1].set_xlabel("boundaries: black = keep, blue dotted = merge, red = uncertain (review)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
