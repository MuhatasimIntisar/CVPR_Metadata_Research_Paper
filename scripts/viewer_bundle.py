#!/usr/bin/env python3
"""Pack every annotation layer of a video into one JSON for viewer/index.html (read-only; nothing is recomputed).

  python scripts/viewer_bundle.py ../outputs --out ../viewer_bundles                      # every video in an outputs root
  python scripts/viewer_bundle.py ../outputs_ab/q27b --ids <id> <id> --out ../viewer_bundles/q27b

Per video (<video_id>.json): source info, the 2 s observation steps (activity raw/corrected, setting, people, speech,
music, speakers, shot scale, camera motion, black frames, OCR, topic change, shot cuts), transcript, all boundary
evidence and the pooled boundary decisions of the correction stage, candidate intervals, Level-2 merge/keep decisions,
segments and moments (cleanup document if present, else Level 2), the entity registry and the cleanup log.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def rj(p, default=None):
    p = Path(p)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else default


def rjl(p):
    p = Path(p)
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()] if p.exists() else []


def r3(x):
    return None if x is None else round(float(x), 3)


def top(lst):
    return lst[0][0] if lst else None


def steps(vdir):
    obs = rjl(vdir / "level1" / "observations.jsonl")
    corr = {c["step"]: c for c in rjl(vdir / "correction" / "corrected_observations.jsonl")}
    out = []
    for o in obs:
        c = corr.get(o["step"], {})
        a, au, ss = o.get("activity") or {}, o.get("audio") or {}, o.get("shot_scale") or {}
        out.append({
            "t0": o["start"], "t1": o["end"],
            "act": c.get("corrected_state") or a.get("top1"), "act_raw": a.get("top1"), "act_rel": r3(a.get("top1_rel")),
            "act_changed": c.get("changed", False), "act_rule": c.get("rule_reason"),
            "set": top((o.get("environment") or {}).get("top3")), "pres": (o.get("presentation") or {}).get("label"),
            "black": o.get("black_frame", False),
            "ppl": (o.get("people") or {}).get("count_median"), "ppl_max": (o.get("people") or {}).get("count_max"),
            "obj": list((o.get("objects") or {}).keys())[:6],
            "speech": r3(au.get("speech_share")), "music": r3((au.get("events") or {}).get("music")),
            "spk": sorted((au.get("speakers") or {}), key=lambda k: -au["speakers"][k])[:3],
            "shot": ss.get("label"), "cam": (o.get("motion") or {}).get("camera"),
            "topic": r3((o.get("topic") or {}).get("change")), "cuts": [r3(x) for x in o.get("shot_cuts") or []],
            "ocr": (o.get("ocr_text") or [])[:4]})
    return out


def bundle(vdir: Path):
    vid = vdir.name
    src = rj(vdir / "source" / "source_metadata.json", {})
    man = rj(vdir / "source" / "source_manifest.json", {}) or {}
    tr = rj(vdir / "source" / "transcript.json", {}) or {}
    cm = rj(vdir / "correction" / "candidate_moments.json", {}) or {}
    be = rj(vdir / "correction" / "boundary_evidence.json", {}) or {}
    doc = rj(vdir / "cleanup" / "clean_metadata.json") or rj(vdir / "level2" / "semantic_metadata.json") or {}
    inp = man.get("inputs") or {}
    return {
        "bundle": "tvm-viewer-1", "video_id": vid,
        "title": src.get("title"), "channel": src.get("creator"), "date": src.get("publication_date"),
        "url": src.get("source_url"), "duration": r3(src.get("duration_s")),
        "video_file": inp.get("video"), "original_name": (inp.get("original_names") or {}).get(inp.get("video")),
        "metadata_stage": "cleanup" if (vdir / "cleanup" / "clean_metadata.json").exists() else "level2",
        "model": (doc.get("annotation_set") or {}).get("model"),
        "cleanup_version": ((doc.get("annotation_set") or {}).get("cleanup") or {}).get("version"),
        "steps": steps(vdir),
        "transcript": [[r3(s["start"]), r3(s["end"]), s.get("text", "")] for s in tr.get("segments", [])],
        "evidence": [{k: (r3(v) if k == "t" else v) for k, v in e.items() if k in ("t", "type", "weight", "from", "to", "note",
                      "clip_cosine", "shot_cut", "z", "words_before", "words_after", "stream", "before_share", "after_share",
                      "from_bin_median", "to_bin_median", "new_state_persists_s")} for e in be.get("all_evidence", [])],
        "pooled": [{"t": r3(g["t"]), "score": g.get("score"), "accepted": g.get("accepted"), "types": g.get("types"),
                    "decided_by": g.get("decided_by"), "reason": g.get("reason")} for g in be.get("pooled_candidates", [])],
        "candidates": [{"id": m["interval_id"], "t0": m["start_s"], "t1": m["end_s"], "status": m.get("status"),
                        "activities": m.get("activities"), "environment": m.get("environment"), "sub_events": m.get("sub_events"),
                        "audio": m.get("audio"), "people": m.get("people_median"),
                        "start_reason": (m.get("start_boundary") or {}).get("reason")} for m in cm.get("moments", [])],
        "decisions": [{"id": d["boundary"], "t": d["t"], "decision": d["decision"], "decided_by": d.get("decided_by"),
                       "explanation": d.get("explanation"), "needs_review": d.get("needs_review"),
                       "evidence": (d.get("automatic_evidence") or {}).get("reason")}
                      for d in rjl(vdir / "level2" / "semantic_boundary_decisions.jsonl")],
        "segments": doc.get("segments", []), "moments": doc.get("moments", []),
        "entity_registry": doc.get("entity_registry", []), "video_location": (doc.get("video") or {}).get("location"),
        "cleanup_log": rjl(vdir / "cleanup" / "cleanup_log.jsonl"),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("outputs_root")
    ap.add_argument("--ids", nargs="*")
    ap.add_argument("--out", default="../viewer_bundles")
    a = ap.parse_args(argv)
    root, out = Path(a.outputs_root), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    vids = a.ids or sorted(p.name for p in root.iterdir() if (p / "level1" / "observations.jsonl").exists())
    index = []
    for vid in vids:
        vdir = root / vid
        if not (vdir / "level1" / "observations.jsonl").exists():
            print(f"skip {vid}: no Level-1 observations")
            continue
        b = bundle(vdir)
        (out / f"{vid}.json").write_text(json.dumps(b, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        index.append({"video_id": vid, "title": b["title"], "moments": len(b["moments"]), "original_name": b["original_name"]})
        print(f"{vid}: {len(b['steps'])} steps, {len(b['candidates'])} candidates, {len(b['segments'])} segments, "
              f"{len(b['moments'])} moments ({b['metadata_stage']})")
    (out / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(index)} bundles -> {out}")


if __name__ == "__main__":
    main()
