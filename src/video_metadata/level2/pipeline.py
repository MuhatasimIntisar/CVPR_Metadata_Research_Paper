"""Level 2 stage: boundary decisions -> segments -> retrievable moments with schema-v2 metadata.

Outputs (outputs/<id>/level2/):
  semantic_metadata.json          schema v2 document: video, segments, moments, entity_registry (docs/metadata_schema.md)
  semantic_boundary_decisions.jsonl, consolidated_moments.json, merge_windows_audit.json, run_stats.json,
  level2_report.md, cache/ (every raw VLM output); reports/semantic_timeline.png.
Earlier stages are read-only here.
"""
from __future__ import annotations

import datetime as dt
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np

from .. import PIPELINE_VERSION
from ..schemas import check_time_bounds
from ..utils.io import read_json, read_jsonl, write_json, write_jsonl
from .context_builder import transcript_excerpt
from .metadata_generator import describe_batch
from .semantic_merging import consolidate, decide
from .vlm_backend import CachedVLM

SCHEMA_VERSION = "vmp-2.1"


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
    obs = read_jsonl(vdir / "level1" / "observations.jsonl")
    corr = {o["step"]: o for o in read_jsonl(vdir / "correction" / "corrected_observations.jsonl")}
    ocr = read_json(vdir / "level1" / "raw" / "ocr.json") or {}
    cfg = {**cfg, "_ocr_filter": ocr_filter(ocr),
           "_video_words": words(" ".join(s["text"] for s in transcript.get("segments", [])) + " " +
                                 " ".join(it["text"] for smp in ocr.get("samples", []) for it in smp["items"]))}
    root = Path(cfg["paths"].get("corpus_root") or manifest["corpus_root"])
    video = root / entry["item_dir"] / man["inputs"]["video"]
    vhash = man["hashes"][man["inputs"]["video"]]["digest"]
    out = vdir / "level2"
    out.mkdir(parents=True, exist_ok=True)
    vlm = CachedVLM(cfg, device_of(cfg), out / "cache", vhash, backend)
    c = cfg["level2"]
    t0 = time.time()
    decisions, audit = decide(vlm, cfg, video, cands, transcript, src, log)
    t_merge = time.time() - t0
    events = consolidate(vid, cands, decisions)
    errs = check_time_bounds(events, src.get("duration_s"))
    if errs:
        raise RuntimeError(f"segments are not time-consistent: {errs[:3]}")
    byid = {m["interval_id"]: m for m in cands}
    dby = {d["right_interval"]: d for d in decisions}
    plan, jobs = [], []  # every metadata call of the video is planned first, then sent as batches
    for k, ev in enumerate(events):
        seg_id = f"{vid}_s{k:03d}"
        sources = [byid[i] for i in ev["source_intervals"]]
        kids = children(ev, sources, dby, c) if ev["duration_s"] > c["max_moment_s"] else []
        long_unsplit = ev["duration_s"] > c["max_moment_s"] and len(kids) <= 1
        if len(kids) > 1:
            seg_job = len(jobs)
            jobs.append(({**ev, "moment_id": seg_id}, sources))
            kid_jobs = []
            for j, kd in enumerate(kids):
                mid = f"{seg_id}_m{j + 1:02d}"
                kid_jobs.append((mid, kd, len(jobs)))
                jobs.append(({**kd, "moment_id": mid}, [byid[i] for i in kd["source_intervals"]]))
        else:
            mid = f"{seg_id}_m01"
            seg_job = len(jobs)
            kid_jobs = [(mid, ev, seg_job)]
            jobs.append(({**ev, "moment_id": mid}, sources))
        plan.append((seg_id, ev, seg_job, kid_jobs, long_unsplit))
    metas = [dict(zip(("obj", "attempts", "frames", "repairs"), r))
             for r in describe_batch(vlm, cfg, video, jobs, transcript, src)]
    segments, moments = [], []
    for seg_id, ev, seg_job, kid_jobs, long_unsplit in plan:
        seg_meta = metas[seg_job]
        for mid, span, ji in kid_jobs:
            moments.append(moment_record(mid, seg_id, span, metas[ji], obs, corr, cands, transcript, cfg))
        segments.append({"segment_id": seg_id, "start_s": ev["start_s"], "end_s": ev["end_s"],
                         "description": (seg_meta["obj"] or {}).get("description"),
                         "children": [m["moment_id"] for m in moments if m["segment_id"] == seg_id],
                         "source_intervals": ev["source_intervals"], "merged_boundaries": ev["merged_boundaries"],
                         "boundary": {"start": ev["start_boundary"], "end": ev["end_boundary"]},
                         "long_unsplit": long_unsplit, "needs_review": ev["needs_review"] or long_unsplit,
                         "annotation": {"status": "automatic", "review_status": "unreviewed",
                                        "generation_status": "ok" if seg_meta["obj"] else "invalid_output"}})
    for s in segments:
        errs += [f"{s['segment_id']}: {e}" for e in check_time_bounds([m for m in moments if m["segment_id"] == s["segment_id"]], None)]
        kids = [m for m in moments if m["segment_id"] == s["segment_id"]]
        if kids and (kids[0]["start_s"] < s["start_s"] - 1e-6 or kids[-1]["end_s"] > s["end_s"] + 1e-6):
            errs.append(f"{s['segment_id']}: child outside segment")
    if errs:
        raise RuntimeError(f"moments are not time-consistent: {errs[:3]}")
    registry = link_entities(vid, moments)
    stats = {**vlm.stats, "merge_stage_s": round(t_merge, 1), "total_s": round(time.time() - t0, 1)}
    if vlm._backend is not None:
        stats["gpu_peak_gb"] = vlm._backend.gpu_memory_gb()
    doc = {"schema_version": SCHEMA_VERSION,
           "video": video_block(vid, src, man),
           "annotation_set": {"pipeline_version": PIPELINE_VERSION, "model": c["model_id"],
                              "inference_backend": getattr(vlm._backend, "name", "custom") if vlm.stats["calls"] else "cache only (no new model calls)",
                              "merge_prompt_version": c["prompt_version"],
                              "metadata_prompt_version": c.get("metadata_prompt_version", c["prompt_version"]),
                              "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                              "status": "automatic (not gold)"},
           "segments": segments, "moments": moments, "entity_registry": registry}
    write_jsonl(out / "semantic_boundary_decisions.jsonl", decisions)
    write_json(out / "merge_windows_audit.json", audit)
    write_json(out / "consolidated_moments.json", {"video_id": vid, "n_segments": len(events), "segments": events})
    write_json(out / "run_stats.json", stats)
    write_json(out / "semantic_metadata.json", doc)
    report(doc, decisions, stats, out / "level2_report.md")
    if cfg["paths"].get("metadata_root"):  # the dataset copy: metadata/<original video name>_metadata.json (+ .md)
        stem = Path((man["inputs"].get("original_names") or {}).get(man["inputs"]["video"]) or vid).stem
        md = Path(cfg["paths"]["metadata_root"])
        md.mkdir(parents=True, exist_ok=True)
        write_json(md / f"{stem}_metadata.json", doc)
        report(doc, decisions, stats, md / f"{stem}_metadata.md")
    try:
        plot(cands, decisions, segments, moments, src.get("duration_s") or events[-1]["end_s"], vdir / "reports" / "semantic_timeline.png")
    except Exception as e:  # noqa: BLE001
        log.warning(f"timeline plot failed: {e}")
    log.info(f"{len(cands)} candidate intervals -> {len(segments)} segments, {len(moments)} moments; decisions "
             f"{dict(Counter(d['decision'] for d in decisions))}; VLM calls {stats['calls']} (cache hits {stats['cache_hits']}), "
             f"{stats['generate_s']:.0f} s")


def video_block(vid, src, man):
    m = man.get("media") or {}
    return {"video_id": vid,
            "source": {"provider": "Internet Archive", "url": src.get("source_url"), "archive_identifier": src.get("ia_identifier"),
                       "title": src.get("title"), "description": src.get("description"), "creator": src.get("creator"),
                       "publication_date": src.get("publication_date"), "languages": [x for x in [src.get("language_code")] if x],
                       "source_tags": src.get("subjects", []), "original_platform": src.get("original_platform") or None,
                       "role": "video-level context only; never moment-level ground truth"},
            "media": {"duration_s": src.get("duration_s"), "fps": m.get("fps"), "width": m.get("width"), "height": m.get("height")}}


def ocr_filter(ocr: dict, min_share: float = 0.10) -> dict:
    """Channel branding = the static overlay the OCR step found plus single words visible in >= min_share of OCR
    samples (station logo words such as MORNING / DELAWARE). Lines of several words are never branding."""
    S = ocr.get("samples", [])
    c = Counter()
    for smp in S:
        c.update({it["text"] for it in smp["items"] if it.get("conf", 1) >= 0.5 and len(it["text"].split()) == 1})
    brand = sorted(t for t, n in c.items() if S and n / len(S) >= min_share)
    return {"overlays": tuple(ocr.get("static_overlay", [])), "brand_words": tuple(brand)}


def observed_block(a, b, obs, corr, cands, ocr=None):
    """Detector-measured facts for the span (no VLM): objective, cheap, useful for compositional queries."""
    st = [o for o in obs if a <= o["t"] < b]
    if not st:
        return {}
    acts = Counter(corr[o["step"]]["corrected_state"] for o in st if o["step"] in corr)
    envs = Counter(o["environment"]["top3"][0][0] for o in st if not o["black_frame"])
    n = len(st)
    pc = [o["people"]["count_median"] for o in st if o["people"]["count_median"] is not None]
    texts = list(dict.fromkeys(x for o in st for x in o["ocr_text"] if useful_text(x, ocr or {})))[:15]
    sc = Counter()
    for o in st:
        sc.update((o.get("shot_scale") or {}).get("frames") or {})
    ns = sum(sc.values())
    cam = Counter(o["motion"]["camera"] for o in st if o["motion"]["camera"])
    subs = [s for m in cands if m["end_s"] > a and m["start_s"] < b for s in m["sub_events"] if s["start_s"] >= a - 1e-6 and s["end_s"] <= b + 1e-6]
    return {"people_count_median": float(np.median(pc)) if pc else None, "people_count_max": max((o["people"]["count_max"] or 0) for o in st),
            "speech_share": round(float(np.mean([o["audio"]["speech_share"] for o in st])), 3),
            "music_share": round(float(np.mean([o["audio"]["events"]["music"] or 0 for o in st])), 3),
            "speakers": sorted({sp for o in st for sp in o["audio"]["speakers"]}),
            "setting_detector": [{"label": k, "share": round(v / n, 3)} for k, v in envs.most_common(2)],
            "activity_detector": [{"state": k, "share": round(v / n, 3)} for k, v in acts.most_common(3)],
            "detector_sub_events": subs, "on_screen_text": texts,
            "shot_cuts": sum(len(o["shot_cuts"]) for o in st),
            "shot_scale_detector": [{"label": k, "share": round(v / ns, 3)} for k, v in sc.most_common()] if ns else [],
            "camera_motion_detector": [{"label": k, "share": round(v / sum(cam.values()), 3)} for k, v in cam.most_common()],
            "provenance": "Level-1 detectors (YOLO, YOLO-pose framing heuristic, CLIP, X-CLIP, VAD, diarisation, AST, OCR, "
                          "shot detection, optical-flow camera motion); automatic"}


def useful_text(x, f):
    """Drop OCR fragments (< 4 letters/digits), the static overlay (station bug such as LIVE) with near-identical
    misreads (LIVB, UVE), and single-word misreads of recurring logo words (MORANG, DEORAXNG for MORNING/DELAWARE).
    Lower thirds, captions and title cards (several words) are kept."""
    if sum(ch.isalnum() for ch in x) < 4:
        return False
    import difflib

    def sim(u, v):
        return difflib.SequenceMatcher(None, u.lower(), v.lower()).ratio()
    if any(sim(x, o) >= 0.6 for o in f.get("overlays", ())):
        return False
    return not (len(x.split()) == 1 and any(sim(x, w) >= 0.5 or x.lower() in w.lower() for w in f.get("brand_words", ())))


def moment_record(mid, seg_id, span, meta, obs, corr, cands, transcript, cfg):
    o = meta["obj"]
    c = cfg["level2"]
    rec = {"moment_id": mid, "segment_id": seg_id, "start_s": span["start_s"], "end_s": span["end_s"],
           "source_intervals": span["source_intervals"],
           "boundary": {"start": span["start_boundary"], "end": span["end_boundary"]}}
    if o:
        rec.update({k: o.get(k) for k in ("description", "tags", "entities", "actions", "relationships", "context", "attributes")})
    else:  # never invent fields: leave them empty and say why
        rec.update(description=None, tags=[], entities=[], actions=[], relationships=[], context={}, attributes={})
    rec["observed"] = observed_block(span["start_s"], span["end_s"], obs, corr, cands, cfg.get("_ocr_filter"))
    a, b = span["start_s"], span["end_s"]
    sw = {"transcript": words(transcript_excerpt(transcript, a, b, 10 ** 7)),
          "on_screen_text": words(" ".join(x for st in obs if a <= st["t"] < b for x in st["ocr_text"]))}
    verify_names(rec.get("entities"), sw, cfg.get("_video_words", set()), meta["repairs"])
    reasons = review_reasons(rec, o, meta["repairs"], c.get("review", {}))
    if span.get("needs_review"):
        reasons.insert(0, "boundary decision uncertain or disputed")
    rec["annotation"] = {"status": "automatic", "review_status": "unreviewed", "needs_review": bool(reasons),
                         "review_reasons": reasons,
                         "generation_status": "ok" if o else "invalid_output",
                         "generation_errors": None if o else meta["attempts"][-1]["errors"],
                         "normalisation_repairs": meta["repairs"], "attempts": len(meta["attempts"]),
                         "model": c["model_id"], "metadata_prompt_version": c.get("metadata_prompt_version", c["prompt_version"]),
                         "evidence_provenance": {"frames_s": meta["frames"], "transcript": (transcript.get("primary") or {}).get("file"),
                                                 "level1_intervals": span["source_intervals"],
                                                 "video_context_used": c["use_video_context"]},
                         "entity_ids": "local to this moment; see video_entity_id for cross-moment links"}
    return rec


def _said(name, text_words):
    """A name is evidenced if one of its words (>= 3 letters) matches a word of the text (ASR spelling tolerated)."""
    import difflib
    toks = [t for t in re.findall(r"\w+", name.lower()) if len(t) >= 3]
    return any(t in text_words or any(difflib.SequenceMatcher(None, t, w).ratio() >= 0.85 for w in text_words
                                      if abs(len(w) - len(t)) <= 2) for t in toks)


def words(text):
    return set(re.findall(r"\w+", text.lower()))


def verify_names(entities, span_words, video_words, repairs):
    """Make name_source factual: transcript / on_screen_text when the name occurs in this span, elsewhere_in_video
    when it is only said or shown at another time of the video; a name found only in the title/description
    (source metadata) is removed - video-level metadata is never moment-level evidence."""
    for e in entities or []:
        nm = e.get("name")
        if not nm:
            continue
        if _said(nm, span_words["transcript"]):
            src = "transcript"
        elif _said(nm, span_words["on_screen_text"]):
            src = "on_screen_text"
        elif _said(nm, video_words):
            src = "elsewhere_in_video"
        else:
            repairs.append(f"name removed: entity {e.get('id')} name {nm!r} is not said or shown anywhere in the video "
                           "(only in source metadata)")
            e["name"] = e["name_source"] = None
            continue
        if src != e.get("name_source"):
            repairs.append(f"entity {e.get('id')} name {nm!r}: name_source {e.get('name_source')!r} -> {src!r} (checked)")
            e["name_source"] = src


def _count(e):
    if e.get("type") == "person":
        return 1
    if e.get("type") == "person_group":
        m = re.match(r"\s*(\d+)", str(e.get("count") or ""))
        return int(m.group(1)) if m else 2
    return 0


def review_reasons(rec, obj, repairs, rv) -> list[str]:
    """Deterministic reasons a moment needs a human look (never changes content)."""
    R = []
    if obj is None:
        R.append("metadata generation failed")
    dur = rec["end_s"] - rec["start_s"]
    if dur < rv.get("min_moment_s", 0):
        R.append(f"very short moment ({dur:g} s)")
    R += [x for x in repairs if x.startswith("name removed") and "entities" in x]  # one name on several people
    R += [x for x in repairs if x.startswith("relationship") and "dropped" in x]
    ob = rec.get("observed") or {}
    n = sum(_count(e) for e in rec.get("entities") or [])
    if ob.get("people_count_max") is not None and n > ob["people_count_max"] + rv.get("people_excess", 1):
        R.append(f"people: metadata describes {n}, detector saw at most {ob['people_count_max']}")
    from .metadata_generator import SHOT_SCALES
    v = (rec.get("attributes") or {}).get("shot_scale")
    det = {d["label"]: d["share"] for d in ob.get("shot_scale_detector") or []}
    person = 1 - det.get("no_person", 0)
    if v in SHOT_SCALES and person >= rv.get("shot_min_person_share", 0.5):
        share = det.get(v, 0) / person
        dom = max((k for k in det if k in SHOT_SCALES), key=lambda k: det[k], default=None)
        if dom and share < rv.get("shot_min_label_share", 0.2) and abs(SHOT_SCALES.index(dom) - SHOT_SCALES.index(v)) > rv.get("shot_max_distance", 1):
            R.append(f"shot scale: metadata {v}, detector mostly {dom}")
    return R


def link_entities(vid, moments):
    """Video-level registry: named entities (name with a source) linked across moments by normalised name."""
    reg = {}
    for m in moments:
        for e in m.get("entities") or []:
            if not e.get("name") or not e.get("name_source"):
                continue
            key = " ".join(e["name"].lower().split())
            r = reg.setdefault(key, {"video_entity_id": f"{vid}_e{len(reg) + 1:02d}", "name": e["name"], "type": e.get("type"),
                                     "name_sources": set(), "mentions": []})
            r["name_sources"].add(e["name_source"])
            r["mentions"].append({"moment_id": m["moment_id"], "local_id": e["id"]})
            e["video_entity_id"] = r["video_entity_id"]
    out = []
    for r in reg.values():
        out.append({**r, "name_sources": sorted(r["name_sources"]),
                    "link_method": "same normalised name (on-screen text or transcript); unverified"})
    return out


def children(event, sources, dby, c):
    """Child moments of a long segment: its candidate intervals, short ones (< min_child_s) joined to a neighbour."""
    spans = [[s] for s in sources]
    i = 0
    while i < len(spans) and len(spans) > 1:
        if spans[i][-1]["end_s"] - spans[i][0]["start_s"] < c["min_child_s"]:
            j = i + 1 if i + 1 < len(spans) else i - 1
            lo, hi = min(i, j), max(i, j)
            spans[lo:hi + 1] = [spans[lo] + spans[hi]]
            i = max(0, lo - 1) if lo else 0
            continue
        i += 1
    out = []
    for k, g in enumerate(spans):
        d = dby.get(g[0]["interval_id"])
        out.append({"moment_id": f"{event['moment_id']}.{k + 1:02d}", "video_id": event["video_id"],
                    "start_s": g[0]["start_s"], "end_s": g[-1]["end_s"], "duration_s": round(g[-1]["end_s"] - g[0]["start_s"], 3),
                    "source_intervals": [x["interval_id"] for x in g],
                    "start_boundary": event["start_boundary"] if k == 0 else
                    ({"boundary": d["boundary"], "t": d["t"], "decision": f"child split inside segment (VLM said {d['decision']})",
                      "explanation": d["explanation"], "automatic_evidence": d["automatic_evidence"]} if d else {"decision": "child start"}),
                    "end_boundary": event["end_boundary"] if k == len(spans) - 1 else {"decision": "next child"},
                    "needs_review": event["needs_review"]})
    return out


def report(doc, D, stats, path):
    M, S = doc["moments"], doc["segments"]
    L = [f"# Level 2 report: {doc['video']['video_id']} (schema {doc['schema_version']})", "",
         f"- {len(S)} segments, {len(M)} retrievable moments ({sum(len(s['children']) > 1 for s in S)} segments split into children, "
         f"{sum(s['long_unsplit'] for s in S)} long segments without split points)",
         f"- boundary decisions: {dict(Counter(d['decision'] for d in D))}; needing review: {sum(d['needs_review'] for d in D)}",
         f"- metadata: {dict(Counter(m['annotation']['generation_status'] for m in M))}; normalisation repairs: "
         f"{sum(len(m['annotation']['normalisation_repairs']) for m in M)}",
         f"- named entities linked across moments: {len(doc['entity_registry'])}; moments needing review: "
         f"{sum(m['annotation']['needs_review'] for m in M)}",
         f"- VLM: {stats.get('calls')} calls, {stats.get('cache_hits')} cache hits, {stats.get('frames')} frames, "
         f"{stats.get('generate_s', 0):.0f} s generation, GPU peak {stats.get('gpu_peak_gb')} GB", ""]
    for s in S:
        L += [f"## {s['segment_id'][-4:]}: {s['start_s']:g}-{s['end_s']:g} s" + (" [needs review]" if s["needs_review"] else ""),
              f"{s['description']}", f"- start: {s['boundary']['start'].get('decision')} - {s['boundary']['start'].get('explanation', '')}", ""]
        for m in [m for m in M if m["segment_id"] == s["segment_id"]]:
            ents = ", ".join(f"{e['id']}({e['type']}{', ' + e['name'] if e.get('name') else ''})" for e in m["entities"] or [])
            rels = "; ".join(f"{r['subject']} {r['predicate']} {r['object']}" for r in m["relationships"] or [])
            L += [f"### {m['moment_id'].split('_')[-1]}: {m['start_s']:g}-{m['end_s']:g} s",
                  f"- description: {m['description']}", f"- tags: {', '.join(m['tags'] or [])}",
                  f"- entities: {ents}", f"- actions: {[a.get('type') for a in m['actions'] or []]}; relationships: {rels or '-'}",
                  f"- context: {m['context']}; attributes: {m['attributes']}",
                  f"- observed: people {m['observed'].get('people_count_median')}, speech {m['observed'].get('speech_share')}, "
                  f"music {m['observed'].get('music_share')}, setting {[x['label'] for x in m['observed'].get('setting_detector', [])]}",
                  f"- observed camera: shot {[(x['label'], x['share']) for x in m['observed'].get('shot_scale_detector', [])[:2]]}, "
                  f"motion {[(x['label'], x['share']) for x in m['observed'].get('camera_motion_detector', [])[:2]]}",
                  f"- repairs: {m['annotation']['normalisation_repairs'] or '-'}",
                  f"- review: {m['annotation'].get('review_reasons') or '-'}", ""]
    path.write_text("\n".join(L), encoding="utf-8")


def plot(cands, D, S, M, dur, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(3, 1, figsize=(16, 4.6), sharex=True)
    col = {"moment": "tab:green", "ambiguous": "tab:orange", "unobservable": "tab:gray"}
    for m in cands:
        ax[0].axvspan(m["start_s"], m["end_s"], ymin=.2, ymax=.8, color=col.get(m["status"], "tab:blue"), ec="white", lw=.8)
    dc = {"merge": "tab:blue", "keep": "black", "uncertain": "red"}
    for d in D:
        ax[0].axvline(d["t"], color=dc[d["decision"]], lw=1.6 if d["decision"] != "merge" else .8, ls="-" if d["decision"] != "merge" else ":")
    cm = plt.get_cmap("tab20")
    for k, s in enumerate(S):
        ax[1].axvspan(s["start_s"], s["end_s"], ymin=.2, ymax=.8, color=cm(k % 20), ec="white", lw=1)
        ax[1].text((s["start_s"] + s["end_s"]) / 2, .5, s["segment_id"][-4:], ha="center", va="center", fontsize=6)
    for k, m in enumerate(M):
        ax[2].axvspan(m["start_s"], m["end_s"], ymin=.2, ymax=.8, color=cm((k * 3) % 20), ec="white", lw=1)
    for a_, lab in zip(ax, ("candidate\nintervals", "segments", "moments")):
        a_.set_yticks([])
        a_.set_ylabel(lab, rotation=0, ha="right", fontsize=8)
    ax[-1].set_xlim(0, dur)
    ax[-1].xaxis.set_major_formatter(lambda v, _p=None: f"{int(v // 60)}:{int(v % 60):02d}")
    ax[-1].set_xlabel("boundaries: black = keep, blue dotted = merge, red = uncertain (review)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
