"""Retrieval view of the cleaned metadata: only what a retrieval system indexes or a benchmark query can ask about.

The full annotation document (boundary decisions, evidence, repairs, cleanup log, review reasons) stays in
outputs/<id>/cleanup/clean_metadata.json for review and provenance. The retrieval file
(metadata/<original video name>_metadata.json, schema tvm-retrieval-1) keeps per moment: time span, description,
tags, entities, actions, relationships, setting / topic / location, camera attributes, detector-measured facts and
its status (automatic / approved). Nothing in it says how it was produced.
"""
from __future__ import annotations

RETRIEVAL_SCHEMA = "tvm-retrieval-1"


def _clean(d):
    """Drop None / empty values (keeps 0 and False)."""
    return {k: v for k, v in d.items() if v is not None and v != [] and v != {} and v != ""}


def _entity(e):
    out = {"id": e.get("id"), "type": e.get("type"), "name": e.get("name"), "description": e.get("description"),
           "count": e.get("count")}
    if e.get("name"):
        out["video_entity_id"] = e.get("video_entity_id")
        out["name_binding"] = e.get("name_binding")
    if e.get("evidence") == "transcript":
        out["visible"] = False  # mentioned, not on screen
    return _clean(out)


def _moment(m):
    ctx, at, ob, an = m.get("context") or {}, m.get("attributes") or {}, m.get("observed") or {}, m.get("annotation") or {}
    rs = an.get("review_status") or "unreviewed"
    return _clean({
        "moment_id": m["moment_id"], "segment_id": m["segment_id"],
        "start_s": round(m["start_s"], 3), "end_s": round(m["end_s"], 3),
        "description": m.get("description"), "tags": m.get("tags"),
        "entities": [_entity(e) for e in m.get("entities") or []],
        "actions": [_clean({"type": a.get("type"), "participants": a.get("participants")}) for a in m.get("actions") or []],
        "relationships": [{"subject": r["subject"], "predicate": r["predicate"], "object": r["object"]}
                          for r in m.get("relationships") or []],
        "setting": ctx.get("setting"), "topic": ctx.get("topic"), "location": ctx.get("location"),
        "camera": _clean({"shot_scale": at.get("shot_scale"), "camera_motion": at.get("camera_motion"),
                          "camera_angle": at.get("camera_angle"), "note": at.get("cinematography")}),
        "activity": at.get("action"),
        "detected": _clean({"people_median": ob.get("people_count_median"), "people_max": ob.get("people_count_max"),
                            "speech_share": ob.get("speech_share"), "music_share": ob.get("music_share"),
                            "n_speakers": len(ob["speakers"]) if ob.get("speakers") is not None else None,
                            "shot_cuts": ob.get("shot_cuts"), "on_screen_text": ob.get("on_screen_text")}),
        "status": "approved" if rs in ("approved", "edited") else "automatic",
        "needs_review": bool(an.get("needs_review")),
    })


def retrieval_view(doc, category=None):
    v = doc.get("video") or {}
    src, media = v.get("source") or {}, v.get("media") or {}
    loc = v.get("location") or {}
    video = _clean({"video_id": v.get("video_id"), "title": src.get("title"), "url": src.get("url"),
                    "channel": src.get("creator"), "date": src.get("publication_date"), "category": category,
                    "duration_s": media.get("duration_s") and round(float(media["duration_s"]), 3),
                    "languages": src.get("languages"),
                    "location": _clean({k: loc.get(k) for k in ("country", "region", "place")}) if loc else None,
                    "description": src.get("description")})
    moments = [_moment(m) for m in doc.get("moments") or []]
    segments = [_clean({"segment_id": s["segment_id"], "start_s": round(s["start_s"], 3), "end_s": round(s["end_s"], 3),
                        "description": None if s.get("single_moment") else s.get("description"),
                        "moment_ids": s.get("children")}) for s in doc.get("segments") or []]
    people = [_clean({"video_entity_id": r["video_entity_id"], "name": r["name"], "type": r.get("type"),
                      "moment_ids": list(dict.fromkeys(x["moment_id"] for x in r.get("mentions", [])))})
              for r in doc.get("entity_registry") or []]
    aset = doc.get("annotation_set") or {}
    return {"schema": RETRIEVAL_SCHEMA, "video": video,
            "generated": _clean({"model": aset.get("model"), "cleanup": (aset.get("cleanup") or {}).get("version"),
                                 "status": "automatic unless a moment says approved"}),
            "segments": segments, "moments": moments, "named_entities": people}


def _ts(s):
    s = int(round(s))
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}" if s >= 3600 else f"{s // 60}:{s % 60:02d}"


def retrieval_md(view):
    v = view["video"]
    loc = ", ".join(x for x in [(v.get("location") or {}).get(k) for k in ("place", "region", "country")] if x)
    L = [f"# {v.get('title') or v['video_id']}", "",
         f"{v.get('channel', '')} · {v.get('date', '')} · {_ts(v.get('duration_s') or 0)} · {v.get('category', '')}"
         + (f" · {loc}" if loc else ""), "",
         f"{len(view['segments'])} segments, {len(view['moments'])} moments; "
         f"{sum(m.get('needs_review', False) for m in view['moments'])} need review", ""]
    byseg = {}
    for m in view["moments"]:
        byseg.setdefault(m["segment_id"], []).append(m)
    for s in view["segments"]:
        kids = byseg.get(s["segment_id"], [])
        if s.get("description"):
            L += [f"## {_ts(s['start_s'])}–{_ts(s['end_s'])} · segment", s["description"], ""]
        for m in kids:
            cam = m.get("camera") or {}
            ents = "; ".join((f"{e['name']} ({e.get('description', e['type'])})" if e.get("name") else e.get("description", e["type"]))
                             + ("" if e.get("visible", True) else " [mentioned]") for e in m.get("entities", []))
            rels = ", ".join(f"{r['subject']} {r['predicate']} {r['object']}" for r in m.get("relationships", []))
            d = m.get("detected") or {}
            L += [f"### {_ts(m['start_s'])}–{_ts(m['end_s'])}" + (" · needs review" if m.get("needs_review") else ""),
                  m.get("description") or "*(no description)*", "",
                  f"- tags: {', '.join(m.get('tags', []))}",
                  f"- setting: {m.get('setting', '-')}; topic: {m.get('topic', '-')}" + (f"; location: {m['location']}" if m.get("location") else ""),
                  f"- camera: {cam.get('shot_scale', '-')}, {cam.get('camera_motion', '-')}, {cam.get('camera_angle', '-')}",
                  f"- people: {ents or '-'}"]
            if m.get("actions"):
                L.append(f"- actions: {', '.join(a['type'] for a in m['actions'])}" + (f"; {rels}" if rels else ""))
            L.append(f"- detected: {d.get('people_median', '-')} people, speech {d.get('speech_share', '-')}, music {d.get('music_share', '-')}"
                     + (f"; text: {' / '.join(d['on_screen_text'][:4])}" if d.get("on_screen_text") else ""))
            L.append("")
    return "\n".join(L)
