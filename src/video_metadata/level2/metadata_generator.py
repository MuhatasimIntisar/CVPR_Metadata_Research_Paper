"""Semantic metadata for one segment/moment (schema v2; one VLM call, frames spread across the span).

Schema v2 (docs/metadata_schema.md) follows the reviewed design: description, tags, entities, actions,
relationships, context, attributes. Names need a source; groups of interchangeable people are one person_group;
relationships link two entities with a predicate from a closed list (subject acts on object); no time_of_day;
unknown = null; camera attributes use closed vocabularies (shot_scale, camera_motion, camera_angle).
Deterministic post-processing (normalise) repairs only form (null spellings, tag case/duplicates, action-type
aliases, dropping relationships that break the rules, removing names without a valid source or given to several
entities, mapping the model's own camera words onto the vocabularies) and records every repair; it never adds content.
"""
from __future__ import annotations

import re
from collections import Counter

from .context_builder import interval_text, transcript_excerpt, video_context
from .frame_sampler import grab, spread
from .output_validator import ask_batch, check_metadata

PREDICATES = ["interviews", "talks_to", "listens_to", "performs_for", "performs_with", "dances_with", "sings_with",
              "plays_music_with", "demonstrates_to", "teaches", "instructs", "presents", "introduces", "watches",
              "applauds", "holds", "uses", "gives_to", "hugs", "points_at"]
ENTITY_TYPES = ["person", "person_group", "object", "organization", "place", "animal", "text"]
NAME_SOURCES = ("on_screen_text", "transcript")
# Closed camera vocabularies (shot scale is ordinal: index distance is used to compare with the Level-1 estimate)
SHOT_SCALES = ["extreme_close_up", "close_up", "medium", "medium_wide", "wide", "extreme_wide"]
CAMERA_MOTIONS = ["static", "pan", "tilt", "zoom", "tracking", "handheld"]
CAMERA_ANGLES = ["eye_level", "high", "low", "overhead"]
# keyword -> closed value, used only to map the model's own free-text camera words onto the vocabularies
SCALE_WORDS = [("extreme close", "extreme_close_up"), ("close-up", "close_up"), ("close up", "close_up"),
               ("closeup", "close_up"), ("medium wide", "medium_wide"), ("medium long", "medium_wide"),
               ("medium-wide", "medium_wide"), ("cowboy", "medium_wide"), ("medium", "medium"), ("mid shot", "medium"),
               ("extreme wide", "extreme_wide"), ("extreme long", "extreme_wide"), ("aerial", "extreme_wide"),
               ("full shot", "wide"), ("wide", "wide"), ("long shot", "wide"), ("establishing", "wide")]
MOTION_WORDS = [("static", "static"), ("fixed", "static"), ("locked", "static"), ("stationary", "static"),
                ("pan", "pan"), ("tilt", "tilt"), ("zoom", "zoom"), ("tracking", "tracking"), ("dolly", "tracking"),
                ("follow", "tracking"), ("handheld", "handheld"), ("hand-held", "handheld"), ("shaky", "handheld")]
ANGLE_WORDS = [("eye level", "eye_level"), ("eye-level", "eye_level"), ("overhead", "overhead"), ("top-down", "overhead"),
               ("bird", "overhead"), ("high angle", "high"), ("high-angle", "high"), ("low angle", "low"), ("low-angle", "low")]

SCHEMA = """Describe ONE span of a video for a moment-retrieval dataset. Use only what the frames show and what the
transcript of THIS span says. Rules:
- Unknown or unsupported values are null (never "unknown", "none" or guesses).
- description: 1-3 factual sentences about what happens, not about editing ("the video cuts to...").
- People: a named person only if the name is shown on screen or said in this span's transcript; set name_source to
  "on_screen_text" or "transcript"; otherwise name null and name_source null. A name belongs to ONE entity only:
  never give the same name to several people; if you cannot tell which visible person it is, leave all names null.
  Several interchangeable people doing the same thing (dancers, audience, band) are ONE entity of type person_group
  with a count ("2", "3-5", "6-10", "10+"); a group gets a name only if it has a proper name (e.g. a band's name),
  never a description such as "dancers".
- Clothing and appearance go in the entity description, never in relationships.
- actions: what entities do (snake_case verb_noun, e.g. perform_ballet, give_interview, demonstrate_step).
- relationships: only between two different entity ids, only with one of these predicates:
  """ + ", ".join(PREDICATES) + """. Being in the same frame is not a relationship. Fewer is better.
  Direction matters: the subject does the predicate TO the object (the host who asks questions "interviews" the
  guest who answers; the dancer "performs_for" the audience).
- context.topic: what is being talked about, from the transcript (null if no speech). context.location: a real-world
  place only if shown or said in this span (not inferred from the programme's name). context.setting: the visible
  physical setting (e.g. television studio, theatre stage, kitchen, council chamber).
- The video-level title/description is background only; do not copy its names or keywords unless this span's
  frames or transcript show/say them.
- attributes.shot_scale: one of """ + ", ".join(SHOT_SCALES) + """ (the most common framing of the main subject);
  attributes.camera_motion: one of """ + ", ".join(CAMERA_MOTIONS) + """; attributes.camera_angle: one of
  """ + ", ".join(CAMERA_ANGLES) + """; null if not clear. attributes.cinematography: optional short free-text note.
- At most 12 tags (lowercase natural-language keywords), 8 entities, 6 actions, 6 relationships.
- Mark evidence "visual", "transcript" or "both".
Answer with ONLY this JSON object on ONE line:
{"description": "...", "tags": ["..."],
 "entities": [{"id": "person_1", "type": "person|person_group|object|organization|place|animal|text", "name": null,
   "name_source": null, "count": null, "description": "...", "evidence": "visual"}],
 "actions": [{"id": "a1", "type": "verb_noun", "description": "...", "participants": ["person_1"], "evidence": "visual"}],
 "relationships": [{"subject": "person_1", "predicate": "interviews", "object": "person_2", "evidence": "both"}],
 "context": {"setting": "...", "topic": null, "location": null, "notes": null},
 "attributes": {"action": "...", "shot_scale": "medium", "camera_motion": "static", "camera_angle": "eye_level",
   "cinematography": null}}"""

NULLS = {"unknown", "none", "null", "n/a", "na", "", "not specified", "not visible", "unspecified"}
ACTION_ALIASES = {"performing_ballet": "perform_ballet", "ballet_performance": "perform_ballet", "ballet": "perform_ballet",
                  "dancing": "dance", "perform_dance": "dance", "dance_performance": "dance", "interview": "give_interview",
                  "interviewing": "conduct_interview", "speaking": "speak", "talking": "speak", "converse": "converse",
                  "conversation": "converse", "listening": "listen", "singing": "sing", "applauding": "applaud",
                  "applause": "applaud", "cooking": "cook", "walking": "walk"}


def _null(v):
    if isinstance(v, str) and v.strip().lower() in NULLS:
        return None
    return v


def snake(s):
    return re.sub(r"[^a-z0-9]+", "_", str(s).strip().lower()).strip("_")


def normalise(obj: dict) -> tuple[dict, list[str]]:
    """Form-only repairs; returns (record, repairs log)."""
    rep = []
    o = dict(obj)
    for k in ("context", "attributes"):
        d = dict(o.get(k) or {})
        for kk, vv in list(d.items()):
            if _null(vv) is None and vv is not None:
                rep.append(f"{k}.{kk}: {vv!r} -> null")
                d[kk] = None
        o[k] = d
    if "time_of_day" in o["attributes"]:
        o["attributes"].pop("time_of_day")
        rep.append("attributes.time_of_day dropped (not in schema v2)")
    tags, seen = [], set()
    for t in o.get("tags") or []:
        n = " ".join(str(t).replace("_", " ").lower().split())
        if n and n not in seen:
            seen.add(n)
            tags.append(n)
    if len(tags) != len(o.get("tags") or []):
        rep.append("tags: lowercased/deduplicated")
    o["tags"] = tags[:12]
    camera(o["attributes"], rep)
    ents = []
    key = lambda x: " ".join(str(x).lower().split())  # noqa: E731
    named = Counter(key(e.get("name")) for e in o.get("entities") or [] if _null(e.get("name")))
    for e in o.get("entities") or []:
        e = dict(e)
        for kk in ("name", "name_source", "count", "description"):
            if kk in e and _null(e[kk]) is None and e[kk] is not None:
                e[kk] = None
        nm = e.get("name")
        why = (None if not nm else
               f"was given to {named[key(nm)]} entities" if named[key(nm)] > 1 else
               "without name_source" if not e.get("name_source") else
               f"name_source {e['name_source']!r} is not on_screen_text/transcript" if e["name_source"] not in NAME_SOURCES else
               "is a description, not a proper name" if not proper_name(nm) else None)
        if why:
            rep.append(f"name removed: entity {e.get('id')} name {nm!r} {why}")
            e["name"] = e["name_source"] = None
        elif not nm:
            e["name_source"] = None
        ents.append(e)
    o["entities"] = ents
    ids = {e.get("id") for e in ents}
    acts = []
    for a in o.get("actions") or []:
        a = dict(a)
        t = snake(a.get("type", ""))
        if ACTION_ALIASES.get(t, t) != t:
            rep.append(f"action type {t} -> {ACTION_ALIASES[t]}")
        a["type"] = ACTION_ALIASES.get(t, t)
        acts.append(a)
    o["actions"] = acts
    rels = []
    for r in o.get("relationships") or []:
        why = None
        if r.get("subject") not in ids or r.get("object") not in ids:
            why = "subject/object is not an entity id"
        elif r["subject"] == r["object"]:
            why = "subject equals object"
        elif snake(r.get("predicate")) not in PREDICATES:
            why = f"predicate {r.get('predicate')!r} not allowed"
        if why:
            rep.append(f"relationship {r.get('subject')} {r.get('predicate')} {r.get('object')} dropped: {why}")
        else:
            rels.append({**r, "predicate": snake(r["predicate"])})
    o["relationships"] = rels
    return o, rep


def proper_name(name: str) -> bool:
    """A proper name has at least one capital letter or digit (scripts without case always pass)."""
    cased = [ch for ch in name if ch.isalpha() and ch.lower() != ch.upper()]
    return not cased or any(ch.isupper() or ch.isdigit() for ch in name)


def _first_word(text, table):
    t = str(text or "").lower()
    hits = [(t.find(w), -len(w), v) for w, v in table if w in t]  # earliest mention; longest keyword on ties
    return min(hits)[2] if hits else None


def camera(att: dict, rep: list) -> None:
    """Closed camera fields: keep valid values, map the model's own words (in the field or in its cinematography
    note) onto the vocabulary, otherwise null. Never guesses from anything but the model's text."""
    note = att.get("cinematography")
    for key, vocab, table in (("shot_scale", SHOT_SCALES, SCALE_WORDS), ("camera_motion", CAMERA_MOTIONS, MOTION_WORDS),
                              ("camera_angle", CAMERA_ANGLES, ANGLE_WORDS)):
        v = att.get(key)
        sv = snake(v) if isinstance(v, str) else None
        if sv in vocab:
            att[key] = sv
            continue
        m = _first_word(v, table) or _first_word(note, table)
        if m or v is not None:
            rep.append(f"attributes.{key}: {v!r} -> {m!r}" + (" (from cinematography note)" if m and not _first_word(v, table) else ""))
        att[key] = m


def build_content(cfg, video, span, sources, transcript, src):
    c = cfg["level2"]
    a, b = span["start_s"], span["end_s"]
    n = int(min(c["max_frames_moment"], max(c["min_frames_moment"], round((b - a) / c["seconds_per_frame_moment"]))))
    content = [{"type": "text", "text": SCHEMA}]
    vc = video_context(src, c["use_video_context"])
    if vc:
        content.append({"type": "text", "text": vc})
    content.append({"type": "text", "text": f"The span is {a:g}-{b:g} s ({b - a:g} s). Automatic Level-1 evidence for its parts:\n"
                    + "\n".join(interval_text(m, f"Part {i + 1}") for i, m in enumerate(sources))})
    frames = grab(video, spread(a, b, n), c["frame_max_side"])
    for t, img in frames:
        content.append({"type": "text", "text": f"Frame at {t:.1f} s:"})
        content.append({"type": "image", "image": img, "t": t})
    content.append({"type": "text", "text": "Transcript of this span (automatic captions, may contain errors): "
                    + transcript_excerpt(transcript, a, b, c["transcript_chars"])})
    return content, [t for t, _ in frames]


def describe_batch(vlm, cfg, video, jobs, transcript, src):
    """jobs: [(span, sources)] -> [(obj, attempts, frame_times, repairs)]; one metadata call per span, batched."""
    c = cfg["level2"]
    bs = int(c.get("batch_size", 64))
    out = []
    for j in range(0, len(jobs), bs):  # frames are decoded per chunk to bound memory
        chunk = jobs[j:j + bs]
        built = [build_content(cfg, video, span, sources, transcript, src) for span, sources in chunk]
        items = [(content, c["max_new_tokens_metadata"], f"meta_{span['moment_id']}") for (span, _), (content, _) in zip(chunk, built)]
        for (content, frames), (obj, attempts) in zip(built, ask_batch(vlm, items, check_metadata, c["retries"])):
            repairs = []
            if obj is not None:
                obj, repairs = normalise(obj)
            out.append((obj, attempts, frames, repairs))
    return out


def describe(vlm, cfg, video, span, sources, transcript, src):
    return describe_batch(vlm, cfg, video, [(span, sources)], transcript, src)[0]
