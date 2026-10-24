"""Semantic metadata for each consolidated moment (one VLM call per moment, frames spread across it)."""
from __future__ import annotations

from .context_builder import interval_text, transcript_excerpt, video_context
from .frame_sampler import grab, spread
from .output_validator import ask, check_metadata

SCHEMA = """Describe ONE moment of a video for a moment-retrieval dataset. Use only what the frames show and the
transcript says. If something cannot be determined, use null, "unknown" or an empty list; never guess identities.
People whose identity is not shown on screen or stated in the transcript get anonymous ids (person_1, group_1).
Only state a relationship when the frames or transcript support it (appearing together is not enough).
Mark each entity, event and relationship with evidence "visual", "transcript" or "both".
Answer with ONLY this JSON object:
{"description": "<1-3 factual sentences>",
 "tags": ["<searchable keyword>", ...],
 "entities": [{"id": "person_1", "type": "person|person_group|object|organization|place|animal|text", "name": null,
               "description": "<visible attributes>", "evidence": "visual|transcript|both"}],
 "events": [{"type": "<main event or activity, snake_case>", "description": "<short>", "participants": ["<entity id>"],
             "evidence": "visual|transcript|both"}],
 "relationships": [{"subject": "<entity id>", "predicate": "<verb phrase>", "object": "<entity id or value>",
                    "evidence": "visual|transcript|both"}],
 "context": {"setting": "<physical setting or unknown>", "topic": "<conversation topic or null>",
             "location": "<place if shown/stated, else null>", "notes": "<other supported context or null>"},
 "attributes": {"action": "<main visible action>", "time_of_day": "<if visible, else null>",
                "cinematography": "<framing/camera, e.g. static wide shot, cuts between close-ups>"}}"""


def describe(vlm, cfg, video, moment, sources, transcript, src):
    c = cfg["level2"]
    a, b = moment["start_s"], moment["end_s"]
    n = int(min(c["max_frames_moment"], max(c["min_frames_moment"], round((b - a) / c["seconds_per_frame_moment"]))))
    times = spread(a, b, n)
    content = [{"type": "text", "text": SCHEMA}]
    vc = video_context(src, c["use_video_context"])
    if vc:
        content.append({"type": "text", "text": vc})
    content.append({"type": "text", "text": f"The moment spans {a:g}-{b:g} s ({b - a:g} s). Automatic Level-1 evidence for "
                    "its parts:\n" + "\n".join(interval_text(m, f"Part {i + 1}") for i, m in enumerate(sources))})
    frames = grab(video, times, c["frame_max_side"])
    for t, img in frames:
        content.append({"type": "text", "text": f"Frame at {t:.1f} s:"})
        content.append({"type": "image", "image": img, "t": t})
    content.append({"type": "text", "text": "Transcript of the moment (automatic captions, may contain errors): "
                    + transcript_excerpt(transcript, a, b, c["transcript_chars"])})
    obj, attempts = ask(vlm, content, c["max_new_tokens_metadata"], check_metadata, c["retries"], f"meta_{moment['moment_id']}")
    return obj, attempts, [t for t, _ in frames]
