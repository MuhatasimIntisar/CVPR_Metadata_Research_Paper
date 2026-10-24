"""Text context for the VLM: candidate-interval evidence, boundary evidence, transcript excerpts, video-level context.

Every block says where it comes from, so the model (and a reviewer reading the cached prompt) can tell observed
detector evidence, transcript text and source metadata apart.
"""
from __future__ import annotations


def interval_text(m: dict, label: str) -> str:
    acts = ", ".join(f"{a['state']} {a['share_of_steps']:.0%}" for a in m["activities"][:4])
    subs = "; ".join(f"{s['state']} {s['start_s']:g}-{s['end_s']:g}s" for s in m["sub_events"][:5]) or "none"
    env = ", ".join(f"{e['label']} {e['share_of_steps']:.0%}" for e in m["environment"][:2]) or "unknown"
    au = m.get("audio") or {}
    return (f"{label} = candidate interval {m['interval_id']}, {m['start_s']:g}-{m['end_s']:g} s ({m['duration_s']:g} s), "
            f"automatic status '{m['status']}'.\n  Level-1 detector evidence (automatic, may be wrong): activity per 2 s "
            f"step: {acts}; sustained sub-events: {subs}; setting (CLIP): {env}; people visible (median): "
            f"{m.get('people_median')}; speech present {au.get('speech_share', 0) or 0:.0%} of the time, music "
            f"{au.get('music_share', 0) or 0:.0%}.")


def boundary_text(bid: str, t: float, b: dict) -> str:
    return (f"{bid} at {t:g} s: automatic boundary evidence = {'+'.join(b.get('types', []))} "
            f"(score {b.get('score')}); {b.get('reason', '')}")


def transcript_excerpt(tr: dict, a: float, b: float, chars: int) -> str:
    segs = [s for s in tr.get("segments", []) if s["end"] > a and s["start"] < b]
    if not segs:
        return "(no transcript in this span)"
    txt = " ".join(f"[{s['start']:.0f}s] {s['text']}" for s in segs)
    if len(txt) > chars:
        half = chars // 2
        txt = txt[:half] + " [...] " + txt[-half:]
    return txt


def video_context(src: dict, enabled: bool) -> str:
    if not enabled or not src:
        return ""
    d = (src.get("description") or "").replace("\n", " ")
    return ("Video-level source metadata (context only; it is NOT evidence that any particular moment shows these "
            f"things): title: {src.get('title')!r}; creator: {src.get('creator')}; date: {src.get('publication_date')}; "
            f"description: {d[:400]!r}; tags: {', '.join(src.get('subjects', [])[:10])}.")
