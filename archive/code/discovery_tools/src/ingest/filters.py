"""(2d) Cheap shot filters, run before any expensive annotation.

Each filter marks shots as discarded with a reason. Implemented now: too_short (in segment.py),
no_keyframe, near_duplicate. The others are stubs that keep everything and say so.
"""
from __future__ import annotations

import imagehash


def no_keyframe(shots: list[dict], cfg: dict) -> None:
    for s in shots:
        if s["kept"] and s.get("phash") is None:
            s["kept"], s["discard_reason"] = False, "no_keyframe"


def near_duplicate(shots: list[dict], cfg: dict) -> None:
    max_d = cfg["filters"]["phash_max_distance"]
    seen: list[imagehash.ImageHash] = []
    for s in shots:
        if not s["kept"]:
            continue
        h = imagehash.hex_to_hash(s["phash"])
        if any(h - prev <= max_d for prev in seen):
            s["kept"], s["discard_reason"] = False, "near_duplicate"
        else:
            seen.append(h)


def _stub(name: str, plan: str):
    def f(shots: list[dict], cfg: dict) -> None:
        print(f"      [filter stub] {name}: not implemented, keeping all ({plan})")
    f.__name__ = name
    return f


studio_anchor = _stub("studio_anchor", "cluster repeated similar shots within a video")
title_card = _stub("title_card", "OCR text-area ratio")
sponsor_segment = _stub("sponsor_segment", "SponsorBlock API timestamps")

FILTERS = [no_keyframe, near_duplicate, studio_anchor, title_card, sponsor_segment]


def apply_all(shots: list[dict], cfg: dict) -> None:
    for f in FILTERS:
        f(shots, cfg)
