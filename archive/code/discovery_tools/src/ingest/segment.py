"""(2b) Shot segmentation.

Current: PySceneDetect ContentDetector (CPU, no model weights).
Planned for live: TransNetV2 — Souček & Lokoč, "TransNet V2: An effective deep network
architecture for fast shot transition detection", arXiv:2008.04838 (2020).
"""
from __future__ import annotations

from pathlib import Path

from scenedetect import ContentDetector, detect


def segment(video: Path, duration_s: float, threshold: float, min_shot_s: float,
            max_shot_s: float) -> list[dict]:
    scenes = detect(str(video), ContentDetector(threshold=threshold))
    spans = [(s.get_seconds(), e.get_seconds()) for s, e in scenes] or [(0.0, duration_s)]

    shots = []
    for start, end in spans:
        dur = end - start
        if dur < min_shot_s:
            shots.append({"start_s": start, "end_s": end, "kept": False, "discard_reason": "too_short"})
            continue
        n = max(1, int(-(-dur // max_shot_s)))  # ceil: split long shots into equal chunks
        step = dur / n
        for i in range(n):
            shots.append({"start_s": start + i * step, "end_s": start + (i + 1) * step,
                          "kept": True, "discard_reason": None})
    for i, s in enumerate(shots):
        s["shot_idx"] = i
        s["duration_s"] = round(s["end_s"] - s["start_s"], 3)
        s["start_s"], s["end_s"] = round(s["start_s"], 3), round(s["end_s"], 3)
    return shots
