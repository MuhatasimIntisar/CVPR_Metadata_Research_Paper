"""Rule B (top-1 activity evidence): one state per 2 s step, never promoted from lower-ranked labels.

state = 'unobservable' (black frame) | 'no person visible' (YOLO sees nobody) | top-1 activity label when its score is
>= dominant_rel x uniform AND >= top1_margin_ratio x the runner-up | 'uncertain' otherwise. All model scores stay in
the observation record; the state is evidence, not a verified label.
"""
from __future__ import annotations

UNOBS, NOPERSON, UNCERTAIN = "unobservable", "no person visible", "uncertain"


def step_state(obs: dict, c: dict) -> tuple[str, str]:
    if obs["black_frame"]:
        return UNOBS, "black frame (luminance)"
    p = obs["people"]
    if p["n_frames"] and (p["count_median"] or 0) == 0:
        return NOPERSON, "YOLO detected no person"
    a = obs["activity"]
    if a["top1_rel"] >= c["dominant_rel"] and a["top1_margin_ratio"] >= c["top1_margin_ratio"]:
        return a["top1"], f"top-1 {a['top1_rel']}x uniform, {a['top1_margin_ratio']}x runner-up"
    return UNCERTAIN, f"top-1 {a['top1']} too weak ({a['top1_rel']}x uniform, {a['top1_margin_ratio']}x runner-up)"
