"""Human overrides from review (authoritative; never reversed by automatic reprocessing).

Read from <paths.review_root>/<video_id>/overrides.json (tracked in git; falls back to outputs/<id>/review/):
  {"activity_overrides": [{"start": s, "end": e, "state": "dancing", "edit_id": ...}],
   "forced_boundaries":  [{"t": s, "edit_id": ...}],
   "removed_boundaries": [{"t": s, "tolerance_s": 2.0, "edit_id": ...}]}
"""
from __future__ import annotations

from pathlib import Path

from ..utils.io import read_json

EMPTY = {"activity_overrides": [], "forced_boundaries": [], "removed_boundaries": []}


def load(vdir: Path, review_dir: Path | None = None) -> dict:
    o = (read_json(Path(review_dir) / "overrides.json") if review_dir else None) or \
        read_json(Path(vdir) / "review" / "overrides.json") or {}
    return {k: o.get(k, []) for k in EMPTY}


def apply_activity(states, times, overrides, log):
    for o in overrides["activity_overrides"]:
        for k, t in enumerate(times):
            if o["start"] <= t < o["end"]:
                if states[k] != o["state"]:
                    log.append({"rule": "human_override", "t": t, "from": states[k], "to": o["state"], "edit_id": o.get("edit_id")})
                states[k] = o["state"]
    return states
