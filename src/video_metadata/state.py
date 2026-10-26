"""Per-video checkpoint state (outputs/<video_id>/pipeline_state.json) for resume and status."""
from __future__ import annotations

import datetime as dt
from pathlib import Path

from .utils.io import read_json, write_json


def _now():
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


class State:
    def __init__(self, vdir: Path):
        self.path = Path(vdir) / "pipeline_state.json"
        self.doc = read_json(self.path, {"stages": {}})

    def stage(self, name):
        return self.doc["stages"].get(name, {})

    def done(self, name, fingerprint=None) -> bool:
        s = self.stage(name)
        return s.get("status") == "done" and (fingerprint is None or s.get("fingerprint") == fingerprint)

    def mark(self, name, status, **kw):
        s = self.doc["stages"].setdefault(name, {})
        s.update(status=status, updated=_now(), **kw)
        if status == "running":
            s["started"] = s["updated"]
            s.pop("error", None)
        write_json(self.path, self.doc)

    def invalidate_from(self, name, order):
        """A stage rerun makes every later stage stale."""
        if name in order:
            for later in order[order.index(name) + 1:]:
                if self.doc["stages"].get(later, {}).get("status") == "done":
                    self.doc["stages"][later]["status"] = "stale"
            write_json(self.path, self.doc)
