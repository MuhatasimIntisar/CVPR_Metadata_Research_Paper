"""Timestamp parsing/formatting for SRT ("00:01:02,345"), VTT ("00:01:02.345" or "01:02.345") and inline tags."""
from __future__ import annotations

import re

TS = re.compile(r"^(?:(\d+):)?(\d{1,2}):(\d{2})[.,](\d{1,3})$")


def parse_ts(s: str) -> float:
    m = TS.match(s.strip())
    if not m:
        raise ValueError(f"malformed timestamp: {s!r}")
    h, mi, se, ms = m.groups()
    return int(h or 0) * 3600 + int(mi) * 60 + int(se) + int(ms.ljust(3, "0")) / 1000.0


def fmt_ts(t: float) -> str:
    t = max(0.0, float(t))
    h, rem = divmod(t, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h):02d}:{int(m):02d}:{s:06.3f}"


def mmss(t: float) -> str:
    return f"{int(t // 60)}:{int(t % 60):02d}"
