"""(2f) Parse YouTube .vtt captions and attach the text overlapping each shot."""
from __future__ import annotations

import re
from pathlib import Path

_TS = re.compile(r"(\d+):(\d{2}):(\d{2})\.(\d{3})\s+-->\s+(\d+):(\d{2}):(\d{2})\.(\d{3})")
_TAG = re.compile(r"<[^>]+>")


def _sec(h, m, s, ms) -> float:
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000


def parse_vtt(path: Path) -> list[tuple[float, float, list[str]]]:
    cues, cur, lines = [], None, []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines() + [""]:
        m = _TS.search(line)
        if m:
            cur, lines = (_sec(*m.groups()[:4]), _sec(*m.groups()[4:])), []
        elif not line.strip():
            if cur and lines:
                cues.append((cur[0], cur[1], lines))
            cur, lines = None, []
        elif cur:
            text = _TAG.sub("", line).strip()
            if text:
                lines.append(text)
    return cues


def attach(shots: list[dict], vtt: Path | None) -> None:
    """Adds 'caption_text' to each shot. Auto-captions repeat rolling lines, so lines are de-duplicated."""
    cues = parse_vtt(vtt) if vtt else []
    for s in shots:
        seen, parts = set(), []
        for start, end, cue_lines in cues:
            if start < s["end_s"] and end > s["start_s"]:
                for line in cue_lines:
                    if line not in seen:
                        seen.add(line)
                        parts.append(line)
        s["caption_text"] = " ".join(parts) or None
