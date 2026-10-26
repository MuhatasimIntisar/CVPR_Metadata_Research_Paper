"""Silences: ffmpeg silencedetect on the audio track (no model).
Boundary at the middle of each silence; confidence grows with silence length (1.0 at >= 3 s).
"""
from __future__ import annotations

import re
import subprocess
import time
from pathlib import Path

from .common import r3, write_result

NAME = "silence"


def detect(wav: Path, noise_db: float, min_s: float) -> list[tuple[float, float]]:
    p = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", str(wav), "-af",
                        f"silencedetect=noise={noise_db}dB:d={min_s}", "-f", "null", "-"],
                       capture_output=True, text=True)
    starts = [float(x) for x in re.findall(r"silence_start: (-?[\d.]+)", p.stderr)]
    ends = [float(x) for x in re.findall(r"silence_end: ([\d.]+)", p.stderr)]
    out = []
    for i, s in enumerate(starts):
        e = ends[i] if i < len(ends) else None
        out.append((max(0.0, s), e))
    return out


def run(video: Path, out_dir: Path, info: dict, wav: Path, noise_db: float = -35.0,
        min_silence_s: float = 0.7, **_) -> Path:
    t0 = time.time()
    spans = detect(wav, noise_db, min_silence_s)
    sil, bounds = [], []
    for s, e in spans:
        e = e if e is not None else info["duration"]
        d = e - s
        sil.append({"start": r3(s), "end": r3(e), "duration": r3(d)})
        bounds.append({"t": r3((s + e) / 2), "kind": "silence", "conf": r3(min(1.0, d / 3.0)),
                       "detail": f"{d:.2f}s"})
    return write_result(out_dir, NAME, "ffmpeg silencedetect",
                        {"noise_db": noise_db, "min_silence_s": min_silence_s}, t0,
                        stats={"n_silences": len(sil), "total_silence_s": r3(sum(x["duration"] for x in sil))},
                        boundaries=bounds, silences=sil)
