"""(2e) Cut kept shots into small standalone clips with ffmpeg."""
from __future__ import annotations

import subprocess
from pathlib import Path


def cut(video: Path, shot: dict, out_path: Path, max_height: int, keep_audio: bool) -> None:
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
           "-ss", f"{shot['start_s']:.3f}", "-i", str(video), "-t", f"{shot['duration_s']:.3f}",
           "-vf", f"scale=-2:'min({max_height},ih)'",
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "28", "-pix_fmt", "yuv420p"]
    cmd += ["-c:a", "aac", "-b:a", "64k", "-ac", "1"] if keep_audio else ["-an"]
    cmd += ["-movflags", "+faststart", str(out_path)]
    subprocess.run(cmd, check=True)
