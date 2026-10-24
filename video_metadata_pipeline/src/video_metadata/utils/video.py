"""Media probing via ffprobe (required on PATH) with an OpenCV fallback."""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path


def probe(path) -> dict:
    p = Path(path)
    if shutil.which("ffprobe"):
        r = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                            "stream=codec_type,codec_name,width,height,r_frame_rate:format=duration",
                            "-of", "json", str(p)], capture_output=True, text=True)
        if r.returncode == 0:
            d = json.loads(r.stdout)
            out = {"duration": float(d.get("format", {}).get("duration") or 0), "width": None, "height": None,
                   "fps": None, "has_audio": False, "video_codec": None}
            for s in d.get("streams", []):
                if s.get("codec_type") == "video" and out["width"] is None:
                    num, den = (s.get("r_frame_rate") or "0/1").split("/")
                    out.update(width=s.get("width"), height=s.get("height"), video_codec=s.get("codec_name"),
                               fps=round(float(num) / float(den), 3) if float(den) else None)
                if s.get("codec_type") == "audio":
                    out["has_audio"] = True
            out["prober"] = "ffprobe"
            return out
    import cv2
    cap = cv2.VideoCapture(str(p))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    n = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    out = {"duration": n / fps if fps else 0.0, "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
           "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), "fps": fps, "has_audio": None,
           "video_codec": None, "prober": "opencv"}
    cap.release()
    return out
