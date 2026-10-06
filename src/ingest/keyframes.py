"""(2c) Extract a small keyframe per shot and its perceptual hash (used for dedupe)."""
from __future__ import annotations

from pathlib import Path

import cv2
import imagehash
from PIL import Image


def extract(video: Path, shots: list[dict], out_dir: Path, video_id: str,
            max_side: int, quality: int) -> None:
    """Adds 'keyframe' (Path) and 'phash' (str) to each shot dict, in place."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    try:
        for s in shots:
            s["keyframe"], s["phash"] = None, None
            if not s["kept"]:
                continue
            cap.set(cv2.CAP_PROP_POS_MSEC, (s["start_s"] + s["end_s"]) / 2 * 1000)
            ok, frame = cap.read()
            if not ok:
                continue
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            img.thumbnail((max_side, max_side))
            path = out_dir / f"{video_id}__{s['shot_idx']:04d}.jpg"
            img.save(path, "JPEG", quality=quality)
            s["keyframe"], s["phash"] = path, str(imagehash.phash(img))
    finally:
        cap.release()
