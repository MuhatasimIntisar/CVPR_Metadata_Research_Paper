"""M1: PySceneDetect ContentDetector with its default threshold, native behaviour (no change-score
logic applied). Every detected cut is reported as a boundary."""
from __future__ import annotations

import time


def run(video, threshold: float, min_scene_len: int) -> dict:
    from scenedetect import ContentDetector, SceneManager, open_video
    t0 = time.time()
    v = open_video(str(video))
    sm = SceneManager()
    sm.add_detector(ContentDetector(threshold=threshold, min_scene_len=min_scene_len))
    sm.detect_scenes(v, show_progress=False)
    cuts = [s.get_seconds() for s, _ in sm.get_scene_list()[1:]]
    return {"boundaries_seconds": [round(c, 3) for c in cuts], "runtime_s": round(time.time() - t0, 2),
            "params": {"detector": "ContentDetector", "threshold": threshold, "min_scene_len_frames": min_scene_len}}
