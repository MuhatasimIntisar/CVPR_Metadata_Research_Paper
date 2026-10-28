"""Shot boundaries: PySceneDetect AdaptiveDetector (cuts, robust to camera motion)
+ ThresholdDetector (fades to/from black). Confidence = colour-histogram change across the cut.
"""
from __future__ import annotations

import time
from pathlib import Path

from .common import grab_frame, r3, write_result

NAME = "shots"


def _hist_change(video: Path, t: float, eps: float = 0.12) -> float:
    import cv2
    a, b = grab_frame(video, max(0.0, t - eps), 320), grab_frame(video, t + eps, 320)
    if a is None or b is None:
        return 0.0
    hs = []
    for f in (a, b):
        hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)
        h = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
        hs.append(cv2.normalize(h, h).flatten())
    corr = cv2.compareHist(hs[0], hs[1], cv2.HISTCMP_CORREL)
    return max(0.0, min(1.0, 1.0 - corr))


def _cuts(video: Path, detector) -> list[float]:
    from scenedetect import SceneManager, open_video
    v = open_video(str(video))
    sm = SceneManager()
    sm.add_detector(detector)
    sm.detect_scenes(v, show_progress=False)
    return [s.get_seconds() for s, _ in sm.get_scene_list()[1:]]


def run(video: Path, out_dir: Path, info: dict, adaptive_threshold: float = 3.0,
        min_scene_len_frames: int = 8, fade_threshold: int = 12, **_) -> Path:
    from scenedetect import AdaptiveDetector, ThresholdDetector
    t0 = time.time()
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(2) as ex:  # two independent decodes; OpenCV releases the GIL
        fc = ex.submit(_cuts, video, AdaptiveDetector(adaptive_threshold=adaptive_threshold,
                                                      min_scene_len=min_scene_len_frames))
        ff = ex.submit(_cuts, video, ThresholdDetector(threshold=fade_threshold, min_scene_len=min_scene_len_frames))
        cuts, fades = fc.result(), ff.result()

    bounds = [{"t": r3(t), "kind": "cut", "conf": r3(_hist_change(video, t))} for t in cuts]
    bounds += [{"t": r3(t), "kind": "fade", "conf": 1.0} for t in fades
               if all(abs(t - c) > 0.5 for c in cuts)]
    bounds.sort(key=lambda b: b["t"])

    edges = [0.0] + [b["t"] for b in bounds] + [info["duration"]]
    shots = [{"id": i, "start": r3(edges[i]), "end": r3(edges[i + 1]),
              "duration": r3(edges[i + 1] - edges[i])} for i in range(len(edges) - 1)]
    durs = sorted(s["duration"] for s in shots)
    stats = {"n_shots": len(shots), "median_shot_s": durs[len(durs) // 2] if durs else 0,
             "longest_shot_s": durs[-1] if durs else 0}
    return write_result(out_dir, NAME, "PySceneDetect AdaptiveDetector + ThresholdDetector",
                        {"adaptive_threshold": adaptive_threshold, "min_scene_len_frames": min_scene_len_frames,
                         "fade_threshold": fade_threshold}, t0,
                        stats=stats, boundaries=bounds, shots=shots)
