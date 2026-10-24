"""On-screen text changes: EasyOCR on one frame per second.

Text that is on screen most of the time (station logo, channel bug, show title) is treated as a
static overlay and ignored. The remaining text forms a per-second "state"; a state must hold for
`min_stable` samples to count, and states are compared fuzzily (OCR spelling jitter). Every switch
between stable states is reported as a raw boundary: appear / disappear / change.
"""
from __future__ import annotations

import difflib
import re
import time
from pathlib import Path

from .common import free_gpu, iter_frames, r3, write_result

NAME = "ocr"


def norm(s: str) -> str:
    s = re.sub(r"[^\w\s]", " ", s.lower())
    return re.sub(r"\s+", " ", s).strip()


def same_state(a: tuple, b: tuple, thr: float = 0.8) -> bool:
    if not a and not b:
        return True
    if not a or not b:
        return False
    return difflib.SequenceMatcher(None, " | ".join(a), " | ".join(b)).ratio() >= thr


def change_events(samples: list[dict], overlay_share: float, min_stable: int) -> tuple[list[dict], list[str]]:
    counts = {}
    for s in samples:
        for k in {it["norm"] for it in s["items"]}:
            counts[k] = counts.get(k, 0) + 1
    overlay = sorted(k for k, c in counts.items() if c >= overlay_share * max(len(samples), 1))
    # fuzzy overlay match too, so "good morning delawar" counts as the overlay "good morning delaware"
    def is_overlay(k):
        return any(difflib.SequenceMatcher(None, k, o).ratio() >= 0.85 for o in overlay)

    states = []
    for s in samples:
        items = [it for it in s["items"] if not is_overlay(it["norm"])]
        states.append((s["t"], tuple(sorted({it["norm"] for it in items})), items))

    events, stable, run_start, run_len, cand = [], (), 0.0, 0, None
    for t, st, items in states:
        if cand is not None and same_state(st, cand[1]):
            run_len += 1
        else:
            cand, run_len, run_start = (t, st, items), 1, t
        if run_len == min_stable and not same_state(cand[1], stable):
            kind = "appear" if not stable else ("disappear" if not cand[1] else "change")
            lower = any(it["box"][1] >= 0.66 for it in cand[2])
            events.append({"t": r3(run_start), "kind": f"text_{kind}",
                           "conf": r3(sum(it["conf"] for it in cand[2]) / len(cand[2])) if cand[2] else 1.0,
                           "text": " | ".join(cand[1]), "previous": " | ".join(stable),
                           "lower_third": lower})
            stable = cand[1]
    return events, overlay


def run(video: Path, out_dir: Path, info: dict, device: str = "cpu", every_s: float = 1.0,
        langs: str = "en", min_conf: float = 0.4, overlay_share: float = 0.5, min_stable: int = 2, **_) -> Path:
    import easyocr
    t0 = time.time()
    reader = easyocr.Reader(langs.split(","), gpu=(device == "cuda"), verbose=False,
                            model_storage_directory=str(out_dir.parent / "_models" / "easyocr"))
    samples = []
    for t, frame in iter_frames(video, every_s, max_side=960):
        h, w = frame.shape[:2]
        items = []
        for box, text, conf in reader.readtext(frame):
            n = norm(text)
            if conf < min_conf or len(n) < 3:
                continue
            xs, ys = [p[0] for p in box], [p[1] for p in box]
            items.append({"text": text, "norm": n, "conf": r3(conf),
                          "box": [r3(min(xs) / w), r3(min(ys) / h), r3(max(xs) / w), r3(max(ys) / h)]})
        samples.append({"t": r3(t), "items": items})
    del reader
    free_gpu()
    ev, overlay = change_events(samples, overlay_share, min_stable)
    return write_result(out_dir, NAME, "EasyOCR", {"every_s": every_s, "langs": langs, "min_conf": min_conf,
                                                    "overlay_share": overlay_share, "min_stable": min_stable}, t0,
                        stats={"n_samples": len(samples), "n_changes": len(ev),
                               "samples_with_text": sum(1 for s in samples if s["items"])},
                        static_overlay=overlay, boundaries=ev, samples=samples)
