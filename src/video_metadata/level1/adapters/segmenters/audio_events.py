"""Audio events: Audio Spectrogram Transformer fine-tuned on AudioSet (527 classes, multi-label).

2 s windows every 1 s. AudioSet classes are grouped into a few event types; each group's score is
the max sigmoid probability of its classes. A group switches on above `on` and off below `off`
(hysteresis); every on/off switch is reported as a raw boundary.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from .common import free_gpu, load_audio, r3, write_result

NAME = "audio_events"
MODEL = "MIT/ast-finetuned-audioset-10-10-0.4593"

GROUPS = {
    "speech": ["Speech", "Conversation", "Narration, monologue", "Male speech, man speaking",
               "Female speech, woman speaking"],
    "music": ["Music", "Musical instrument", "Singing"],
    "applause": ["Applause", "Clapping"],
    "cheering": ["Cheering", "Crowd", "Yell", "Children shouting"],
    "laughter": ["Laughter", "Giggle", "Chuckle, chortle", "Belly laugh"],
    "whistle": ["Whistle", "Whistling"],
    "bell_gavel": ["Bell", "Ding", "Knock", "Tap"],
    "silence": ["Silence"],
}


def window_scores(audio, sr, device, win=2.0, hop=1.0, batch=16):
    import torch
    from transformers import ASTFeatureExtractor, ASTForAudioClassification
    fe = ASTFeatureExtractor.from_pretrained(MODEL)
    model = ASTForAudioClassification.from_pretrained(MODEL).to(device).eval()
    names = [model.config.id2label[i] for i in range(model.config.num_labels)]
    idx = {g: [names.index(n) for n in ns if n in names] for g, ns in GROUPS.items()}
    starts = np.arange(0, max(len(audio) / sr - win, 0) + 1e-6, hop)
    rows = []
    with torch.no_grad():
        for i in range(0, len(starts), batch):
            chunk = [audio[int(s * sr):int((s + win) * sr)] for s in starts[i:i + batch]]
            x = fe(chunk, sampling_rate=sr, return_tensors="pt").to(device)
            p = torch.sigmoid(model(**x).logits).cpu().numpy()
            for s, pr in zip(starts[i:i + batch], p):
                top = np.argsort(pr)[::-1][:3]
                rows.append({"t": r3(s + win / 2),
                             "groups": {g: r3(pr[ix].max()) if ix else 0.0 for g, ix in idx.items()},
                             "top": [[names[k], r3(pr[k])] for k in top]})
    del model
    free_gpu()
    return rows


def events(rows, hop: float, on: float, off: float) -> list[dict]:
    out = []
    for g in GROUPS:
        active, peak, start = False, 0.0, None
        for r in rows:
            v = r["groups"][g]
            if not active and v >= on:
                active, peak, start = True, v, r["t"] - hop / 2
            elif active:
                peak = max(peak, v)
                if v < off:
                    out.append({"t": r3(start), "kind": f"{g}_start", "conf": r3(peak), "group": g})
                    out.append({"t": r3(r["t"] - hop / 2), "kind": f"{g}_end", "conf": r3(peak), "group": g,
                                "run_s": r3(r["t"] - hop / 2 - start)})
                    active = False
        if active:
            out.append({"t": r3(start), "kind": f"{g}_start", "conf": r3(peak), "group": g})
    return sorted(out, key=lambda e: e["t"])


def run(video: Path, out_dir: Path, info: dict, wav: Path, device: str = "cpu",
        on: float = 0.3, off: float = 0.15, **_) -> Path:
    t0 = time.time()
    audio, sr = load_audio(wav)
    rows = window_scores(audio, sr, device)
    ev = events(rows, 1.0, on, off)
    cover = {g: r3(np.mean([r["groups"][g] >= on for r in rows])) if rows else 0 for g in GROUPS}
    return write_result(out_dir, NAME, MODEL, {"window_s": 2.0, "hop_s": 1.0, "on": on, "off": off,
                                                "groups": GROUPS}, t0,
                        stats={"n_windows": len(rows), "share_of_windows_active": cover},
                        boundaries=ev, windows=rows)
