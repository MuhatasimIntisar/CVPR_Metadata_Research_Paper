"""Speaker turns without pyannote (no Hugging Face token needed).

1. Windows (1.5 s, hop 0.75 s) over the speech found by ASR.
2. ECAPA-TDNN speaker embedding per window (SpeechBrain, VoxCeleb).
3. Agglomerative clustering (cosine, average linkage, distance threshold) -> speaker ids.
4. Each word takes the majority label of the windows covering it; consecutive same-speaker
   words form a run; every change between runs is reported as a raw turn (no minimum hold applied).
Needs asr.json.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from .common import free_gpu, load_audio, r3, read_result, write_result

NAME = "speakers"
MODEL = "speechbrain/spkrec-ecapa-voxceleb"


def make_windows(sentences: list[dict], win: float = 1.5, hop: float = 0.75, min_win: float = 0.8,
                 audio_s: float | None = None):
    out = []
    for s in sentences:
        a, b = s["start"], s["end"]
        if audio_s is not None:
            a, b = max(0.0, min(a, audio_s)), max(0.0, min(b, audio_s))
            if b - a <= 0:
                continue
        if b - a < min_win:
            c = (a + b) / 2
            out.append((max(0.0, c - min_win / 2), c + min_win / 2))
            continue
        t = a
        while t + win < b:
            out.append((t, t + win))
            t += hop
        out.append((max(a, b - win), b))
    if audio_s is not None:
        out = [(max(0.0, x), min(y, audio_s)) for x, y in out]
    return [(x, y) for x, y in out if y - x >= 0.3]


def embed(audio: np.ndarray, sr: int, windows, device: str, batch: int = 64, cache_dir: Path | None = None):
    import torch
    try:
        from speechbrain.inference.speaker import EncoderClassifier
    except ImportError:  # speechbrain < 1.0
        from speechbrain.pretrained import EncoderClassifier
    kw = {"savedir": str(cache_dir or Path("models") / "spkrec-ecapa-voxceleb"), "run_opts": {"device": device}}
    try:  # Windows: copy files instead of symlinking (symlinks need admin/developer mode)
        from speechbrain.utils.fetching import LocalStrategy
        kw["local_strategy"] = LocalStrategy.COPY
    except ImportError:
        pass
    enc = EncoderClassifier.from_hparams(source=MODEL, **kw)
    embs = []
    with torch.no_grad():
        for i in range(0, len(windows), batch):
            chunk = [audio[int(a * sr):int(b * sr)] for a, b in windows[i:i + batch]]
            n = max(max(len(c) for c in chunk), 1)
            x = np.zeros((len(chunk), n), dtype=np.float32)
            lens = np.zeros(len(chunk), dtype=np.float32)
            for j, c in enumerate(chunk):
                x[j, :len(c)] = c
                lens[j] = max(len(c), 1) / n
            e = enc.encode_batch(torch.from_numpy(x).to(device), torch.from_numpy(lens).to(device))
            embs.append(e.squeeze(1).cpu().numpy())
    del enc
    free_gpu()
    e = np.concatenate(embs)
    return e / np.linalg.norm(e, axis=1, keepdims=True)


def cluster(emb: np.ndarray, threshold: float, min_share: float) -> np.ndarray:
    from sklearn.cluster import AgglomerativeClustering
    if len(emb) < 2:
        return np.zeros(len(emb), dtype=int)
    labels = AgglomerativeClustering(n_clusters=None, distance_threshold=threshold, metric="cosine",
                                     linkage="average").fit_predict(emb)
    # fold tiny clusters (noise, music, crosstalk) into the nearest real speaker
    counts = np.bincount(labels)
    big = [k for k in range(len(counts)) if counts[k] >= max(2, min_share * len(labels))] or [int(counts.argmax())]
    cents = {k: emb[labels == k].mean(0) for k in big}
    for i, l in enumerate(labels):
        if l not in cents:
            labels[i] = max(cents, key=lambda k: float(emb[i] @ cents[k]))
    # relabel by first appearance: S0, S1, ...
    order = {}
    for l in labels:
        order.setdefault(int(l), len(order))
    return np.array([order[int(l)] for l in labels])


def label_words(words: list[dict], windows, labels) -> list[int]:
    starts = np.array([a for a, _ in windows])
    ends = np.array([b for _, b in windows])
    centres = (starts + ends) / 2
    out = []
    for w in words:
        m = (w["start"] + w["end"]) / 2
        cover = np.where((starts <= m) & (ends >= m))[0]
        if len(cover):
            out.append(int(np.bincount(labels[cover]).argmax()))
        else:
            out.append(int(labels[int(np.abs(centres - m).argmin())]))
    return out


def runs_and_turns(words: list[dict], wl: list[int], cents: dict) -> tuple[list[dict], list[dict]]:
    runs = []
    for w, l in zip(words, wl):
        if runs and runs[-1]["speaker"] == f"S{l}":
            runs[-1]["end"] = w["end"]
            runs[-1]["n_words"] += 1
        else:
            runs.append({"speaker": f"S{l}", "start": w["start"], "end": w["end"], "n_words": 1})
    turns = []
    for a, b in zip(runs, runs[1:]):
        ka, kb = int(a["speaker"][1:]), int(b["speaker"][1:])
        sim = float(cents[ka] @ cents[kb] / (np.linalg.norm(cents[ka]) * np.linalg.norm(cents[kb])))
        turns.append({"t": r3((a["end"] + b["start"]) / 2), "kind": "speaker_turn",
                      "conf": r3(max(0.0, min(1.0, 1 - sim))),
                      "from": a["speaker"], "to": b["speaker"],
                      "prev_run_s": r3(a["end"] - a["start"]), "new_run_s": r3(b["end"] - b["start"]),
                      "gap_s": r3(b["start"] - a["end"])})
    for r in runs:
        r["start"], r["end"] = r3(r["start"]), r3(r["end"])
    return runs, turns


def run(video: Path, out_dir: Path, info: dict, wav: Path, device: str = "cpu",
        threshold: float = 0.7, min_cluster_share: float = 0.02, **_) -> Path:
    t0 = time.time()
    asr = read_result(out_dir, "asr")
    if asr is None:
        raise RuntimeError("speakers needs asr.json; run the asr step first")
    audio, sr = load_audio(wav)
    windows = make_windows(asr["sentences"], audio_s=len(audio) / sr)
    if not windows:
        raise RuntimeError("no speech windows found in asr.json")
    emb = embed(audio, sr, windows, device, cache_dir=out_dir.parent / "_models" / "spkrec-ecapa-voxceleb")
    labels = cluster(emb, threshold, min_cluster_share)
    cents = {k: emb[labels == k].mean(0) for k in set(labels.tolist())}
    wl = label_words(asr["words"], windows, labels)
    runs, turns = runs_and_turns(asr["words"], wl, cents)
    talk = {}
    for r in runs:
        talk[r["speaker"]] = talk.get(r["speaker"], 0) + r["end"] - r["start"]
    return write_result(out_dir, NAME, f"{MODEL} + agglomerative clustering",
                        {"window_s": 1.5, "hop_s": 0.75, "distance_threshold": threshold,
                         "min_cluster_share": min_cluster_share}, t0,
                        stats={"n_speakers": len(cents), "n_windows": len(windows), "n_turns": len(turns),
                               "talk_time_s": {k: r3(v) for k, v in sorted(talk.items())}},
                        boundaries=turns, runs=runs,
                        word_speakers=[f"S{l}" for l in wl])
