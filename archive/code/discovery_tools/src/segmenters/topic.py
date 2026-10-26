"""Topic shifts in the transcript: TextTiling-style with multilingual sentence embeddings.

Each sentence is embedded (multilingual-e5-small). At every gap between sentences, the mean
embedding of the `block` sentences before is compared with the `block` after (cosine). The curve is
smoothed; a gap's depth = how far it dips below the nearest peaks on each side. Local depth maxima
above mean + `k_std` * std are reported as boundaries. The full curve is saved too. Needs asr.json.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from .common import free_gpu, r3, read_result, write_result

NAME = "topic"
MODEL = "intfloat/multilingual-e5-small"


def tiling(emb: np.ndarray, block: int, smooth: int):
    n = len(emb)
    sims = []
    for g in range(n - 1):  # gap after sentence g
        a = emb[max(0, g - block + 1):g + 1].mean(0)
        b = emb[g + 1:g + 1 + block].mean(0)
        sims.append(float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)))
    sims = np.array(sims)
    if smooth > 1 and len(sims) >= smooth:
        k = np.ones(smooth) / smooth
        pad = smooth // 2
        sims_s = np.convolve(np.pad(sims, pad, mode="edge"), k, mode="valid")[:len(sims)]
    else:
        sims_s = sims
    depth = np.zeros_like(sims_s)
    for i in range(len(sims_s)):
        l = i
        while l > 0 and sims_s[l - 1] >= sims_s[l]:
            l -= 1
        r = i
        while r < len(sims_s) - 1 and sims_s[r + 1] >= sims_s[r]:
            r += 1
        depth[i] = (sims_s[l] - sims_s[i]) + (sims_s[r] - sims_s[i])
    return sims, sims_s, depth


def run(video: Path, out_dir: Path, info: dict, device: str = "cpu", block: int = 3, smooth: int = 3,
        k_std: float = 0.5, **_) -> Path:
    from sentence_transformers import SentenceTransformer
    t0 = time.time()
    asr = read_result(out_dir, "asr")
    if asr is None:
        raise RuntimeError("topic needs asr.json; run the asr step first")
    sents = asr["sentences"]
    if len(sents) < 2 * block + 1:
        return write_result(out_dir, NAME, MODEL, {"block": block}, t0, stats={"note": "too few sentences"},
                            boundaries=[], gaps=[])
    model = SentenceTransformer(MODEL, device=device)
    emb = model.encode(["passage: " + s["text"] for s in sents], normalize_embeddings=True,
                       batch_size=64, show_progress_bar=False)
    del model
    free_gpu()
    sims, sims_s, depth = tiling(np.asarray(emb), block, smooth)
    cut = depth.mean() + k_std * depth.std()
    gaps, bounds = [], []
    dmax = float(depth.max()) or 1.0
    for g in range(len(depth)):
        t = (sents[g]["end"] + sents[g + 1]["start"]) / 2
        gaps.append({"t": r3(t), "after_sentence": g, "sim": r3(sims[g]), "sim_smoothed": r3(sims_s[g]),
                     "depth": r3(depth[g])})
        is_peak = (g == 0 or depth[g] >= depth[g - 1]) and (g == len(depth) - 1 or depth[g] >= depth[g + 1])
        if is_peak and depth[g] >= cut and depth[g] > 0:
            bounds.append({"t": r3(t), "kind": "topic_shift", "conf": r3(depth[g] / dmax),
                           "depth": r3(depth[g]),
                           "before": sents[g]["text"][-120:], "after": sents[g + 1]["text"][:120]})
    return write_result(out_dir, NAME, f"{MODEL} + TextTiling depth scores",
                        {"block": block, "smooth": smooth, "k_std": k_std}, t0,
                        stats={"n_sentences": len(sents), "depth_cutoff": r3(cut), "n_shifts": len(bounds)},
                        boundaries=bounds, gaps=gaps)
