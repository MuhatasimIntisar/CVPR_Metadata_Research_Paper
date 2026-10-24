"""M3 input: LAION-CLAP (HTSAT, via transformers) embeddings of overlapping audio windows (cached).
CLAP captures acoustic character (speech vs music vs crowd, room tone, voice timbre); it does not
understand discussion topics."""
from __future__ import annotations

import time

import numpy as np

from .media_utils import extract_audio, free_gpu, load_audio, load_cache, model_revision, save_cache

NAME = "clap"


def params(cfg):
    return {"model": cfg.clap_model, "window_s": cfg.audio_window_s, "hop_s": cfg.audio_hop_s,
            "sample_rate": cfg.audio_sr, "channels": "mono downmix (ffmpeg -ac 1)"}


def encode(item, cfg, dirs, device, force=False, batch=16):
    src = item.audio or item.video
    if src is None:
        return None, {"status": "unavailable", "reason": "no audio source"}
    arrays, meta = load_cache(dirs["cache"], item.video_id, NAME)
    if arrays is not None and not force and meta.get("params") == params(cfg):
        return arrays, meta
    import torch
    from transformers import ClapFeatureExtractor, ClapModel
    t0 = time.time()
    try:
        wav = extract_audio(src, dirs["cache"] / f"{item.video_id}_audio{cfg.audio_sr // 1000}k.wav", cfg.audio_sr)
    except Exception as e:  # noqa: BLE001 - e.g. a video without an audio stream
        return None, {"status": "unavailable", "reason": f"audio extraction failed: {e}"}
    audio, sr = load_audio(wav)
    dur = len(audio) / sr
    win, hop = cfg.audio_window_s, cfg.audio_hop_s
    starts = np.arange(0.0, max(dur - win, 0.0) + 1e-6, hop)
    fe = ClapFeatureExtractor.from_pretrained(cfg.clap_model)
    model = ClapModel.from_pretrained(cfg.clap_model).to(device).eval()
    embs = []
    with torch.no_grad():
        for i in range(0, len(starts), batch):
            chunk = [audio[int(s * sr):int((s + win) * sr)] for s in starts[i:i + batch]]
            x = fe(chunk, sampling_rate=sr, return_tensors="pt").to(device)
            e = model.get_audio_features(**x)
            e = getattr(e, "pooler_output", e)  # transformers v5 returns an output object
            embs.append(torch.nn.functional.normalize(e.float(), dim=-1).cpu().numpy())
    rev = model_revision(model)
    del model
    free_gpu()
    meta = {"status": "ok", "modality": "audio", "params": params(cfg), "model_revision": rev,
            "n": len(starts), "media_duration_s": round(dur, 3), "audio_source": str(src),
            "runtime_s": round(time.time() - t0, 2), "device": device}
    arrays = {"times": (starts + win / 2).astype(np.float32),
              "emb": np.concatenate(embs).astype(np.float32) if embs else np.zeros((0, 512), np.float32)}
    save_cache(dirs["cache"], item.video_id, NAME, meta, **arrays)
    return arrays, meta
