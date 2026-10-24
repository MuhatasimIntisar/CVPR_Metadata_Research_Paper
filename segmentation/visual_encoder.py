"""M2 input: CLIP ViT-B/32 frame embeddings at `visual_fps` (cached)."""
from __future__ import annotations

import time

import numpy as np

from .media_utils import ffprobe, free_gpu, load_cache, model_revision, sample_frames, save_cache

NAME = "clip"


def params(cfg):
    return {"model": cfg.clip_model, "fps": cfg.visual_fps, "preprocessing": "CLIPProcessor defaults (resize 224, centre crop, normalise)"}


def encode(item, cfg, dirs, device, force=False, batch=64):
    if item.video is None:
        return None, {"status": "unavailable", "reason": "no video file"}
    arrays, meta = load_cache(dirs["cache"], item.video_id, NAME)
    if arrays is not None and not force and meta.get("params") == params(cfg):
        return arrays, meta
    import torch
    from transformers import CLIPModel, CLIPProcessor
    t0 = time.time()
    model = CLIPModel.from_pretrained(cfg.clip_model).to(device).eval()
    proc = CLIPProcessor.from_pretrained(cfg.clip_model)
    times, embs, buf = [], [], []

    def flush(model=model):
        with torch.no_grad():
            x = proc(images=buf, return_tensors="pt").to(device)
            e = model.get_image_features(**x)
            e = getattr(e, "pooler_output", e)  # transformers v5 returns an output object
            embs.append(torch.nn.functional.normalize(e.float(), dim=-1).cpu().numpy())
        buf.clear()

    for t, frame in sample_frames(item.video, cfg.visual_fps):
        times.append(t)
        buf.append(frame)
        if len(buf) == batch:
            flush()
    if buf:
        flush()
    rev = model_revision(model)
    del model
    free_gpu()
    info = ffprobe(item.video)
    meta = {"status": "ok", "modality": "visual", "params": params(cfg), "model_revision": rev,
            "n": len(times), "media_duration_s": (info["video"] or {}).get("duration", info["duration"]),
            "runtime_s": round(time.time() - t0, 2), "device": device}
    arrays = {"times": np.asarray(times, dtype=np.float32),
              "emb": np.concatenate(embs).astype(np.float32) if embs else np.zeros((0, 512), np.float32)}
    save_cache(dirs["cache"], item.video_id, NAME, meta, **arrays)
    return arrays, meta
