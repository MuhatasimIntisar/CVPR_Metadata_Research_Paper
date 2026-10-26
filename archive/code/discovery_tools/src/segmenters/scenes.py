"""Scene boundaries: group consecutive shots that show the same setting.

One CLIP image embedding per shot (middle frame). A shot joins the current scene if it is similar
to any of the scene's last `lookback` shots (handles A/B/A/B camera alternation in interviews);
otherwise it starts a new scene. Needs shots.json.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from .common import free_gpu, grab_frame, r3, read_result, write_result

NAME = "scenes"
MODEL = "openai/clip-vit-base-patch32"


def embed_shots(video: Path, shots: list[dict], device: str, batch: int = 32) -> np.ndarray:
    import cv2
    import torch
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor
    model = CLIPModel.from_pretrained(MODEL).to(device).eval()
    proc = CLIPProcessor.from_pretrained(MODEL)
    imgs = []
    for s in shots:
        f = grab_frame(video, (s["start"] + s["end"]) / 2, 384)
        imgs.append(Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) if f is not None
                    else Image.new("RGB", (224, 224)))
    out = []
    with torch.no_grad():
        for i in range(0, len(imgs), batch):
            x = proc(images=imgs[i:i + batch], return_tensors="pt").to(device)
            e = model.get_image_features(**x)
            e = getattr(e, "pooler_output", e)  # transformers v5 returns an output object
            out.append(torch.nn.functional.normalize(e, dim=-1).cpu().numpy())
    del model
    free_gpu()
    return np.concatenate(out) if out else np.zeros((0, 512))


def group(emb: np.ndarray, threshold: float, lookback: int) -> tuple[list[int], list[float]]:
    """Return scene id per shot and the best similarity that decided it."""
    scene_ids, best_sims, members = [], [], []
    for i in range(len(emb)):
        if i == 0:
            scene_ids.append(0)
            best_sims.append(1.0)
            members = [0]
            continue
        recent = members[-lookback:]
        sim = float(max(emb[i] @ emb[j] for j in recent))
        best_sims.append(sim)
        if sim >= threshold:
            scene_ids.append(scene_ids[-1])
            members.append(i)
        else:
            scene_ids.append(scene_ids[-1] + 1)
            members = [i]
    return scene_ids, best_sims


def run(video: Path, out_dir: Path, info: dict, device: str = "cpu", threshold: float = 0.85,
        lookback: int = 6, **_) -> Path:
    t0 = time.time()
    shots_doc = read_result(out_dir, "shots")
    if shots_doc is None:
        raise RuntimeError("scenes needs shots.json; run the shots segmenter first")
    shots = shots_doc["shots"]
    emb = embed_shots(video, shots, device)
    ids, sims = group(emb, threshold, lookback)

    scenes, bounds = [], []
    for i, s in enumerate(shots):
        if i == 0 or ids[i] != ids[i - 1]:
            scenes.append({"id": ids[i], "start": s["start"], "end": s["end"], "shot_ids": [s["id"]]})
            if i > 0:
                bounds.append({"t": s["start"], "kind": "scene_change", "conf": r3(1 - sims[i]),
                               "detail": f"best similarity to previous scene {sims[i]:.3f}"})
        else:
            scenes[-1]["end"] = s["end"]
            scenes[-1]["shot_ids"].append(s["id"])
    for sc in scenes:
        sc["duration"] = r3(sc["end"] - sc["start"])
    per_shot = [{"shot_id": s["id"], "scene_id": ids[i], "best_sim": r3(sims[i])} for i, s in enumerate(shots)]
    return write_result(out_dir, NAME, f"{MODEL} shot embeddings + lookback grouping",
                        {"threshold": threshold, "lookback": lookback}, t0,
                        stats={"n_scenes": len(scenes), "n_shots": len(shots)},
                        boundaries=bounds, scenes=scenes, per_shot=per_shot)
