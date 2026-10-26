"""Level-1 v2 inference (GPU machine): revised, independent classification heads + overlay masking.

    python -m level1.detect_v2 --video-id detvde-Chocolate_Ballerina_Founder_Chane_5b4f44

Reruns ONLY what the revised label sets / masking require (CLIP, X-CLIP). YOLO, pose, VAD, OCR,
diarisation and audio events are reused unchanged. Every score is computed twice - on the original frame
and on the overlay-masked frame - so the two can be compared before committing to one.

Outputs (data/level1/<id>/v2/, v1 files untouched):
  overlay_mask.npy / overlay_mask.png   persistent-overlay mask (logos, bugs) estimated from the whole video
  frame_stats.npz                       4 fps luminance mean/std (black-frame detection, no model)
  clip_v2.npz                           CLIP frames every 5 s: environment + presentation heads, both variants
  xclip_v2.npz                          X-CLIP 8-frame (2 s) clips: physical-activity head, both variants
  detect_v2_meta.json                   label sets, prompts, mask method, timings
Scores are softmax-normalised similarities WITHIN one head; they are relative scores, not calibrated
probabilities of real-world occurrence.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .detect import BASE_FPS, CLIP, CLIP_EVERY_S, XCLIP, XCLIP_FRAMES, decode, device, paths

# ---- independent heads (each a closed vocabulary; properties of different kinds never compete)
ACTIVITY = ["talking while seated", "talking to the camera", "talking while standing", "dancing",
            "playing a musical instrument", "singing", "walking", "marching in a parade",
            "cooking or preparing food", "eating or drinking", "playing a sport", "exercising",
            "applauding", "demonstrating or presenting an object", "working with tools", "shopping", "driving"]
ENVIRONMENT = ["a television studio", "a stage or auditorium", "an office", "a meeting room or council chamber",
               "a classroom", "a residential interior", "a kitchen", "a shop, restaurant or cafe",
               "a gym or dance studio", "a gallery or museum", "a sports venue", "a city street",
               "a park or garden", "a waterfront or beach", "countryside or forest", "a vehicle interior",
               "another indoor place", "another outdoor place"]
PRESENTATION = ["live camera footage of a real place", "a full-screen graphic or title card",
                "a black or blank screen", "a split screen or picture-in-picture",
                "a still photograph shown on screen"]
ACT_PROMPT, ENV_PROMPT, PRES_PROMPT = "a video of people {}", "a photo taken in {}", "{}"

MASK = {"sample_fps": 1.0, "width": 160, "diff_thr": 10, "stable_share": 0.85, "edge_thr": 25,
        "border": 0.22, "dilate": 3, "ocr_pad": 0.015}


def static_overlay_mask(video):
    """Pixels that stay identical across the whole video (despite shot changes) and carry edges, restricted
    to the border band where broadcast bugs/logos live. Deterministic, no model."""
    import cv2
    frames = []
    for _t, rgb in decode(video, MASK["sample_fps"]):
        g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        h, w = g.shape
        frames.append(cv2.resize(g, (MASK["width"], int(h * MASK["width"] / w)), interpolation=cv2.INTER_AREA))
    F = np.stack(frames).astype(np.int16)
    med = np.median(F, 0)
    stable = (np.abs(F - med) < MASK["diff_thr"]).mean(0) >= MASK["stable_share"]
    m8 = med.astype(np.uint8)
    edges = np.hypot(cv2.Sobel(m8, cv2.CV_32F, 1, 0), cv2.Sobel(m8, cv2.CV_32F, 0, 1)) > MASK["edge_thr"]
    h, w = med.shape
    b = MASK["border"]
    border = np.zeros_like(stable)
    border[: int(h * b)] = border[int(h * (1 - b)):] = True
    border[:, : int(w * b)] = border[:, int(w * (1 - b)):] = True
    seed = stable & edges & border
    k = np.ones((2 * MASK["dilate"] + 1,) * 2, np.uint8)
    mask = cv2.dilate(cv2.morphologyEx(seed.astype(np.uint8), cv2.MORPH_CLOSE, k), k) > 0
    mask &= stable  # do not grow into changing content
    return mask, len(frames)


class Masker:
    def __init__(self, static_mask, ocr):
        self.static = static_mask
        self.ocr_t = np.array([s["t"] for s in ocr["samples"]]) if ocr else np.array([])
        self.ocr = ocr["samples"] if ocr else []

    def boxes(self, t):
        if not len(self.ocr_t):
            return []
        i = int(np.argmin(np.abs(self.ocr_t - t)))
        return [it["box"] for it in self.ocr[i]["items"]] if abs(self.ocr_t[i] - t) <= 1.0 else []

    def apply(self, rgb, t):
        import cv2
        h, w = rgb.shape[:2]
        m = cv2.resize(self.static.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
        p = MASK["ocr_pad"]
        for x1, y1, x2, y2 in self.boxes(t):
            m[max(0, int((y1 - p) * h)):int((y2 + p) * h), max(0, int((x1 - p) * w)):int((x2 + p) * w)] = True
        if not m.any() or m.all():
            return rgb, float(m.mean())
        out = rgb.copy()
        out[m] = np.median(rgb[~m], axis=0).astype(rgb.dtype)
        return out, float(m.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video-id", required=True)
    ap.add_argument("--root", default=".")
    ap.add_argument("--clip-every-s", type=float, default=CLIP_EVERY_S)
    a = ap.parse_args()
    video, seg_dir, l1 = paths(Path(a.root), a.video_id)
    out = l1 / "v2"
    out.mkdir(parents=True, exist_ok=True)
    import cv2
    import torch
    from transformers import CLIPModel, CLIPProcessor, XCLIPModel, XCLIPProcessor
    dev = device()
    timing = {}
    t0 = time.time()
    smask, n_mask_frames = static_overlay_mask(video)
    np.save(out / "overlay_mask.npy", smask)
    cv2.imwrite(str(out / "overlay_mask.png"), (smask * 255).astype(np.uint8))
    timing["overlay_mask_s"] = round(time.time() - t0, 1)
    ocr_p = seg_dir / "ocr.json"
    masker = Masker(smask, json.loads(ocr_p.read_text(encoding="utf-8")) if ocr_p.exists() else None)

    cp, cm = CLIPProcessor.from_pretrained(CLIP), CLIPModel.from_pretrained(CLIP).to(dev).eval()
    xp, xm = XCLIPProcessor.from_pretrained(XCLIP), XCLIPModel.from_pretrained(XCLIP).to(dev).eval()

    def text_feats(labels, tmpl):
        with torch.no_grad():
            x = cp(text=[tmpl.format(l) for l in labels], return_tensors="pt", padding=True).to(dev)
            e = cm.get_text_features(**x)
            return torch.nn.functional.normalize(getattr(e, "pooler_output", e).float(), dim=-1)

    T_env, T_pres = text_feats(ENVIRONMENT, ENV_PROMPT), text_feats(PRESENTATION, PRES_PROMPT)
    act_prompts = [ACT_PROMPT.format(x) for x in ACTIVITY]
    xc = {"times": [], "raw": [], "masked": [], "mask_share": []}
    cl = {"times": [], "raw": [], "masked": [], "mask_share": []}
    fs = {"t": [], "luma_mean": [], "luma_std": []}
    buf_raw, buf_msk, buf_share, clips_raw, clips_msk = [], [], [], [], []
    next_clip = a.clip_every_s / 2
    t_x = t_c = 0.0

    def run_xclip():
        nonlocal t_x
        t1 = time.time()
        with torch.no_grad():
            for key, clips in (("raw", clips_raw), ("masked", clips_msk)):
                x = xp(text=act_prompts, videos=clips, return_tensors="pt", padding=True).to(dev)
                xc[key].append(xm(**x).logits_per_video.softmax(-1).float().cpu().numpy())
        clips_raw.clear()
        clips_msk.clear()
        t_x += time.time() - t1

    for t, rgb in decode(video, BASE_FPS, short_side=256):
        g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        fs["t"].append(t)
        fs["luma_mean"].append(float(g.mean()))
        fs["luma_std"].append(float(g.std()))
        mrgb, share = masker.apply(rgb, t)
        buf_raw.append(rgb)
        buf_msk.append(mrgb)
        buf_share.append(share)
        if len(buf_raw) == XCLIP_FRAMES:
            clips_raw.append(list(buf_raw))
            clips_msk.append(list(buf_msk))
            xc["times"].append(t - (XCLIP_FRAMES - 1) / BASE_FPS / 2)
            xc["mask_share"].append(float(np.mean(buf_share)))
            buf_raw.clear()
            buf_msk.clear()
            buf_share.clear()
            if len(clips_raw) == 8:
                run_xclip()
        if t >= next_clip:
            t1 = time.time()
            with torch.no_grad():
                for key, img in (("raw", rgb), ("masked", mrgb)):
                    ie = cm.get_image_features(**cp(images=[img], return_tensors="pt").to(dev))
                    ie = torch.nn.functional.normalize(getattr(ie, "pooler_output", ie).float(), dim=-1)
                    s = cm.logit_scale.exp()
                    cl[key].append(np.concatenate([(s * ie @ T_env.T).softmax(-1).cpu().numpy()[0],
                                                   (s * ie @ T_pres.T).softmax(-1).cpu().numpy()[0],
                                                   ie.cpu().numpy()[0]]))
            cl["times"].append(t)
            cl["mask_share"].append(share)
            next_clip += a.clip_every_s
            t_c += time.time() - t1
        if len(fs["t"]) % 400 == 0:
            print(f"\r  {t:7.1f}s", end="", flush=True)
    print()
    if clips_raw:
        run_xclip()
    ne, npr = len(ENVIRONMENT), len(PRESENTATION)
    np.savez_compressed(out / "xclip_v2.npz", times=np.float32(xc["times"]), raw=np.concatenate(xc["raw"]),
                        masked=np.concatenate(xc["masked"]), mask_share=np.float32(xc["mask_share"]))
    R, M = np.stack(cl["raw"]), np.stack(cl["masked"])
    np.savez_compressed(out / "clip_v2.npz", times=np.float32(cl["times"]), env_raw=R[:, :ne],
                        env_masked=M[:, :ne], pres_raw=R[:, ne:ne + npr], pres_masked=M[:, ne:ne + npr],
                        emb_raw=R[:, ne + npr:], emb_masked=M[:, ne + npr:], mask_share=np.float32(cl["mask_share"]))
    np.savez_compressed(out / "frame_stats.npz", **{k: np.float32(v) for k, v in fs.items()})
    timing.update(xclip_s=round(t_x, 1), clip_s=round(t_c, 1), total_s=round(time.time() - t0, 1))
    meta = {"video": str(video), "device": dev, "models": {"clip": CLIP, "xclip": XCLIP},
            "heads": {"activity": ACTIVITY, "environment": ENVIRONMENT, "presentation": PRESENTATION},
            "prompts": {"activity": ACT_PROMPT, "environment": ENV_PROMPT, "presentation": PRES_PROMPT},
            "sampling": {"clip_every_s": a.clip_every_s, "xclip_clip_s": XCLIP_FRAMES / BASE_FPS, "base_fps": BASE_FPS},
            "mask": {**MASK, "frames_used": n_mask_frames, "static_mask_share": float(smask.mean()),
                     "method": "static border overlay (whole-video pixel stability + edges) + per-frame OCR boxes, "
                               "filled with the frame's median colour"},
            "score_note": "softmax within each head = relative score, not a calibrated probability",
            "timing_s": timing}
    (out / "detect_v2_meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print("done:", out, timing, f"static mask covers {smask.mean():.1%} of the frame")


if __name__ == "__main__":
    main()
