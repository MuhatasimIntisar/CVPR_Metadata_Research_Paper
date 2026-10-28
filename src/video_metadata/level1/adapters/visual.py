"""YOLO detection + tracking + pose, and the CLIP/X-CLIP classification heads with overlay masking.

Packaged from level1/detect.py and level1/detect_v2.py of the research code. One sequential decode at base_fps
feeds every model. Raw, timestamped outputs only:
  yolo_track.json  [class, conf, track_id, x1, y1, x2, y2] per frame (normalised boxes), 2 fps, ByteTrack
  yolo_pose.json   COCO-17 keypoints per person per frame
  clip_v2.npz      CLIP every clip_every_s: environment + presentation heads and embeddings (raw + masked frames)
  xclip_v2.npz     X-CLIP 8-frame clips (2 s): physical-activity head (raw + masked frames)
  frame_stats.npz  luminance mean/std at base_fps (black-frame detection)
  overlay_mask.npy persistent-overlay mask
Scores are softmax within one head: relative model scores, not calibrated probabilities.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

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
                "a black or blank screen", "a split screen or picture-in-picture", "a still photograph shown on screen"]
PROMPTS = {"activity": "a video of people {}", "environment": "a photo taken in {}", "presentation": "{}"}
MASK = {"sample_fps": 1.0, "width": 160, "diff_thr": 10, "stable_share": 0.85, "edge_thr": 25,
        "border": 0.22, "dilate": 3, "ocr_pad": 0.015}


def decode(video: Path, fps: float, short_side: int = 480):
    import cv2
    cap = cv2.VideoCapture(str(video))
    vfps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    k, idx = 0, 0
    while cap.grab():  # decode every frame, but convert/resize only the sampled ones
        t = idx / vfps
        if t + 0.5 / vfps >= (k + 0.5) / fps:
            ok, frame = cap.retrieve()
            if not ok:
                break
            h, w = frame.shape[:2]
            s = short_side / min(h, w)
            if s < 1:
                frame = cv2.resize(frame, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
            yield (k + 0.5) / fps, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            k += 1
        idx += 1
    cap.release()


def static_overlay_mask(video):
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
    k = np.ones((2 * MASK["dilate"] + 1,) * 2, np.uint8)
    mask = cv2.dilate(cv2.morphologyEx((stable & edges & border).astype(np.uint8), cv2.MORPH_CLOSE, k), k) > 0
    return mask & stable, len(frames)


class Masker:
    def __init__(self, static_mask, ocr):
        self.static = static_mask
        self.samples = (ocr or {}).get("samples", [])
        self.t = np.array([s["t"] for s in self.samples])

    def apply(self, rgb, t):
        import cv2
        h, w = rgb.shape[:2]
        m = cv2.resize(self.static.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
        if len(self.t):
            i = int(np.argmin(np.abs(self.t - t)))
            if abs(self.t[i] - t) <= 1.0:
                p = MASK["ocr_pad"]
                for it in self.samples[i]["items"]:
                    x1, y1, x2, y2 = it["box"]
                    m[max(0, int((y1 - p) * h)):int((y2 + p) * h), max(0, int((x1 - p) * w)):int((x2 + p) * w)] = True
        if not m.any() or m.all():
            return rgb, float(m.mean())
        out = rgb.copy()
        out[m] = np.median(rgb[~m], axis=0).astype(rgb.dtype)
        return out, float(m.mean())


def r(x):
    return round(float(x), 3)


def run_yolo(video: Path, raw: Path, cfg: dict, device: str) -> dict:
    from ultralytics import YOLO
    yc = cfg["level1"]["yolo"]
    det, pose = YOLO(yc["detect_model"]), YOLO(yc["pose_model"])
    t0, track, poses = time.time(), [], []
    for t, rgb in decode(video, yc["fps"]):
        bgr = np.ascontiguousarray(rgb[:, :, ::-1])
        res = det.track(bgr, persist=True, tracker="bytetrack.yaml", conf=yc["conf"], verbose=False, device=device)[0]
        track.append({"t": r(t), "dets": [[res.names[int(b.cls)], r(b.conf), int(b.id) if b.id is not None else -1,
                                           *[r(v) for v in b.xyxyn[0].tolist()]] for b in res.boxes]})
        pr = pose.predict(bgr, conf=yc["conf"], verbose=False, device=device)[0]
        persons = []
        if pr.keypoints is not None:
            kx = pr.keypoints.xyn.cpu().numpy()
            kc = pr.keypoints.conf.cpu().numpy() if pr.keypoints.conf is not None else np.ones(kx.shape[:2])
            for j, b in enumerate(pr.boxes):
                persons.append({"conf": r(b.conf), "box": [r(v) for v in b.xyxyn[0].tolist()],
                                "kpts": [[r(a), r(c), r(d)] for (a, c), d in zip(kx[j], kc[j])]})
        poses.append({"t": r(t), "persons": persons})
    (raw / "yolo_track.json").write_text(json.dumps({"model": yc["detect_model"], "tracker": "bytetrack.yaml", "fps": yc["fps"],
        "conf": yc["conf"], "format": "[class, conf, track_id, x1, y1, x2, y2] (normalised)", "frames": track},
        separators=(",", ":")), encoding="utf-8")
    (raw / "yolo_pose.json").write_text(json.dumps({"model": yc["pose_model"], "fps": yc["fps"], "conf": yc["conf"],
        "format": "kpts = COCO-17 [x, y, conf] (normalised)", "frames": poses}, separators=(",", ":")), encoding="utf-8")
    return {"runtime_s": round(time.time() - t0, 1), "frames": len(track)}


def run_vision_language(video: Path, raw: Path, cfg: dict, device: str) -> dict:
    import torch
    from transformers import CLIPModel, CLIPProcessor, XCLIPModel, XCLIPProcessor
    vc = cfg["level1"]["vision_language"]
    t0 = time.time()
    smask, n_mask = static_overlay_mask(video)
    np.save(raw / "overlay_mask.npy", smask)
    ocr = json.loads((raw / "ocr.json").read_text(encoding="utf-8")) if (raw / "ocr.json").exists() else None
    masker = Masker(smask, ocr)
    cp, cm = CLIPProcessor.from_pretrained(vc["clip_model"]), CLIPModel.from_pretrained(vc["clip_model"]).to(device).eval()
    xp, xm = XCLIPProcessor.from_pretrained(vc["xclip_model"]), XCLIPModel.from_pretrained(vc["xclip_model"]).to(device).eval()

    def text_feats(labels, tmpl):
        with torch.no_grad():
            e = cm.get_text_features(**cp(text=[tmpl.format(l) for l in labels], return_tensors="pt", padding=True).to(device))
            return torch.nn.functional.normalize(getattr(e, "pooler_output", e).float(), dim=-1)

    T_env, T_pres = text_feats(ENVIRONMENT, PROMPTS["environment"]), text_feats(PRESENTATION, PROMPTS["presentation"])
    act_prompts = [PROMPTS["activity"].format(x) for x in ACTIVITY]
    xc = {"times": [], "raw": [], "masked": [], "mask_share": []}
    cl = {"times": [], "raw": [], "masked": [], "mask_share": []}
    fs = {"t": [], "luma_mean": [], "luma_std": []}
    buf_r, buf_m, buf_s, clips_r, clips_m = [], [], [], [], []
    nxt = vc["clip_every_s"] / 2
    n_frames = vc["xclip_frames"]

    def flush():
        with torch.no_grad():
            for key, clips in (("raw", clips_r), ("masked", clips_m)):
                x = xp(text=act_prompts, videos=clips, return_tensors="pt", padding=True).to(device)
                xc[key].append(xm(**x).logits_per_video.softmax(-1).float().cpu().numpy())
        clips_r.clear()
        clips_m.clear()

    import cv2
    for t, rgb in decode(video, vc["base_fps"], short_side=256):
        g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        fs["t"].append(t)
        fs["luma_mean"].append(float(g.mean()))
        fs["luma_std"].append(float(g.std()))
        mrgb, share = masker.apply(rgb, t)
        buf_r.append(rgb)
        buf_m.append(mrgb)
        buf_s.append(share)
        if len(buf_r) == n_frames:
            clips_r.append(list(buf_r))
            clips_m.append(list(buf_m))
            xc["times"].append(t - (n_frames - 1) / vc["base_fps"] / 2)
            xc["mask_share"].append(float(np.mean(buf_s)))
            buf_r.clear(), buf_m.clear(), buf_s.clear()
            if len(clips_r) == 8:
                flush()
        if t >= nxt:
            with torch.no_grad():
                for key, img in (("raw", rgb), ("masked", mrgb)):
                    ie = cm.get_image_features(**cp(images=[img], return_tensors="pt").to(device))
                    ie = torch.nn.functional.normalize(getattr(ie, "pooler_output", ie).float(), dim=-1)
                    s = cm.logit_scale.exp()
                    cl[key].append(np.concatenate([(s * ie @ T_env.T).softmax(-1).cpu().numpy()[0],
                                                   (s * ie @ T_pres.T).softmax(-1).cpu().numpy()[0], ie.cpu().numpy()[0]]))
            cl["times"].append(t)
            cl["mask_share"].append(share)
            nxt += vc["clip_every_s"]
    if clips_r:
        flush()
    ne, npr = len(ENVIRONMENT), len(PRESENTATION)
    np.savez_compressed(raw / "xclip_v2.npz", times=np.float32(xc["times"]), raw=np.concatenate(xc["raw"]),
                        masked=np.concatenate(xc["masked"]), mask_share=np.float32(xc["mask_share"]))
    R, M = np.stack(cl["raw"]), np.stack(cl["masked"])
    np.savez_compressed(raw / "clip_v2.npz", times=np.float32(cl["times"]), env_raw=R[:, :ne], env_masked=M[:, :ne],
                        pres_raw=R[:, ne:ne + npr], pres_masked=M[:, ne:ne + npr], emb_raw=R[:, ne + npr:],
                        emb_masked=M[:, ne + npr:], mask_share=np.float32(cl["mask_share"]))
    np.savez_compressed(raw / "frame_stats.npz", **{k: np.float32(v) for k, v in fs.items()})
    meta = {"models": {"clip": vc["clip_model"], "xclip": vc["xclip_model"]},
            "heads": {"activity": ACTIVITY, "environment": ENVIRONMENT, "presentation": PRESENTATION}, "prompts": PROMPTS,
            "sampling": {k: vc[k] for k in ("clip_every_s", "base_fps", "xclip_frames")},
            "mask": {**MASK, "frames_used": n_mask, "static_mask_share": float(smask.mean())},
            "score_note": "softmax within each head = relative score, not a calibrated probability",
            "timing_s": {"total_s": round(time.time() - t0, 1)}}
    (raw / "detect_v2_meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return {"runtime_s": meta["timing_s"]["total_s"]}


def run_vad(raw: Path, wav: Path) -> dict:
    import soundfile as sf
    import torch
    from silero_vad import get_speech_timestamps, load_silero_vad
    t0 = time.time()
    a, sr = sf.read(str(wav), dtype="float32")
    a = a.mean(1) if a.ndim > 1 else a
    ts = get_speech_timestamps(torch.from_numpy(a), load_silero_vad(), sampling_rate=sr, return_seconds=True)
    (raw / "vad.json").write_text(json.dumps({"model": "silero-vad", "sampling_rate": sr, "duration_s": r(len(a) / sr),
        "speech": [{"start": r(x["start"]), "end": r(x["end"])} for x in ts]}), encoding="utf-8")
    return {"runtime_s": round(time.time() - t0, 1)}


def transcript_words(raw: Path, transcript: dict) -> dict:
    """asr.json-compatible words/sentences from the selected transcript, so diarisation can run without Whisper."""
    words = [{"w": " " + w["w"], "start": w["start"], "end": max(w["end"], w["start"] + 0.05)} for w in transcript["words"]]
    sents = [{"start": s["start"], "end": max(s["end"], s["start"] + 0.3), "text": s["text"]} for s in transcript["segments"]]
    (raw / "asr.json").write_text(json.dumps({"segmenter": "asr", "model": f"transcript: {(transcript.get('primary') or {}).get('file')}",
        "params": {"source": "ingested transcript (no ASR run)"}, "runtime_s": 0.0, "words": words, "sentences": sents,
        "stats": {"n_words": len(words)}}, ensure_ascii=False), encoding="utf-8")
    return {"n_words": len(words)}
