"""Level-1 detectors (run once, GPU machine). Writes raw, timestamped outputs only - no aggregation.

    python -m level1.detect --video-id detvde-Chocolate_Ballerina_Founder_Chane_5b4f44

One video decode at 4 fps feeds:
  * YOLO detection + ByteTrack (2 fps)            -> yolo_track.json
  * YOLO pose (2 fps)                             -> yolo_pose.json
  * X-CLIP action, 8-frame clips over 2 s          -> xclip_actions.npz (+ labels in detect_meta.json)
  * CLIP environment, 1 frame every 5 s            -> clip_env.npz (embeddings + label probabilities)
Plus Silero VAD on the continuous audio            -> vad.json
Reused from data/segments/<id>/ by the aggregator (not rerun): shots, speakers, audio_events, ocr,
camera_motion, silence.
Every output is cached; rerun with --force to recompute, --only to run a subset.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np

YOLO_DET, YOLO_POSE = "yolo11s.pt", "yolo11n-pose.pt"
XCLIP, CLIP = "microsoft/xclip-base-patch32", "openai/clip-vit-base-patch32"
BASE_FPS, YOLO_FPS, CLIP_EVERY_S, XCLIP_FRAMES = 4.0, 2.0, 5.0, 8   # X-CLIP clip = 8 frames at 4 fps = 2 s

ENVIRONMENTS = ["a television studio set", "a news desk", "an office", "a meeting or council room",
                "a classroom or lecture hall", "a stage or auditorium", "a kitchen", "a home interior",
                "a shop or market", "a restaurant or cafe", "a city street", "a park or garden",
                "a beach or waterfront", "countryside or forest", "a sports field or arena",
                "a gym or dance studio", "a factory or workshop", "inside a vehicle",
                "an on-screen graphic or title card", "a black or blank screen"]
ACTIVITIES = ["people talking while seated", "a person talking to the camera", "people talking while standing",
              "people dancing", "people performing music", "people walking", "a parade or procession",
              "people cooking or preparing food", "people eating or drinking", "people playing sports",
              "people exercising", "a crowd watching or applauding", "a person presenting with slides",
              "people working with tools or machines", "people shopping", "vehicles moving on a road",
              "scenery with no people", "on-screen text or graphics"]


def paths(root: Path, vid: str):
    video = next((root / "data" / "pilot").glob(f"*/{vid}/video.mp4"), None)
    return video, root / "data" / "segments" / vid, root / "data" / "level1" / vid


def device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def decode(video: Path, fps: float, short_side: int = 480):
    """Sequential decode; yields (t, rgb) for sample times (k + 0.5) / fps."""
    import cv2
    cap = cv2.VideoCapture(str(video))
    vfps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    k, idx = 0, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = idx / vfps
        if t + 0.5 / vfps >= (k + 0.5) / fps:
            h, w = frame.shape[:2]
            s = short_side / min(h, w)
            if s < 1:
                frame = cv2.resize(frame, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
            yield (k + 0.5) / fps, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            k += 1
        idx += 1
    cap.release()


def r(x, n=3):
    return round(float(x), n)


def dump(path: Path, obj):
    path.write_text(json.dumps(obj, separators=(",", ":"), ensure_ascii=False), encoding="utf-8")


def run_visual(video, out, todo, dev):
    import torch
    timing = {}
    yolo = pose = xm = xp = cm = cp = None
    if "yolo" in todo:
        from ultralytics import YOLO
        yolo, pose = YOLO(YOLO_DET), YOLO(YOLO_POSE)
        track_frames, pose_frames = [], []
        timing["yolo_s"] = 0.0
    if "xclip" in todo:
        from transformers import XCLIPModel, XCLIPProcessor
        xp = XCLIPProcessor.from_pretrained(XCLIP)
        xm = XCLIPModel.from_pretrained(XCLIP).to(dev).eval()
        clips, clip_buf, clip_t, act_probs = [], [], [], []
        timing["xclip_s"] = 0.0
    if "clip" in todo:
        from transformers import CLIPModel, CLIPProcessor
        cp = CLIPProcessor.from_pretrained(CLIP)
        cm = CLIPModel.from_pretrained(CLIP).to(dev).eval()
        env_frames, env_t = [], []
        next_env = CLIP_EVERY_S / 2
        timing["clip_s"] = 0.0
    step = int(round(BASE_FPS / YOLO_FPS))

    def flush_xclip():
        t0 = time.time()
        with torch.no_grad():
            x = xp(text=[f"a video of {a}" for a in ACTIVITIES], videos=clips, return_tensors="pt", padding=True).to(dev)
            p = xm(**x).logits_per_video.softmax(-1).float().cpu().numpy()
        act_probs.append(p)
        clips.clear()
        timing["xclip_s"] += time.time() - t0

    n = 0
    t_dec = time.time()
    for k, (t, rgb) in enumerate(decode(video, BASE_FPS)):
        n += 1
        if yolo is not None and k % step == 0:
            t0 = time.time()
            bgr = np.ascontiguousarray(rgb[:, :, ::-1])
            h, w = bgr.shape[:2]
            res = yolo.track(bgr, persist=True, tracker="bytetrack.yaml", conf=0.25, verbose=False, device=dev)[0]
            dets = []
            for b in res.boxes:
                x1, y1, x2, y2 = b.xyxyn[0].tolist()
                dets.append([res.names[int(b.cls)], r(b.conf), int(b.id) if b.id is not None else -1,
                             r(x1), r(y1), r(x2), r(y2)])
            track_frames.append({"t": r(t), "dets": dets})
            pr = pose.predict(bgr, conf=0.25, verbose=False, device=dev)[0]
            persons = []
            if pr.keypoints is not None:
                kx = pr.keypoints.xyn.cpu().numpy()
                kc = pr.keypoints.conf.cpu().numpy() if pr.keypoints.conf is not None else np.ones(kx.shape[:2])
                for j, b in enumerate(pr.boxes):
                    persons.append({"conf": r(b.conf), "box": [r(v) for v in b.xyxyn[0].tolist()],
                                    "kpts": [[r(a), r(c), r(d)] for (a, c), d in zip(kx[j], kc[j])]})
            pose_frames.append({"t": r(t), "persons": persons})
            timing["yolo_s"] += time.time() - t0
        if xm is not None:
            clip_buf.append(rgb)
            if len(clip_buf) == XCLIP_FRAMES:
                clips.append(list(clip_buf))
                clip_t.append(r(t - (XCLIP_FRAMES - 1) / BASE_FPS / 2))  # clip centre
                clip_buf.clear()
                if len(clips) == 8:
                    flush_xclip()
        if cm is not None and t >= next_env:
            env_frames.append(rgb)
            env_t.append(r(t))
            next_env += CLIP_EVERY_S
        if n % 400 == 0:
            print(f"\r  decoded {t:7.1f}s", end="", flush=True)
    print()
    timing["decode_and_all_s"] = round(time.time() - t_dec, 1)
    meta = {}
    if yolo is not None:
        dump(out / "yolo_track.json", {"model": YOLO_DET, "tracker": "bytetrack.yaml", "fps": YOLO_FPS,
                                       "conf": 0.25, "format": "[class, conf, track_id, x1, y1, x2, y2] (normalised)",
                                       "frames": track_frames})
        dump(out / "yolo_pose.json", {"model": YOLO_POSE, "fps": YOLO_FPS, "conf": 0.25,
                                      "format": "kpts = COCO-17 [x, y, conf] (normalised)", "frames": pose_frames})
        meta["yolo_frames"] = len(track_frames)
    if xm is not None:
        if clips:
            flush_xclip()
        np.savez_compressed(out / "xclip_actions.npz", times=np.array(clip_t, np.float32),
                            probs=np.concatenate(act_probs).astype(np.float32))
        meta["xclip_clips"] = len(clip_t)
    if cm is not None:
        t0 = time.time()
        with torch.no_grad():
            tx = cp(text=[f"a video frame of {e}" for e in ENVIRONMENTS], return_tensors="pt", padding=True).to(dev)
            te = cm.get_text_features(**tx)
            te = torch.nn.functional.normalize(getattr(te, "pooler_output", te).float(), dim=-1)
            embs = []
            for i in range(0, len(env_frames), 32):
                ix = cp(images=env_frames[i:i + 32], return_tensors="pt").to(dev)
                ie = cm.get_image_features(**ix)
                embs.append(torch.nn.functional.normalize(getattr(ie, "pooler_output", ie).float(), dim=-1))
            ie = torch.cat(embs)
            probs = (cm.logit_scale.exp() * ie @ te.T).softmax(-1)
        np.savez_compressed(out / "clip_env.npz", times=np.array(env_t, np.float32),
                            emb=ie.cpu().numpy().astype(np.float32), probs=probs.cpu().numpy().astype(np.float32))
        timing["clip_s"] += time.time() - t0
        meta["clip_frames"] = len(env_t)
    return timing, meta


def run_vad(video, seg_dir, out):
    import soundfile as sf
    import torch
    from silero_vad import get_speech_timestamps, load_silero_vad
    t0 = time.time()
    wav = seg_dir / "_cache" / "audio_16k.wav"
    if not wav.exists():
        wav = out / "audio_16k.wav"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(video), "-vn", "-ac", "1", "-ar", "16000",
                        str(wav)], check=True)
    a, sr = sf.read(str(wav), dtype="float32")
    a = a.mean(1) if a.ndim > 1 else a
    ts = get_speech_timestamps(torch.from_numpy(a), load_silero_vad(), sampling_rate=sr, return_seconds=True)
    dump(out / "vad.json", {"model": "silero-vad", "sampling_rate": sr, "duration_s": r(len(a) / sr),
                            "speech": [{"start": r(x["start"]), "end": r(x["end"])} for x in ts]})
    return round(time.time() - t0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video-id", required=True)
    ap.add_argument("--root", default=".")
    ap.add_argument("--only", default="yolo,xclip,clip,vad")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    video, seg_dir, out = paths(Path(a.root), a.video_id)
    if video is None:
        raise SystemExit(f"video.mp4 for {a.video_id} not found under data/pilot/*/")
    out.mkdir(parents=True, exist_ok=True)
    files = {"yolo": "yolo_pose.json", "xclip": "xclip_actions.npz", "clip": "clip_env.npz", "vad": "vad.json"}
    todo = [k for k in a.only.split(",") if a.force or not (out / files[k]).exists()]
    print("running:", todo or "nothing (all cached; use --force)")
    meta_p = out / "detect_meta.json"
    meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}
    dev = device()
    if any(k in todo for k in ("yolo", "xclip", "clip")):
        timing, m = run_visual(video, out, todo, dev)
        meta.setdefault("timing_s", {}).update({k: round(v, 1) for k, v in timing.items()})
        meta.update(m)
    if "vad" in todo:
        meta.setdefault("timing_s", {})["vad_s"] = run_vad(video, seg_dir, out)
    meta.update({"video": str(video), "device": dev, "models": {"yolo": YOLO_DET, "pose": YOLO_POSE, "xclip": XCLIP,
                 "clip": CLIP, "vad": "silero-vad"}, "environments": ENVIRONMENTS, "activities": ACTIVITIES,
                 "sampling": {"base_fps": BASE_FPS, "yolo_fps": YOLO_FPS, "clip_every_s": CLIP_EVERY_S,
                              "xclip_frames": XCLIP_FRAMES, "xclip_clip_s": XCLIP_FRAMES / BASE_FPS}})
    meta_p.write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print("done:", out, meta.get("timing_s"))


if __name__ == "__main__":
    main()
