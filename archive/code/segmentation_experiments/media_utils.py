"""Media I/O (ffprobe/ffmpeg/OpenCV) and the embedding cache."""
from __future__ import annotations

import gc
import json
import subprocess
from pathlib import Path

import numpy as np


def ffprobe(path: Path) -> dict:
    p = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                        "stream=codec_type,codec_name,width,height,r_frame_rate,sample_rate,channels,duration,nb_frames"
                        ":format=duration", "-of", "json", str(path)], capture_output=True, text=True, check=True)
    d = json.loads(p.stdout)
    out = {"duration": float(d.get("format", {}).get("duration", 0) or 0), "video": None, "audio": None}
    for s in d.get("streams", []):
        if s.get("codec_type") == "video" and out["video"] is None:
            num, den = (s.get("r_frame_rate") or "0/1").split("/")
            out["video"] = {"codec": s.get("codec_name"), "width": s.get("width"), "height": s.get("height"),
                            "fps": float(num) / float(den) if float(den) else 0.0,
                            "duration": float(s.get("duration") or out["duration"]),
                            "frames": int(s["nb_frames"]) if str(s.get("nb_frames", "")).isdigit() else None}
        if s.get("codec_type") == "audio" and out["audio"] is None:
            out["audio"] = {"codec": s.get("codec_name"), "sample_rate": int(s.get("sample_rate") or 0),
                            "channels": s.get("channels"), "duration": float(s.get("duration") or out["duration"])}
    return out


def extract_audio(src: Path, dst: Path, sr: int) -> Path:
    """Mono PCM WAV at `sr` (cached)."""
    if not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-vn", "-ac", "1", "-ar", str(sr),
                        "-c:a", "pcm_s16le", str(dst)], check=True)
    return dst


def load_audio(wav: Path) -> tuple[np.ndarray, int]:
    import soundfile as sf
    a, sr = sf.read(str(wav), dtype="float32")
    if a.ndim > 1:
        a = a.mean(axis=1)
    return a, sr


def sample_frames(video: Path, fps: float):
    """Yield (t, RGB frame) at `fps`, taking the decoded frame nearest each sample time t = (k + 0.5) / fps."""
    import cv2
    cap = cv2.VideoCapture(str(video))
    vfps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    k, idx = 0, 0
    next_t = 0.5 / fps
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = idx / vfps
        if t + 0.5 / vfps >= next_t:
            yield next_t, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            k += 1
            next_t = (k + 0.5) / fps
        idx += 1
    cap.release()


def get_device(pref: str = "auto") -> str:
    if pref != "auto":
        return pref
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def free_gpu():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def model_revision(model) -> str | None:
    return getattr(getattr(model, "config", None), "_commit_hash", None)


# ---------------------------------------------------------------- cache
def cache_paths(cache_dir: Path, video_id: str, name: str) -> tuple[Path, Path]:
    return cache_dir / f"{video_id}_{name}.npz", cache_dir / f"{video_id}_{name}.json"


def save_cache(cache_dir: Path, video_id: str, name: str, meta: dict, **arrays):
    npz, js = cache_paths(cache_dir, video_id, name)
    np.savez_compressed(npz, **arrays)
    js.write_text(json.dumps(meta, indent=1, ensure_ascii=False), encoding="utf-8")


def load_cache(cache_dir: Path, video_id: str, name: str):
    npz, js = cache_paths(cache_dir, video_id, name)
    if not (npz.exists() and js.exists()):
        return None, None
    with np.load(npz, allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    return arrays, json.loads(js.read_text(encoding="utf-8"))
