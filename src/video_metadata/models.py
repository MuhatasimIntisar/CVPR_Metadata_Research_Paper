"""Pre-download every pretrained model into paths.model_cache (run on a machine/login node with internet)."""
from __future__ import annotations

import urllib.request
from pathlib import Path

HF_MODELS = {"level1": ["openai/clip-vit-base-patch32", "microsoft/xclip-base-patch32", "speechbrain/spkrec-ecapa-voxceleb",
                        "MIT/ast-finetuned-audioset-10-10-0.4593",
                        "sentence-transformers/all-MiniLM-L6-v2", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"],
             "vlm": ["Qwen/Qwen3.5-4B"]}
YOLO_URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/{}"


def download(cfg: dict, include_vlm: bool = False, ocr_langs=("en", "es")) -> list[str]:
    cache = cfg["paths"].get("model_cache")
    if not cache:
        raise SystemExit("set paths.model_cache in your config first")
    cache = Path(cache)
    import os
    os.environ["HF_HUB_OFFLINE"] = "0"  # downloading is the point of this command, whatever runtime.offline says
    os.environ["TRANSFORMERS_OFFLINE"] = "0"
    done = []
    from huggingface_hub import snapshot_download
    vlm = [(cfg.get("level2") or {}).get("model_id") or HF_MODELS["vlm"][0]]  # the configured Level-2 model (VMP_LEVEL2__MODEL_ID)
    for repo in HF_MODELS["level1"] + (vlm if include_vlm else []):
        snapshot_download(repo)  # HF_HOME is set from paths.model_cache by config.apply_runtime
        done.append(repo)
    ydir = cache / "ultralytics"
    ydir.mkdir(parents=True, exist_ok=True)
    for key in ("detect_model", "pose_model"):
        name = Path(cfg["level1"]["yolo"][key]).name
        dst = ydir / name
        if not dst.exists():
            urllib.request.urlretrieve(YOLO_URL.format(name), dst)
        done.append(str(dst))
    import easyocr
    easyocr.Reader(list(ocr_langs), gpu=False, verbose=False)  # EASYOCR_MODULE_PATH is set from model_cache
    done.append(f"easyocr {ocr_langs}")
    import silero_vad
    silero_vad.load_silero_vad()  # weights ship inside the pip package
    done.append("silero-vad (bundled)")
    return done
