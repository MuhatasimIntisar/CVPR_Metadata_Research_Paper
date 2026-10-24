"""Configuration: packaged defaults <- optional YAML overrides <- VMP_* environment variables.

Environment overrides use double underscores for nesting, e.g. VMP_PATHS__OUTPUTS_ROOT=/scratch/out,
VMP_RUNTIME__DEVICE=cpu. Values are parsed as YAML scalars.
"""
from __future__ import annotations

import copy
import os
from pathlib import Path

import yaml

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PACKAGE_ROOT / "configs" / "default.yaml"


def deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def load_config(*paths: str | Path | None, env: dict | None = None) -> dict:
    cfg = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8")) if DEFAULT_CONFIG.exists() else {}
    for p in paths:
        if p:
            cfg = deep_merge(cfg, yaml.safe_load(Path(p).read_text(encoding="utf-8")) or {})
    for k, v in (env if env is not None else os.environ).items():
        if not k.startswith("VMP_"):
            continue
        keys = k[4:].lower().split("__")
        node = cfg
        for key in keys[:-1]:
            node = node.setdefault(key, {})
        node[keys[-1]] = yaml.safe_load(v)
    apply_runtime(cfg)
    return cfg


def apply_runtime(cfg: dict) -> None:
    """Model cache / offline settings are applied through standard library environment variables."""
    cache = (cfg.get("paths") or {}).get("model_cache")
    if cache:
        os.environ.setdefault("HF_HOME", str(Path(cache) / "huggingface"))
        os.environ.setdefault("TORCH_HOME", str(Path(cache) / "torch"))
        os.environ.setdefault("YOLO_CONFIG_DIR", str(Path(cache) / "ultralytics"))
        os.environ.setdefault("EASYOCR_MODULE_PATH", str(Path(cache) / "easyocr"))
    if (cfg.get("runtime") or {}).get("offline"):
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


def video_dir(cfg: dict, video_id: str) -> Path:
    return Path(cfg["paths"]["outputs_root"]) / video_id
