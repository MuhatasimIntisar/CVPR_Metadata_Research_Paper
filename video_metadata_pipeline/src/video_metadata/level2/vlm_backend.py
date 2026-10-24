"""VLM backend (default Qwen/Qwen3.5-4B via transformers) with deterministic decoding and an on-disk cache.

Images are passed as PIL frames interleaved with text labels that state each frame's exact timestamp. Thinking mode
is disabled (enable_thinking=False) and decoding is greedy, so identical inputs give identical outputs; outputs are
cached by a hash of (model, prompt version, prompt text, frame timestamps, video hash, generation settings).
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path

from ..utils.io import read_json, write_json

THINK = re.compile(r"<think>.*?</think>", re.S)


class QwenBackend:
    def __init__(self, cfg: dict, device: str):
        import torch
        from transformers import AutoProcessor
        try:
            from transformers import AutoModelForMultimodalLM as Auto
        except ImportError:  # older transformers
            from transformers import AutoModelForImageTextToText as Auto
        c = cfg["level2"]
        self.model_id = c["model_id"]
        dtype = getattr(torch, c.get("dtype", "bfloat16"))
        t0 = time.time()
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = Auto.from_pretrained(self.model_id, dtype=dtype, device_map=device if device != "cpu" else None).eval()
        self.load_s = round(time.time() - t0, 1)
        self.torch = torch

    def generate(self, content: list[dict], max_new_tokens: int) -> str:
        """content: [{"type": "text", "text": ...} | {"type": "image", "image": PIL.Image}, ...]"""
        content = [{"type": "image", "image": c["image"]} if c["type"] == "image" else {"type": "text", "text": c["text"]}
                   for c in content]
        msgs = [{"role": "user", "content": content}]
        try:
            inputs = self.processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True,
                                                        return_tensors="pt", enable_thinking=False)
        except Exception:  # processors that do not load PIL images from the template
            text = self.processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False)
            imgs = [c["image"] for c in content if c["type"] == "image"]
            inputs = self.processor(text=[text], images=imgs or None, return_tensors="pt")
        inputs = inputs.to(self.model.device)
        with self.torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        txt = self.processor.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
        return THINK.sub("", txt).strip()

    def gpu_memory_gb(self):
        if self.torch.cuda.is_available():
            return round(self.torch.cuda.max_memory_allocated() / 2**30, 2)
        return None


class CachedVLM:
    """Wraps a backend; loads it lazily so fully cached reruns never touch the GPU."""

    def __init__(self, cfg: dict, device: str, cache_dir: Path, video_hash: str, backend=None):
        self.cfg, self.device, self.cache_dir, self.video_hash = cfg, device, Path(cache_dir), video_hash
        self._backend = backend
        self.stats = {"calls": 0, "cache_hits": 0, "generate_s": 0.0, "frames": 0}

    @property
    def backend(self):
        if self._backend is None:
            self._backend = QwenBackend(self.cfg, self.device)
            self.stats["model_load_s"] = self._backend.load_s
        return self._backend

    def key(self, content, max_new_tokens):
        sig = {"model": self.cfg["level2"]["model_id"], "prompt_version": self.cfg["level2"]["prompt_version"],
               "video": self.video_hash, "max_new_tokens": max_new_tokens,
               "content": [c["text"] if c["type"] == "text" else f"img@{c.get('t')}" for c in content]}
        return hashlib.sha256(json.dumps(sig, sort_keys=True).encode()).hexdigest()[:24]

    def __call__(self, content, max_new_tokens, tag=""):
        k = self.key(content, max_new_tokens)
        p = self.cache_dir / f"{tag}_{k}.json"
        hit = read_json(p)
        if hit is not None:
            self.stats["cache_hits"] += 1
            return hit["output"], True
        t0 = time.time()
        out = self.backend.generate(content, max_new_tokens)
        dt = time.time() - t0
        self.stats["calls"] += 1
        self.stats["generate_s"] += dt
        self.stats["frames"] += sum(c["type"] == "image" for c in content)
        write_json(p, {"key": k, "tag": tag, "model": self.cfg["level2"]["model_id"], "seconds": round(dt, 2),
                       "frame_times": [c.get("t") for c in content if c["type"] == "image"], "output": out})
        return out, False
