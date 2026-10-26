"""VLM backends for Qwen/Qwen3.5-4B with greedy decoding, batching and an on-disk cache.

level2.backend:
  vllm          batched offline inference (vLLM >= 0.17); all requests of a stage go to the engine together and are
                scheduled with continuous batching. The engine is created once per process and reused for every video.
  transformers  one request at a time (fallback; slow without the fast linear-attention kernels).
Images are PIL frames interleaved with text labels that state each frame's exact timestamp; thinking mode is off.
Outputs are cached by a hash of (model, prompt version, prompt text, frame timestamps, video hash, max tokens), so a
re-run never regenerates a cached answer, whichever backend produced it (the cache file records the backend). Greedy
decoding is deterministic per request; vLLM does not guarantee bit-identical text across different batch
compositions, so the cache - not re-generation - is what makes a run reproducible.
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
    name = "transformers"

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

    def generate_batch(self, contents: list[list[dict]], max_new_tokens: list[int]) -> list[str]:
        return [self.generate(c, n) for c, n in zip(contents, max_new_tokens)]

    def gpu_memory_gb(self):
        if self.torch.cuda.is_available():
            return round(self.torch.cuda.max_memory_allocated() / 2**30, 2)
        return None


class VLLMBackend:
    """vLLM offline engine. The prompt is rendered with the model's own chat template (one image placeholder per
    image, in content order) and the PIL frames are passed in the same order as multi_modal_data."""
    name = "vllm"

    def __init__(self, cfg: dict, device: str):
        from vllm import LLM, SamplingParams
        c = cfg["level2"]
        v = dict(c.get("vllm") or {})
        self.model_id = c["model_id"]
        self.max_images = int(v.pop("max_images_per_prompt", 24))
        t0 = time.time()
        base = dict(model=self.model_id, dtype=c.get("dtype", "bfloat16"), seed=0, limit_mm_per_prompt={"image": self.max_images})
        # fall-backs for known engine-start problems with hybrid (linear-attention) models: prefix caching, CUDA graphs
        tries = [v, {**v, "enable_prefix_caching": False}, {**v, "enable_prefix_caching": False, "enforce_eager": True}]
        for k, kw in enumerate(tries):
            try:
                self.llm = LLM(**base, **kw)
                self.engine_args = kw
                break
            except Exception as e:  # noqa: BLE001
                if k == len(tries) - 1:
                    raise
                print(f"[vllm] engine start failed with {kw}: {type(e).__name__}: {str(e)[:300]} - retrying with {tries[k + 1]}",
                      flush=True)
        self.tok = self.llm.get_tokenizer()
        self.SP = SamplingParams
        self.load_s = round(time.time() - t0, 1)

    def prompt(self, content):
        msgs = [{"role": "user", "content": [{"type": "image"} if c["type"] == "image" else {"type": "text", "text": c["text"]}
                                             for c in content]}]
        text = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        imgs = [c["image"] for c in content if c["type"] == "image"]
        if text.count("<|image_pad|>") != len(imgs):
            raise RuntimeError(f"chat template rendered {text.count('<|image_pad|>')} image placeholders for {len(imgs)} images")
        if len(imgs) > self.max_images:
            raise RuntimeError(f"{len(imgs)} images > level2.vllm.max_images_per_prompt {self.max_images}")
        return {"prompt": text, "multi_modal_data": {"image": imgs}} if imgs else {"prompt": text}

    def generate_batch(self, contents, max_new_tokens):
        reqs = [self.prompt(c) for c in contents]
        sps = [self.SP(temperature=0.0, max_tokens=int(n), seed=0) for n in max_new_tokens]
        outs = self.llm.generate(reqs, sps, use_tqdm=False)
        return [THINK.sub("", o.outputs[0].text).strip() for o in outs]

    def generate(self, content, max_new_tokens):
        return self.generate_batch([content], [max_new_tokens])[0]

    def gpu_memory_gb(self):
        try:
            import torch
            return round(torch.cuda.max_memory_allocated() / 2**30, 2) if torch.cuda.is_available() else None
        except Exception:  # noqa: BLE001
            return None


_BACKENDS: dict = {}


def get_backend(cfg: dict, device: str):
    """One engine per process (loading vLLM takes minutes; it is reused by every video of the job)."""
    c = cfg["level2"]
    name = c.get("backend", "auto")
    if name == "auto":
        try:
            import vllm  # noqa: F401
            name = "vllm" if device != "cpu" else "transformers"
        except ImportError:
            name = "transformers"
    key = (name, c["model_id"])
    if key not in _BACKENDS:
        _BACKENDS[key] = VLLMBackend(cfg, device) if name == "vllm" else QwenBackend(cfg, device)
    return _BACKENDS[key]


class CachedVLM:
    """Wraps a backend; loads it lazily so fully cached reruns never touch the GPU."""

    def __init__(self, cfg: dict, device: str, cache_dir: Path, video_hash: str, backend=None):
        self.cfg, self.device, self.cache_dir, self.video_hash = cfg, device, Path(cache_dir), video_hash
        self._backend = backend
        self.stats = {"calls": 0, "cache_hits": 0, "generate_s": 0.0, "frames": 0, "batches": 0}

    @property
    def backend(self):
        if self._backend is None:
            self._backend = get_backend(self.cfg, self.device)
            self.stats["model_load_s"] = self._backend.load_s
        return self._backend

    def key(self, content, max_new_tokens, tag=""):
        c = self.cfg["level2"]
        pv = c["prompt_version"] if tag.startswith("merge") else c.get("metadata_prompt_version", c["prompt_version"])
        sig = {"model": c["model_id"], "prompt_version": pv,
               "video": self.video_hash, "max_new_tokens": max_new_tokens,
               "content": [c["text"] if c["type"] == "text" else f"img@{c.get('t')}" for c in content]}
        return hashlib.sha256(json.dumps(sig, sort_keys=True).encode()).hexdigest()[:24]

    def __call__(self, content, max_new_tokens, tag=""):
        return self.batch([(content, max_new_tokens, tag)])[0]

    def batch(self, requests):
        """requests: [(content, max_new_tokens, tag)] -> [(output, cached)] in the same order. Cache misses are sent
        to the backend together (in chunks of level2.batch_size)."""
        res, miss = [None] * len(requests), []
        for i, (content, n, tag) in enumerate(requests):
            k = self.key(content, n, tag)
            p = self.cache_dir / f"{tag}_{k}.json"
            hit = read_json(p)
            if hit is not None:
                self.stats["cache_hits"] += 1
                res[i] = (hit["output"], True)
            else:
                miss.append((i, k, p))
        bs = int(self.cfg["level2"].get("batch_size", 64))
        for j in range(0, len(miss), bs):
            chunk = miss[j:j + bs]
            t0 = time.time()
            cs, ns = [requests[i][0] for i, _, _ in chunk], [requests[i][1] for i, _, _ in chunk]
            gb = getattr(self.backend, "generate_batch", None)
            outs = gb(cs, ns) if gb else [self.backend.generate(x, n) for x, n in zip(cs, ns)]
            dt = time.time() - t0
            self.stats["calls"] += len(chunk)
            self.stats["batches"] += 1
            self.stats["generate_s"] += dt
            for (i, k, p), out in zip(chunk, outs):
                content, _, tag = requests[i]
                self.stats["frames"] += sum(c["type"] == "image" for c in content)
                write_json(p, {"key": k, "tag": tag, "model": self.cfg["level2"]["model_id"],
                               "backend": getattr(self.backend, "name", "custom"),
                               "seconds": round(dt / len(chunk), 2), "batch_size": len(chunk),
                               "frame_times": [c.get("t") for c in content if c["type"] == "image"], "output": out})
                res[i] = (out, False)
        return res
