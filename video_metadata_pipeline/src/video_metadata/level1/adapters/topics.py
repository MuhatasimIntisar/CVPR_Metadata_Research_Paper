"""Transcript topic embeddings (Sentence-BERT all-MiniLM-L6-v2 via transformers: mean pooling + L2 norm).

Transcript words are grouped into chunk_s windows every hop_s; each chunk with >= 1 word is embedded once.
The topic-change curve itself is computed later on the 2 s observation grid (level1/observations.py) by comparing
the word-weighted mean embedding of the context_s before and after each step (Experiment C: 60 s context was the
best transcript setting for speech-led footage). Output: raw/transcript_topics.npz + transcript_topics.json.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def chunks(words, chunk_s, hop_s, end):
    out, t = [], 0.0
    while t < end:
        ws = [w["w"] for w in words if t <= (w["start"] + w["end"]) / 2 < t + chunk_s]
        if ws:
            out.append((t, t + chunk_s, " ".join(ws)))
        t += hop_s
    return out


def run(raw: Path, transcript: dict, cfg: dict, device: str) -> dict:
    import torch
    from transformers import AutoModel, AutoTokenizer
    tc = cfg["level1"]["topics"]
    t0 = time.time()
    words = transcript.get("words", [])
    end = max((w["end"] for w in words), default=0.0)
    ch = chunks(words, tc["chunk_s"], tc["hop_s"], end)
    lang = ((transcript.get("primary") or {}).get("language") or "en").lower()
    model_id = tc.get("model", MODEL) if lang.startswith("en") else tc.get("model_non_english", tc.get("model", MODEL))
    meta = {"model": model_id, "language": lang, "chunk_s": tc["chunk_s"], "hop_s": tc["hop_s"], "n_chunks": len(ch),
            "transcript": (transcript.get("primary") or {}).get("file")}
    if not ch:
        np.savez_compressed(raw / "transcript_topics.npz", times=np.zeros(0, np.float32), n_words=np.zeros(0, np.int32),
                            emb=np.zeros((0, 384), np.float32))
        (raw / "transcript_topics.json").write_text(json.dumps({**meta, "status": "no transcript"}), encoding="utf-8")
        return {"runtime_s": 0.0, "n_chunks": 0}
    tok = AutoTokenizer.from_pretrained(meta["model"])
    model = AutoModel.from_pretrained(meta["model"]).to(device).eval()
    embs = []
    with torch.no_grad():
        for i in range(0, len(ch), 64):
            x = tok([c[2] for c in ch[i:i + 64]], padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
            h = model(**x).last_hidden_state
            m = x["attention_mask"].unsqueeze(-1).float()
            e = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)
            embs.append(torch.nn.functional.normalize(e, dim=-1).cpu().numpy())
    np.savez_compressed(raw / "transcript_topics.npz", times=np.float32([(a + b) / 2 for a, b, _ in ch]),
                        n_words=np.int32([len(c[2].split()) for c in ch]), emb=np.concatenate(embs).astype(np.float32))
    meta.update(status="ok", runtime_s=round(time.time() - t0, 1))
    (raw / "transcript_topics.json").write_text(json.dumps(meta), encoding="utf-8")
    return {"runtime_s": meta["runtime_s"], "n_chunks": len(ch)}
