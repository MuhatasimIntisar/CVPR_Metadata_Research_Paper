"""M4 input: Sentence-BERT embeddings of overlapping transcript chunks (cached once).

Embedding chunk length (text_chunk_s) and hop (text_chunk_hop_s) are independent of the context
window used to compare "before" and "after" (text_context_s, applied in change_scores), of the
temporal grid (grid_s) and of the minimum segment duration. Larger contexts are built by
aggregating neighbouring chunk embeddings (word-count weighted mean, then L2 normalised), so
changing the context never re-runs the model.

All speakers are merged in time order (word midpoint). Chunks with no words are dropped: missing
speech is NOT treated as evidence of no change. Transcript sources: AMI manual transcript (oracle)
or, for MovieChat, an optional Whisper transcript (cache/<id>_asr_words.json via --steps transcribe).
"""
from __future__ import annotations

import json
import time

import numpy as np

from .media_utils import free_gpu, load_cache, save_cache

NAME = "text_chunks"


def load_words(item, dirs):
    """Return (words [{start, end, text, speaker, punct}], source) or (None, reason)."""
    if item.transcript is not None and item.transcript.exists():
        raw = json.loads(item.transcript.read_text(encoding="utf-8"))
        src = "oracle_manual_transcript"
    else:
        asr = dirs["cache"] / f"{item.video_id}_asr_words.json"
        if not asr.exists():
            return None, "no transcript (run --steps transcribe for an optional Whisper transcript)"
        raw = json.loads(asr.read_text(encoding="utf-8"))
        src = "whisper_asr"
    words = []
    for w in raw:
        text = (w.get("text") if "text" in w else w.get("w", "")) or ""
        words.append({"start": float(w["start"]), "end": float(w["end"]), "text": text.strip(),
                      "speaker": w.get("speaker"), "punct": bool(w.get("punctuation", False))})
    words.sort(key=lambda w: (w["start"], w["end"]))
    return words, src


def join(ws) -> str:
    out = ""
    for w in ws:
        if not w["text"]:
            continue
        out += w["text"] if (w["punct"] or not out) else " " + w["text"]
    return out


def make_chunks(words, duration: float, chunk_s: float, hop_s: float):
    """Overlapping chunks [s, s + chunk_s) every hop_s; a word belongs to every chunk containing its midpoint."""
    mids = np.array([(w["start"] + w["end"]) / 2 for w in words])
    content = np.array([not w["punct"] for w in words])
    chunks = []
    for s in np.arange(0.0, max(duration - chunk_s, 0.0) + hop_s, hop_s):
        idx = np.where((mids >= s) & (mids < s + chunk_s))[0]
        n = int(content[idx].sum()) if len(idx) else 0
        if n == 0:
            continue
        chunks.append({"start": float(s), "end": float(min(s + chunk_s, duration)), "n_words": n,
                       "text": join([words[i] for i in idx]),
                       "speakers": sorted({words[i]["speaker"] for i in idx if words[i]["speaker"]})})
    return chunks


def params(cfg):
    return {"model": cfg.text_model, "chunk_s": cfg.text_chunk_s, "chunk_hop_s": cfg.text_chunk_hop_s,
            "speaker_merge": "chronological by word midpoint", "aggregation": "done later (context window)"}


def encode(item, cfg, dirs, device, duration: float, force=False):
    words, src = load_words(item, dirs)
    if words is None:
        return None, {"status": "unavailable", "reason": src}
    arrays, meta = load_cache(dirs["cache"], item.video_id, NAME)
    p = dict(params(cfg), transcript_source=src, duration=round(duration, 3))
    if arrays is not None and not force and meta.get("params") == p:
        return arrays, meta
    from sentence_transformers import SentenceTransformer
    t0 = time.time()
    chunks = make_chunks(words, duration, cfg.text_chunk_s, cfg.text_chunk_hop_s)
    model = SentenceTransformer(cfg.text_model, device=device)
    emb = model.encode([c["text"] for c in chunks], batch_size=64, normalize_embeddings=True,
                       show_progress_bar=False) if chunks else np.zeros((0, 384), np.float32)
    del model
    free_gpu()
    meta = {"status": "ok", "modality": "text", "params": p, "n": len(chunks),
            "n_words": sum(1 for w in words if not w["punct"]),
            "speakers": sorted({w["speaker"] for w in words if w["speaker"]}),
            "transcript_span_s": [round(words[0]["start"], 2), round(words[-1]["end"], 2)] if words else None,
            "media_duration_s": round(duration, 3), "runtime_s": round(time.time() - t0, 2), "device": device}
    arrays = {"times": np.array([(c["start"] + c["end"]) / 2 for c in chunks], np.float32),
              "starts": np.array([c["start"] for c in chunks], np.float32),
              "ends": np.array([c["end"] for c in chunks], np.float32),
              "n_words": np.array([c["n_words"] for c in chunks], np.int32),
              "emb": np.asarray(emb, np.float32)}
    save_cache(dirs["cache"], item.video_id, NAME, meta, **arrays)
    return arrays, meta
