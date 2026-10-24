"""Transcript with word timestamps: Whisper large-v3-turbo (transformers pipeline).
Not a segmenter itself; speakers and topic read its words and sentences.

Default mode "segmented": the audio is cut into <= 28 s pieces at the quietest point near each
piece's end (energy-based, no model), every piece is transcribed on its own as short-form audio,
and its start time is added back to its word timestamps. Nothing is stitched, so the chunk-overlap
errors of the pipeline's parallel chunking (duplicated words, timestamps jumping backwards) cannot
occur. Other modes: "chunked" (pipeline chunking) and "longform" (Whisper sequential decoding).
Words always pass through clean_words(); what it fixed is reported in stats.
"""
from __future__ import annotations

import re
import time
from pathlib import Path

from .common import free_gpu, load_audio, r3, write_result

NAME = "asr"
MODEL = "openai/whisper-large-v3-turbo"


def to_sentences(words: list[dict], max_gap: float = 1.0, max_words: int = 40) -> list[dict]:
    sents, cur = [], []
    for w in words:
        if cur and (w["start"] - cur[-1]["end"] > max_gap or len(cur) >= max_words):
            sents.append(cur)
            cur = []
        cur.append(w)
        if re.search(r"[.?!¿¡]$", w["w"].strip()):
            sents.append(cur)
            cur = []
    if cur:
        sents.append(cur)
    return [{"id": i, "start": s[0]["start"], "end": s[-1]["end"],
             "text": " ".join(w["w"].strip() for w in s).strip(),
             "word_ids": [s[0]["id"], s[-1]["id"]]} for i, s in enumerate(sents)]


def clean_words(words: list[dict], duration: float, max_word_s: float = 2.0, back_tol: float = 0.1):
    """Drop words that jump back in time (duplicates) or start after the video ends; clamp ends to
    the video end and the next word's start; cap implausibly long words. Returns (words, report)."""
    rep = {"dropped_backwards": 0, "dropped_past_end": 0, "capped_long": 0, "clamped_end": 0}
    kept, last_start = [], -1.0
    for w in words:
        if w["start"] >= duration:
            rep["dropped_past_end"] += 1
            continue
        if w["start"] < last_start - back_tol:
            rep["dropped_backwards"] += 1
            continue
        kept.append(dict(w))
        last_start = w["start"]
    for i, w in enumerate(kept):
        limit = duration
        if i + 1 < len(kept) and kept[i + 1]["start"] > w["start"]:
            limit = min(limit, kept[i + 1]["start"])
        if w["end"] > limit:
            w["end"] = limit
            rep["clamped_end"] += 1
        if w["end"] - w["start"] > max_word_s:
            w["end"] = w["start"] + max_word_s
            rep["capped_long"] += 1
        if w["end"] <= w["start"]:
            w["end"] = w["start"] + 0.05
        w["id"] = i
        w["start"], w["end"] = r3(w["start"]), r3(w["end"])
    return kept, rep


def split_points(audio, sr: int, max_s: float = 28.0, min_s: float = 12.0, frame_s: float = 0.05):
    """Piece boundaries (seconds): each cut is the quietest 0.3 s within [start+min_s, start+max_s]."""
    import numpy as np
    hop = int(frame_s * sr)
    n = len(audio) // hop
    if n == 0:
        return [0.0, len(audio) / sr]
    rms = np.sqrt(np.mean(audio[:n * hop].reshape(n, hop) ** 2, axis=1) + 1e-12)
    k = max(1, int(0.3 / frame_s))
    smooth = np.convolve(rms, np.ones(k) / k, mode="same")
    total = len(audio) / sr
    cuts, start = [0.0], 0.0
    while total - start > max_s:
        lo, hi = int((start + min_s) / frame_s), int((start + max_s) / frame_s)
        i = lo + int(np.argmin(smooth[lo:hi]))
        start = i * frame_s
        cuts.append(start)
    cuts.append(total)
    return cuts


def run(video: Path, out_dir: Path, info: dict, wav: Path, device: str = "cpu",
        language: str | None = None, mode: str = "segmented", batch_size: int | None = None, **_) -> Path:
    import inspect

    import torch
    from transformers import pipeline
    t0 = time.time()
    audio, sr = load_audio(wav)
    dtype = torch.float16 if device == "cuda" else torch.float32
    dkey = "dtype" if "dtype" in inspect.signature(pipeline).parameters else "torch_dtype"  # renamed in v4.56+
    if batch_size is None:  # chunked mode only; word timestamps keep cross-attention per chunk
        gb = torch.cuda.get_device_properties(0).total_memory / 2**30 if device == "cuda" else 0
        batch_size = 1 if gb < 8 else 2 if gb < 12 else 4 if gb < 20 else 8
    asr = pipeline("automatic-speech-recognition", model=MODEL, device=device, **{dkey: dtype})
    gen = {"task": "transcribe"}
    if language:
        gen["language"] = language

    def call(pipe, **kw):
        try:
            return pipe({"raw": audio, "sampling_rate": sr}, return_timestamps="word", return_language=True,
                        generate_kwargs=gen, **kw)
        except (TypeError, ValueError):  # versions without return_language + word timestamps
            return pipe({"raw": audio, "sampling_rate": sr}, return_timestamps="word", generate_kwargs=gen, **kw)

    used = mode
    res = None
    pieces = None
    if mode == "segmented":
        cuts = split_points(audio, sr)
        pieces = list(zip(cuts, cuts[1:]))
        print(f"       whisper segmented: {len(pieces)} pieces <= 28 s, batch_size={batch_size}", flush=True)
        inputs = [{"raw": audio[int(a * sr):int(b * sr)], "sampling_rate": sr} for a, b in pieces]
        chunks, texts = [], []
        i = 0
        failed = []

        def transcribe_part(part, pipe=asr):
            try:
                return pipe([dict(x) for x in part], return_timestamps="word", return_language=True,
                           generate_kwargs=gen, batch_size=len(part))
            except (TypeError, ValueError):
                return pipe([dict(x) for x in part], return_timestamps="word", generate_kwargs=gen,
                           batch_size=len(part))

        while i < len(inputs):
            part = inputs[i:i + batch_size]
            span = pieces[i:i + len(part)]
            try:
                outs = transcribe_part(part)
            except torch.OutOfMemoryError:
                free_gpu()
                if batch_size == 1:
                    raise
                batch_size = max(1, batch_size // 2)
                print(f"       out of GPU memory, batch_size={batch_size}", flush=True)
                continue
            except Exception as e:  # noqa: BLE001 - Whisper word alignment fails on pieces with no speech
                outs = []
                for x, (a, b) in zip(part, span):
                    try:
                        outs.append(transcribe_part([x])[0])
                    except Exception as e1:  # noqa: BLE001
                        failed.append({"start": r3(a), "end": r3(b), "error": f"{type(e1).__name__}: {e1}"[:120]})
                        outs.append({"text": "", "chunks": []})
                if not failed:
                    print(f"\n       batch error ({type(e).__name__}), retried piece by piece", flush=True)
            for (a, _b), o in zip(span, outs):
                texts.append(o.get("text", "").strip())
                for c in o.get("chunks", []):
                    s0, e0 = c["timestamp"]
                    chunks.append({**c, "timestamp": (None if s0 is None else s0 + a,
                                                      None if e0 is None else e0 + a)})
            i += len(part)
            print(f"\r       {min(i, len(inputs))}/{len(inputs)} pieces"
                  f"{f'  ({len(failed)} without speech/failed)' if failed else ''}", end="", flush=True)
        print()
        res = {"text": " ".join(texts), "chunks": chunks}
    if mode == "longform":
        try:
            print("       whisper sequential long-form", flush=True)
            res = call(asr)
        except torch.OutOfMemoryError:
            raise
        except Exception as e:  # noqa: BLE001 - fall back to chunked mode
            print(f"       long-form failed ({type(e).__name__}: {e}); falling back to chunked", flush=True)
            free_gpu()
            used = "chunked"
    if res is None:
        while True:
            try:
                print(f"       whisper chunked, batch_size={batch_size}", flush=True)
                res = call(asr, chunk_length_s=30, stride_length_s=5, batch_size=batch_size)
                break
            except torch.OutOfMemoryError:
                free_gpu()
                if batch_size == 1:
                    raise
                batch_size = max(1, batch_size // 2)
                print(f"       out of GPU memory, retrying with batch_size={batch_size}", flush=True)
    del asr
    free_gpu()

    raw, langs = [], {}
    for c in res.get("chunks", []):
        s, e = c["timestamp"]
        if s is None:
            continue
        e = e if e is not None and e > s else s + 0.3
        raw.append({"w": c["text"], "start": float(s), "end": float(e)})
        if c.get("language"):
            langs[c["language"]] = langs.get(c["language"], 0) + 1
    words, report = clean_words(raw, info["duration"])
    detected = max(langs, key=langs.get) if langs else (language or "unknown")
    sents = to_sentences(words)
    speech_s = sum(w["end"] - w["start"] for w in words)
    params = {"language": language or "auto", "mode": used}
    if pieces is not None:
        params.update(n_pieces=len(pieces), max_piece_s=28.0, batch_size=batch_size,
                      failed_pieces=failed)
    if used == "chunked":
        params.update(chunk_length_s=30, stride_length_s=5, batch_size=batch_size)
    return write_result(out_dir, NAME, MODEL, params, t0,
                        stats={"n_words": len(words), "n_raw_words": len(raw), "n_sentences": len(sents),
                               "language": detected, "speech_coverage": r3(speech_s / max(info["duration"], 1e-6)),
                               "cleanup": report},
                        text=res.get("text", "").strip(), words=words, sentences=sents)
