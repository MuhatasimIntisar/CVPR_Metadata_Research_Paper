"""SRT/VTT caption parsing (including YouTube 'rolling' automatic captions) and primary-transcript selection.

Rolling automatic captions (observed in the Internet Archive downloads) repeat the previous line in every cue
and carry per-word times as inline tags: `You're<00:00:05.600> excited<00:00:05.920> ...`. New text is taken
only from lines not shown in the previous cue; word times come from the inline tags (the first word starts at
the cue start). Cues with no new text are counted as duplicates. Plain (non-rolling) captions pass through
unchanged. Original cue text is preserved alongside the cleaned segments.
"""
from __future__ import annotations

import html
import re
from pathlib import Path

from ..utils.timestamps import parse_ts

ARROW = re.compile(r"^\s*(\S+)\s*-->\s*(\S+)")
INLINE = re.compile(r"<(\d{1,2}:\d{2}:\d{2}[.,]\d{3}|\d{1,2}:\d{2}[.,]\d{3})>")
TAG = re.compile(r"</?[a-zA-Z][^>]*>")
NAME = re.compile(r"\.(?P<lang>[a-z]{2,3}(?:-[A-Za-z]{2,4})?)(?P<asr>\.asr)?\.(?P<fmt>srt|vtt)$", re.I)

LANG_CODES = {"english": "en", "eng": "en", "en": "en", "spanish": "es", "español": "es", "espanol": "es",
              "spa": "es", "es": "es", "french": "fr", "fre": "fr", "fra": "fr", "fr": "fr", "german": "de",
              "ger": "de", "deu": "de", "de": "de", "portuguese": "pt", "por": "pt", "pt": "pt",
              "chinese": "zh", "zho": "zh", "chi": "zh", "zh": "zh"}


def lang_code(name: str | None) -> str | None:
    if not name:
        return None
    return LANG_CODES.get(str(name).strip().lower())


def read_cues(path: Path) -> tuple[list[dict], list[str]]:
    """Generic SRT/VTT cue reader. Returns cues [{start, end, lines}] and parse issues."""
    text = path.read_text(encoding="utf-8-sig", errors="replace").replace("\r\n", "\n").replace("\r", "\n")
    # Line-based: a cue runs from one '-->' line to the next. (VTT cues may contain whitespace-only lines,
    # so blank-line block splitting is unsafe.) SRT index lines (digits followed by an arrow line) are dropped.
    cues, issues, cur = [], [], None
    lines = text.split("\n")
    for i, line in enumerate(lines):
        if "-->" in line:
            m = ARROW.match(line)
            try:
                a, b = parse_ts(m.group(1)), parse_ts(m.group(2))
            except (ValueError, AttributeError):
                issues.append(f"malformed timestamp line: {line[:60]!r}")
                cur = None
                continue
            cur = {"start": a, "end": b, "lines": []}
            cues.append(cur)
        elif cur is not None and line.strip():
            if line.strip().isdigit() and i + 1 < len(lines) and "-->" in lines[i + 1]:
                continue
            cur["lines"].append(line)
    return cues, issues


def _words_from_line(raw: str, start: float, end: float) -> tuple[list[dict], bool]:
    """Split one caption line into words; inline tags give word start times."""
    parts = INLINE.split(raw)  # text, ts, text, ts, ...
    timed = len(parts) > 1
    tokens = []  # (start_time or None, text)
    t0 = start
    for i, p in enumerate(parts):
        if i % 2 == 1:
            t0 = parse_ts(p)
            continue
        txt = html.unescape(TAG.sub("", p)).strip()
        for j, w in enumerate(txt.split()):
            tokens.append((t0 if j == 0 else None, w))
    if not tokens:
        return [], timed
    if not timed:  # spread evenly over the cue (approximate)
        step = (end - start) / len(tokens)
        return [{"start": round(start + k * step, 3), "end": round(start + (k + 1) * step, 3), "w": w,
                 "approx": True} for k, (_, w) in enumerate(tokens)], False
    starts, last = [], start
    for t, _ in tokens:
        last = t if t is not None else last
        starts.append(last)
    out = []
    for k, (s, (_, w)) in enumerate(zip(starts, tokens)):
        e = starts[k + 1] if k + 1 < len(starts) and starts[k + 1] > s else end
        out.append({"start": round(s, 3), "end": round(max(e, s), 3), "w": w, "approx": False})
    return out, True


def clean(line: str) -> str:
    return " ".join(html.unescape(TAG.sub("", INLINE.sub("", line))).split())


def parse_track(path: Path, duration: float | None = None) -> dict:
    cues, issues = read_cues(path)
    segments, words, dup, prev, timed_any = [], [], 0, set(), False
    bad_order = out_of_range = 0
    for c in cues:
        if c["end"] < c["start"]:
            bad_order += 1
            c["end"] = c["start"]
        if duration and c["start"] > duration + 1.0:
            out_of_range += 1
            continue
        raw_lines = [l for l in c["lines"] if l.strip()]
        cleaned = [clean(l) for l in raw_lines]
        new = [(r, cl) for r, cl in zip(raw_lines, cleaned) if cl and cl not in prev]
        prev = {cl for cl in cleaned if cl}
        if not new:
            dup += 1
            continue
        for raw, cl in new:
            ws, timed = _words_from_line(raw, c["start"], c["end"])
            timed_any |= timed
            words += ws
            segments.append({"start": round(c["start"], 3), "end": round(c["end"], 3), "text": cl,
                             "speaker_change_marker": cl.startswith(">>"), "word_timed": timed})
    overlaps = sum(1 for a, b in zip(segments, segments[1:]) if b["start"] < a["start"] - 0.05)
    span, cur = 0.0, None
    for s in sorted(segments, key=lambda s: s["start"]):
        if cur and s["start"] <= cur[1]:
            cur[1] = max(cur[1], s["end"])
        else:
            if cur:
                span += cur[1] - cur[0]
            cur = [s["start"], s["end"]]
    if cur:
        span += cur[1] - cur[0]
    m = NAME.search(path.name)
    return {"file": path.name, "language": m.group("lang") if m else None, "format": (m.group("fmt") if m else path.suffix[1:]).lower(),
            "filename_marks_asr": bool(m and m.group("asr")), "n_cues": len(cues), "n_segments": len(segments),
            "n_words": len(words), "duplicate_cue_ratio": round(dup / len(cues), 3) if cues else 0.0,
            "rolling_format": dup > 0.3 * max(1, len(cues)), "word_timing": timed_any,
            "coverage_s": round(span, 1), "issues": {"parse": issues, "end_before_start": bad_order,
                                                    "out_of_range": out_of_range, "segment_order_overlaps": overlaps},
            "segments": segments, "words": words}


def discover_tracks(item_dir: Path) -> list[Path]:
    return sorted(p for p in item_dir.iterdir() if p.is_file() and p.suffix.lower() in (".srt", ".vtt"))


def select_primary(tracks: list[dict], item_lang: str | None, duration: float | None, cfg: dict) -> tuple[dict | None, str]:
    """Documented policy: (1) tracks in the item's own language beat other languages (others are treated as
    likely machine translations); (2) word-timed tracks beat untimed; (3) higher speech coverage; (4) fewer
    issues; (5) SRT before VTT (same content in these downloads). Coverage below min_coverage_ratio x duration
    or < 20 words = inadequate."""
    if not tracks:
        return None, "no caption tracks"
    pol = cfg.get("primary_transcript", {})

    def key(t):
        same = (t["language"] or "").split("-")[0] == (item_lang or "")
        n_issues = len(t["issues"]["parse"]) + t["issues"]["end_before_start"] + t["issues"]["out_of_range"]
        return (same if pol.get("prefer_original_language", True) else 0, t["word_timing"], t["coverage_s"],
                -n_issues, t["format"] == "srt")

    best = max(tracks, key=key)
    ratio = best["coverage_s"] / duration if duration else None
    if best["n_words"] < 20 or (ratio is not None and ratio < pol.get("min_coverage_ratio", 0.2)):
        return None, f"best track {best['file']} inadequate (words {best['n_words']}, coverage ratio {ratio})"
    return best, (f"selected {best['file']}: language {best['language']} "
                  f"({'matches' if (best['language'] or '').split('-')[0] == item_lang else 'differs from'} item language "
                  f"{item_lang}), word timing {best['word_timing']}, coverage {best['coverage_s']} s")


def from_whisper(asr: dict) -> dict:
    """Convert an earlier Whisper output (segmenters asr.json) into the track structure."""
    words = [{"start": w["start"], "end": w["end"], "w": w["w"].strip(), "approx": False} for w in asr.get("words", [])]
    segs = [{"start": s["start"], "end": s["end"], "text": s["text"], "speaker_change_marker": False, "word_timed": True}
            for s in asr.get("sentences", [])]
    cov = sum(s["end"] - s["start"] for s in segs)
    return {"file": "asr.json (Whisper, earlier run)", "language": (asr.get("stats") or {}).get("language"),
            "format": "whisper_json", "filename_marks_asr": True, "n_cues": len(segs), "n_segments": len(segs),
            "n_words": len(words), "duplicate_cue_ratio": 0.0, "rolling_format": False, "word_timing": True,
            "coverage_s": round(cov, 1), "issues": {"parse": [], "end_before_start": 0, "out_of_range": 0,
                                                    "segment_order_overlaps": 0},
            "segments": segs, "words": words, "model": asr.get("model")}
