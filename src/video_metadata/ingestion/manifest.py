"""Corpus manifest + per-video ingestion (source_metadata, transcript, source_manifest, ingestion_report)."""
from __future__ import annotations

import datetime as dt
import platform
from pathlib import Path

from .. import PIPELINE_VERSION
from ..schemas import SOURCE_METADATA, TRANSCRIPT, validate
from ..utils.io import file_hash, read_json, write_json
from ..utils.video import probe
from . import archive_reader, metadata_parser, transcript_parser


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def build_manifest(cfg: dict, corpus_root: str | Path) -> dict:
    root = Path(corpus_root)
    exts = set(cfg["ingestion"]["video_extensions"])
    entries, seen_ids, seen_ident = [], {}, {}
    for item in archive_reader.find_items(root, exts):
        inv = archive_reader.inventory(item, exts)
        md = read_json(item / "metadata.json") or {}
        ident = (md.get("metadata") or {}).get("identifier")
        vid = item.name
        issues = []
        if not inv["video"]:
            issues.append("no video file in folder")
        if vid in seen_ids:
            issues.append(f"duplicate folder name (also {seen_ids[vid]})")
        if ident and ident in seen_ident:
            issues.append(f"same Internet Archive identifier as {seen_ident[ident]}")
        seen_ids.setdefault(vid, str(item.relative_to(root)))
        if ident and inv["video"]:
            seen_ident.setdefault(ident, vid)
        entries.append({"video_id": vid, "item_dir": item.relative_to(root).as_posix(),
                        "group": item.parent.name if item.parent != root else None,
                        "ia_identifier": ident, "status": "ok" if inv["video"] else "invalid", "issues": issues,
                        "files": {k: inv[k] for k in ("video", "metadata_json", "files_json", "thumbnail")},
                        "captions": [c["file"] for c in inv["captions"]]})
    man = {"created": now(), "pipeline_version": PIPELINE_VERSION, "corpus_root": str(root),
           "n_items": len(entries), "n_valid": sum(e["status"] == "ok" for e in entries), "videos": entries}
    return man


def ingest_video(cfg: dict, entry: dict, corpus_root: Path, out_dir: Path) -> dict:
    item = Path(corpus_root) / entry["item_dir"]
    exts = set(cfg["ingestion"]["video_extensions"])
    inv = archive_reader.inventory(item, exts)
    report = {"video_id": entry["video_id"], "started": now(), "issues": [], "warnings": []}
    video = item / inv["video"] if inv["video"] else None
    info = probe(video) if video else {}
    doc = read_json(item / "metadata.json")
    files_doc = read_json(item / "files.json")
    src, md_issues = metadata_parser.normalise(entry["video_id"], doc, files_doc, info.get("duration"))
    report["warnings"] += md_issues
    report["issues"] += [f"source_metadata: {e}" for e in validate(src, SOURCE_METADATA)]
    write_json(out_dir / "source_metadata.json", src)

    # transcripts
    dur = src["duration_s"]
    tracks = []
    for p in transcript_parser.discover_tracks(item):
        try:
            tracks.append(transcript_parser.parse_track(p, dur))
        except Exception as e:  # malformed file: report, keep going
            report["warnings"].append(f"caption {p.name} unreadable: {type(e).__name__}: {e}")
    item_lang = src["language_code"]
    for t in tracks:
        base = (t["language"] or "").split("-")[0]
        t["provenance"] = ("automatic speech recognition (filename .asr.)" if t["filename_marks_asr"] else
                           "automatic captions (rolling, word-timed)" if t["rolling_format"] else "captions (unknown origin)")
        if item_lang and base and base != item_lang:
            t["provenance"] += f"; language differs from item language ({item_lang}): likely machine translation"
    primary, why = transcript_parser.select_primary(tracks, item_lang, dur, cfg["ingestion"])
    fallback = None
    if primary is None and cfg["ingestion"].get("fallback_asr") == "legacy" and cfg["paths"].get("legacy_segments_dir"):
        asr = read_json(Path(cfg["paths"]["legacy_segments_dir"]) / entry["video_id"] / "asr.json")
        if asr:
            primary = transcript_parser.from_whisper(asr)
            primary["provenance"] = f"Whisper ASR from an earlier run ({asr.get('model')}); no adequate caption track"
            fallback = "legacy_whisper"
            why += "; fell back to earlier Whisper output"
    if primary is None:
        report["warnings"].append(f"no transcript: {why}" + ("; set ingestion.fallback_asr: run to transcribe"
                                                            if cfg["ingestion"].get("fallback_asr") != "run" else ""))
    track_id = lambda t: t["file"]  # noqa: E731
    transcript = {
        "video_id": entry["video_id"],
        "primary": None if primary is None else {k: primary[k] for k in ("file", "language", "format", "provenance",
                                                                          "word_timing", "coverage_s", "n_words")},
        "segments": [] if primary is None else primary["segments"],
        "words": [] if primary is None else primary["words"],
        "tracks": [{k: v for k, v in t.items() if k not in ("segments", "words")} for t in tracks],
        "alternates": {track_id(t): t["segments"] for t in tracks if primary is None or t["file"] != primary["file"]},
        "policy": why, "fallback": fallback,
        "role": "timestamped linguistic evidence (automatic captions may contain errors); not ground truth",
    }
    report["issues"] += [f"transcript: {e}" for e in validate(transcript, TRANSCRIPT)]
    write_json(out_dir / "transcript.json", transcript)

    hashes = {}
    max_mb = cfg["ingestion"]["hash_full_max_mb"]
    for name in inv["all_files"]:
        hashes[name] = file_hash(item / name, max_mb)
    manifest = {"video_id": entry["video_id"], "ia_identifier": src["ia_identifier"], "source_url": src["source_url"],
                "item_dir": str(item), "inputs": {"video": inv["video"], "captions": [c["file"] for c in inv["captions"]],
                                                   "metadata_json": inv["metadata_json"], "files_json": inv["files_json"],
                                                   "thumbnail": inv["thumbnail"], "original_names": inv["original_names"]},
                "media": info, "hashes": hashes,
                "provenance": {"pipeline_version": PIPELINE_VERSION, "ingested": now(), "python": platform.python_version(),
                               "platform": platform.platform()}}
    write_json(out_dir / "source_manifest.json", manifest)
    report.update(finished=now(), video_file=inv["video"], duration_s=dur, n_caption_tracks=len(tracks),
                  primary_transcript=transcript["primary"], transcript_policy=why,
                  inventory={k: inv[k] for k in ("video", "metadata_json", "files_json", "thumbnail")},
                  status="ok" if video and not report["issues"] else ("failed" if not video else "ok_with_issues"))
    write_json(out_dir / "ingestion_report.json", report)
    return report
