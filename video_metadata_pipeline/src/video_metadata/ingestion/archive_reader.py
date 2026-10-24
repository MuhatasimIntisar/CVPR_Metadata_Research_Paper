"""Discover Internet Archive item folders and inventory their files (no assumptions about exact names)."""
from __future__ import annotations

from pathlib import Path

from ..utils.io import read_json
from .transcript_parser import NAME


def find_items(root: Path, video_exts, max_depth: int = 3) -> list[Path]:
    """Folders that contain a metadata.json or a video file (searched up to max_depth below root)."""
    root = Path(root)
    out = []
    for p in sorted(root.rglob("*")):
        if not p.is_dir() or len(p.relative_to(root).parts) > max_depth:
            continue
        names = {c.name for c in p.iterdir() if c.is_file()}
        if "metadata.json" in names or any(Path(n).suffix.lower() in video_exts for n in names):
            out.append(p)
    return out


def inventory(item: Path, video_exts) -> dict:
    files = sorted(c for c in item.iterdir() if c.is_file())
    videos = [c for c in files if c.suffix.lower() in video_exts]
    video = next((c for c in videos if c.stem == "video"), videos[0] if videos else None)
    caps = []
    for c in files:
        if c.suffix.lower() in (".srt", ".vtt"):
            m = NAME.search(c.name)
            caps.append({"file": c.name, "language": m.group("lang") if m else None,
                         "format": c.suffix[1:].lower(), "filename_marks_asr": bool(m and m.group("asr"))})
    fj = read_json(item / "files.json") if (item / "files.json").exists() else None
    return {"video": video.name if video else None, "other_videos": [v.name for v in videos if v != video],
            "metadata_json": (item / "metadata.json").exists(), "files_json": fj is not None,
            "thumbnail": next((c.name for c in files if c.suffix.lower() in (".jpg", ".jpeg", ".png")), None),
            "captions": caps, "original_names": (fj or {}).get("files", {}),
            "all_files": [c.name for c in files]}
