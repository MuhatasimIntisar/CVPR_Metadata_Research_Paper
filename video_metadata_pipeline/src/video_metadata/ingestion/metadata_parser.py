"""Normalise Internet Archive item metadata (metadata.json = the IA metadata API response) into video-level context.

The raw document is preserved unchanged under raw_metadata. Subjects and descriptions are video-level context,
never moment-level ground truth (role field)."""
from __future__ import annotations

import re

from .transcript_parser import lang_code

RIGHTS_KEYS = ["licenseurl", "rights", "possible-copyright-status", "copyright", "license"]


def _first(v):
    if isinstance(v, list):
        return v[0] if v else None
    return v


def _list(v):
    if v is None:
        return []
    return [x for x in (v if isinstance(v, list) else re.split(r"\s*;\s*", str(v))) if x]


def _duration(md: dict) -> float | None:
    for k in ("duration", "length"):
        try:
            return float(_first(md.get(k)))
        except (TypeError, ValueError):
            pass
    rt = _first(md.get("runtime"))
    if rt:
        parts = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", str(rt))]
        if parts:
            s = 0.0
            for x in parts:
                s = s * 60 + x
            return s
    return None


def normalise(video_id: str, doc: dict | None, files_doc: dict | None, probed_duration: float | None) -> tuple[dict, list[str]]:
    issues = []
    if doc is None:
        issues.append("metadata.json missing")
        doc = {}
    md = doc.get("metadata", doc) if isinstance(doc, dict) else {}
    ident = _first(md.get("identifier")) or (files_doc or {}).get("identifier") or video_id
    lang = _first(md.get("language"))
    dur_meta = _duration(md)
    formats = sorted({f.get("format") for f in doc.get("files", []) if isinstance(f, dict) and f.get("format")})
    yt = {k: v for k, v in md.items() if k.startswith("youtube-")}
    rights = {k: md.get(k) for k in RIGHTS_KEYS if md.get(k)}
    if not rights:
        issues.append("no rights/licence field in source metadata")
    for k in ("title", "description", "date", "language", "creator"):
        if not md.get(k):
            issues.append(f"missing {k}")
    if dur_meta and probed_duration and abs(dur_meta - probed_duration) > 2:
        issues.append(f"duration mismatch: metadata {dur_meta} s vs file {probed_duration:.1f} s")
    out = {
        "video_id": video_id, "ia_identifier": ident, "source_url": f"https://archive.org/details/{ident}",
        "title": _first(md.get("title")), "description": _first(md.get("description")),
        "subjects": _list(md.get("subject")), "creator": _first(md.get("creator")),
        "language": lang, "language_code": lang_code(lang),
        "publication_date": _first(md.get("date")) or _first(md.get("publicdate")),
        "archive_public_date": _first(md.get("publicdate")),
        "duration_s": probed_duration if probed_duration else dur_meta, "duration_metadata_s": dur_meta,
        "formats": formats, "rights": rights or {"status": "not stated in source metadata"},
        "collections": _list(md.get("collection")),
        "original_platform": ({"platform": "YouTube", "url": yt.get("youtube-webpage-url"), "uploader": yt.get("youtube-uploader"),
                               "id": yt.get("youtube-id")} if yt else {}),
        "raw_metadata": doc,
        "role": "video-level context: source tags/descriptions are not moment-level ground truth",
    }
    return out, issues
