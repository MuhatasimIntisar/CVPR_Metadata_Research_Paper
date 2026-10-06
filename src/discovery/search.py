"""(1) Search YouTube for candidate video IDs. Metadata only: no video is downloaded here."""
from __future__ import annotations

import yt_dlp


def search(query: str, n: int, min_s: int, max_s: int) -> list[dict]:
    opts = {"quiet": True, "no_warnings": True, "extract_flat": True, "skip_download": True}
    with yt_dlp.YoutubeDL(opts) as ydl:
        res = ydl.extract_info(f"ytsearch{n * 3}:{query}", download=False)  # over-fetch, then filter
    out = []
    for e in res.get("entries") or []:
        dur = e.get("duration")
        if not e.get("id") or dur is None or not (min_s <= dur <= max_s):
            continue
        if e.get("live_status") in ("is_live", "is_upcoming"):
            continue
        out.append({"video_id": e["id"], "query": query, "title": e.get("title"), "duration_s": float(dur),
                    "channel": e.get("channel_id") or e.get("channel") or e.get("uploader")})
        if len(out) >= n:
            break
    return out
