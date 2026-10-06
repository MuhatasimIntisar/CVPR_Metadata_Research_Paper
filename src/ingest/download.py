"""(2a) Download one video + its info.json + YouTube captions into a temp folder."""
from __future__ import annotations

import json
from pathlib import Path

import yt_dlp

# Fields kept from info.json. The full file is mostly the list of available formats (~90% of its size).
INFO_FIELDS = ["id", "title", "description", "upload_date", "channel", "channel_id", "uploader",
               "duration", "tags", "categories", "location", "language", "webpage_url"]


def download(video_id: str, out_dir: Path, max_height: int) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    url = f"https://www.youtube.com/watch?v={video_id}"
    h = max_height
    base = {"quiet": True, "no_warnings": True, "outtmpl": str(out_dir / f"{video_id}.%(ext)s")}

    # Video + info.json
    video_opts = {**base,
                  "format": (f"bv*[height<={h}][ext=mp4]+ba[ext=m4a]/b[height<={h}][ext=mp4]/"
                             f"bv*[height<={h}]+ba/b[height<={h}]/b"),
                  "merge_output_format": "mp4",
                  "writeinfojson": True}
    with yt_dlp.YoutubeDL(video_opts) as ydl:
        ydl.download([url])

    # Captions in a separate call: caption rate-limits should not fail the whole video.
    sub_opts = {**base, "skip_download": True, "writesubtitles": True, "writeautomaticsub": True,
                "subtitleslangs": ["en", "en-GB", "en-US"], "subtitlesformat": "vtt"}
    try:
        with yt_dlp.YoutubeDL(sub_opts) as ydl:
            ydl.download([url])
    except Exception as e:  # noqa: BLE001
        print(f"      captions unavailable for {video_id}: {type(e).__name__}")

    video = next(iter(sorted(out_dir.glob(f"{video_id}.mp4"))), None)
    if video is None:
        raise FileNotFoundError(f"no mp4 produced for {video_id}")
    info_full = json.loads((out_dir / f"{video_id}.info.json").read_text(encoding="utf-8"))
    info = {k: info_full.get(k) for k in INFO_FIELDS}
    subs = sorted(out_dir.glob(f"{video_id}.*.vtt"))
    # Prefer manual English captions over auto-generated ones when both exist.
    manual = set((info_full.get("subtitles") or {}).keys())
    subs.sort(key=lambda p: (p.suffixes[-2].lstrip(".") not in manual, p.name))
    info["caption_source"] = None
    if subs:
        lang = subs[0].suffixes[-2].lstrip(".")
        info["caption_source"] = "manual" if lang in manual else "auto"
    return {"video": video, "info": info, "captions": subs[0] if subs else None}
