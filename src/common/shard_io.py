"""Reading and writing shards.

A shard is an immutable folder:
    shard-00000/
        clips.tar        kept shots only, <video_id>__<shot_idx>.mp4
        keyframes.tar    <video_id>__<shot_idx>.jpg
        shots.parquet    one row per detected shot (kept AND discarded, with reason)
        info.parquet     one row per source video (trimmed YouTube metadata)
        _SUCCESS         written last; a shard without it is incomplete

Readers for annotators: iter_keyframes() (images, no unpacking) and iter_clips() (one temp file
at a time). keyframes.tar and clips.tar contain exactly the kept shots, so both line up with
read_shots(shard_dir).
"""
from __future__ import annotations

import tarfile
import tempfile
from collections.abc import Iterable, Iterator
from pathlib import Path

import pandas as pd


def is_complete(shard_dir: Path) -> bool:
    return (shard_dir / "_SUCCESS").exists()


def list_complete_shards(shards_dir: Path) -> list[Path]:
    if not shards_dir.exists():
        return []
    return sorted(p for p in shards_dir.glob("shard-*") if is_complete(p))


def write_tar(tar_path: Path, files: list[Path]) -> None:
    # Uncompressed tar: video/JPEG are already compressed, and plain tar allows fast random reads.
    with tarfile.open(tar_path, "w") as tf:
        for f in files:
            tf.add(f, arcname=f.name)


def read_shots(shard_dir: Path, kept_only: bool = True) -> pd.DataFrame:
    df = pd.read_parquet(shard_dir / "shots.parquet")
    return df[df["kept"]].reset_index(drop=True) if kept_only else df


def read_info(shard_dir: Path) -> pd.DataFrame:
    return pd.read_parquet(shard_dir / "info.parquet")


def mark_complete(shard_dir: Path) -> None:
    (shard_dir / "_SUCCESS").write_text("ok\n")


def _members(tf: tarfile.TarFile, names: Iterable[str] | None) -> list[tarfile.TarInfo]:
    if names is None:
        return [m for m in tf.getmembers() if m.isfile()]
    index = {m.name: m for m in tf.getmembers()}
    missing = [n for n in names if n not in index]
    if missing:
        raise KeyError(f"{len(missing)} not in archive, e.g. {missing[:3]}")
    return [index[n] for n in names]


def iter_keyframes(shard_dir: Path, shot_ids: Iterable[str] | None = None) -> Iterator[tuple[str, "Image.Image"]]:
    """Yield (shot_id, PIL RGB image) straight from keyframes.tar, without unpacking to disk.

    shot_ids=None -> every keyframe in the shard (= every kept shot).
        for sid, img in iter_keyframes(shard_dir, shots["shot_id"]): ...
    """
    from PIL import Image

    names = None if shot_ids is None else [f"{sid}.jpg" for sid in shot_ids]
    with tarfile.open(shard_dir / "keyframes.tar") as tf:
        for m in _members(tf, names):
            with tf.extractfile(m) as f:
                img = Image.open(f)
                img.load()
            yield m.name[:-4], img.convert("RGB")


def iter_clips(shard_dir: Path, shot_ids: Iterable[str] | None = None) -> Iterator[tuple[str, Path]]:
    """Yield (shot_id, path to a temporary .mp4) one clip at a time.

    Video decoders (OpenCV, decord, ffmpeg) need a real file, so each clip is unpacked to a temp
    folder, yielded, then deleted before the next one: disk use stays at one clip.
        for sid, path in iter_clips(shard_dir, shots["shot_id"]): cv2.VideoCapture(str(path)) ...
    """
    names = None if shot_ids is None else [f"{sid}.mp4" for sid in shot_ids]
    with tempfile.TemporaryDirectory(prefix="clips_") as tmp, tarfile.open(shard_dir / "clips.tar") as tf:
        for m in _members(tf, names):
            path = Path(tmp) / m.name
            with tf.extractfile(m) as src, open(path, "wb") as dst:
                dst.write(src.read())
            try:
                yield m.name[:-4], path
            finally:
                try:
                    path.unlink(missing_ok=True)
                except OSError:  # Windows: caller still holds the file open; the temp folder cleans it up
                    pass
