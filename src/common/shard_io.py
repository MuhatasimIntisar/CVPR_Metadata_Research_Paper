"""Reading and writing shards.

A shard is an immutable folder:
    shard-00000/
        clips.tar        kept shots only, <video_id>__<shot_idx>.mp4
        keyframes.tar    <video_id>__<shot_idx>.jpg
        shots.parquet    one row per detected shot (kept AND discarded, with reason)
        info.parquet     one row per source video (trimmed YouTube metadata)
        _SUCCESS         written last; a shard without it is incomplete
"""
from __future__ import annotations

import tarfile
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
