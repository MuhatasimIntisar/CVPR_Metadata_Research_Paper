"""The manifest: one row per YouTube video ID, tracking where it is in the pipeline.

Status flow:  candidate -> ingested  (or failed)
Annotation progress is tracked per shard by the presence of annotation files, not here.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PIPELINE_VERSION = "0.1.0"
COLUMNS = ["video_id", "query", "title", "duration_s", "status", "shard_id",
           "n_shots", "n_kept", "error", "pipeline_version", "updated_at"]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame(columns=COLUMNS)


def save(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)  # atomic swap so a crash never leaves a half-written manifest


def add_candidates(df: pd.DataFrame, rows: list[dict]) -> tuple[pd.DataFrame, int]:
    known = set(df["video_id"])
    new = [r for r in rows if r["video_id"] not in known]
    for r in new:
        r.update(status="candidate", shard_id=None, n_shots=None, n_kept=None,
                 error=None, pipeline_version=PIPELINE_VERSION, updated_at=_now())
    if new:
        df = pd.concat([df, pd.DataFrame(new, columns=COLUMNS)], ignore_index=True)
    return df, len(new)


def update(df: pd.DataFrame, video_id: str, **fields) -> pd.DataFrame:
    idx = df.index[df["video_id"] == video_id]
    for k, v in {**fields, "updated_at": _now(), "pipeline_version": PIPELINE_VERSION}.items():
        df.loc[idx, k] = v
    return df


def next_shard_id(shards_dir: Path) -> str:
    existing = sorted(p.name for p in shards_dir.glob("shard-*")) if shards_dir.exists() else []
    n = int(existing[-1].split("-")[1]) + 1 if existing else 0
    return f"shard-{n:05d}"
