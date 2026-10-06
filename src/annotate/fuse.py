"""Fuse all annotator outputs into one label per (shot, field) with provenance.

Rule (placeholder): highest-confidence value wins; all contributing sources are recorded;
'agreement' = number of sources that gave the winning value. Shots/fields with conflicts or
low confidence will later be routed to the human review queue.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

FUSED_VERSION = "0.0-stub"


def fuse(parts: list[pd.DataFrame], out_path: Path) -> pd.DataFrame:
    df = pd.concat(parts, ignore_index=True)
    df = df.sort_values("confidence", ascending=False)
    winner = df.groupby(["shot_id", "field"], as_index=False).first()
    sources = df.groupby(["shot_id", "field"])["source"].apply(lambda s: ",".join(sorted(set(s))))
    winner = winner.merge(sources.rename("all_sources").reset_index(), on=["shot_id", "field"])
    agree = df.merge(winner[["shot_id", "field", "value"]], on=["shot_id", "field", "value"])
    winner = winner.merge(agree.groupby(["shot_id", "field"]).size().rename("agreement").reset_index(),
                          on=["shot_id", "field"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    winner.to_parquet(out_path, index=False)
    print(f"    [fuse] {len(winner)} (shot, field) labels -> {out_path.name}  ({len(df)} raw rows in)")
    return winner
