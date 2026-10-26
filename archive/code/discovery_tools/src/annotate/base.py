"""Annotator interface. Each annotator fills some of the 12 taxonomy fields for every kept shot.

Output: annotations/<name>/<version>/<shard_id>.parquet with one row per (shot_id, field):
    shot_id, field, value, confidence, source, evidence
Annotators are independent, so any one can be re-run or swapped without touching the others.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

ROW_COLUMNS = ["shot_id", "field", "value", "confidence", "source", "evidence"]


@dataclass
class Annotator:
    name: str
    version: str
    dimension: str
    fields: list[str]
    inputs: list[str]                 # which shard parts it reads: clips, keyframes, captions, info
    references: list[str] = field(default_factory=list)

    def output_path(self, annotations_dir: Path, shard_id: str) -> Path:
        return annotations_dir / self.name / self.version / f"{shard_id}.parquet"

    def annotate(self, shots: pd.DataFrame, info: pd.DataFrame, shard_dir: Path, device: str) -> pd.DataFrame:
        """STUB: prints what would run and emits 'unknown' for every field it owns."""
        print(f"    [annotator stub] {self.name} v{self.version} ({self.dimension}) "
              f"-> {', '.join(self.fields)}  on {len(shots)} shots, device={device}, inputs={self.inputs}")
        for ref in self.references:
            print(f"        ref: {ref}")
        rows = [{"shot_id": sid, "field": f, "value": "unknown", "confidence": 0.0,
                 "source": f"{self.name}@{self.version}", "evidence": "stub"}
                for sid in shots["shot_id"] for f in self.fields]
        return pd.DataFrame(rows, columns=ROW_COLUMNS)
