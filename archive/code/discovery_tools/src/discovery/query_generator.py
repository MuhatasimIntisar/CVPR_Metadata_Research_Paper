"""(1) Coverage-aware, randomised query generation.

Instead of hand-editing search terms, each batch samples combinations of the coverage slots
(action x place x location) with weights that favour under-represented combinations, aiming for
EVEN coverage across combinations:

    weight(combo) = 0                                if count >= max_per_combo   (ceiling)
                  = 1 / (1 + count) ** alpha         otherwise

Combos with zero videos get the highest weight, so the floor fills first. Phrasings and the
non-coverage slots (style) are sampled uniformly. Every query records its batch id, seed and
canonical slot keys in the query log, so any batch can be reproduced and reported.

Coverage is currently counted from the *search* slot keys. Once annotators exist, swap
`coverage_counts` to count from the fused labels instead.
"""
from __future__ import annotations

import itertools
import random
from dataclasses import dataclass

import pandas as pd

LOG_COLUMNS = ["video_id", "batch_id", "seed", "query", "combo", "channel", "added"]


@dataclass
class Query:
    text: str
    slots: dict[str, str]          # canonical key per slot, e.g. {"action": "dancing", ...}
    combo: str                     # "dancing|street|Belfast" (coverage slots only)


class QueryGenerator:
    def __init__(self, pools: dict, max_per_combo: int, alpha: float = 1.0):
        self.slots = pools["slots"]
        self.coverage_slots = pools["coverage_slots"]
        self.other_slots = [s for s in self.slots if s not in self.coverage_slots]
        self.combos = list(itertools.product(*(self.slots[s].keys() for s in self.coverage_slots)))
        self.max_per_combo = max_per_combo
        self.alpha = alpha

    def combo_key(self, combo: tuple) -> str:
        return "|".join(combo)

    def weights(self, counts: dict[str, int]) -> list[float]:
        w = []
        for c in self.combos:
            n = counts.get(self.combo_key(c), 0)
            w.append(0.0 if n >= self.max_per_combo else 1.0 / (1 + n) ** self.alpha)
        return w

    def generate(self, n_queries: int, counts: dict[str, int], seed: int) -> list[Query]:
        rng = random.Random(seed)
        w = self.weights(counts)
        if not any(w):
            return []  # every combo is at its ceiling
        picks = rng.choices(self.combos, weights=w, k=n_queries)
        queries = []
        for combo in picks:
            slots = dict(zip(self.coverage_slots, combo))
            for s in self.other_slots:
                slots[s] = rng.choice(list(self.slots[s].keys()))
            words = [rng.choice(self.slots[s][slots[s]]) for s in self.coverage_slots + self.other_slots]
            queries.append(Query(" ".join(words), slots, self.combo_key(combo)))
        return queries

    def coverage_report(self, counts: dict[str, int]) -> str:
        vals = [counts.get(self.combo_key(c), 0) for c in self.combos]
        empty = sum(v == 0 for v in vals)
        full = sum(v >= self.max_per_combo for v in vals)
        return (f"{len(self.combos)} combos | min {min(vals)} / median {sorted(vals)[len(vals) // 2]} "
                f"/ max {max(vals)} per combo | {empty} empty | {full} at ceiling ({self.max_per_combo})")


def coverage_counts(log: pd.DataFrame, manifest: pd.DataFrame) -> dict[str, int]:
    """Videos per combo, excluding videos that later failed ingest. Each video counts once."""
    if log.empty:
        return {}
    added = log[log["added"]]
    failed = set(manifest.loc[manifest["status"] == "failed", "video_id"])
    added = added[~added["video_id"].isin(failed)].drop_duplicates("video_id")
    return added["combo"].value_counts().to_dict()


def channel_counts(log: pd.DataFrame) -> dict[str, int]:
    if log.empty:
        return {}
    return log.loc[log["added"], "channel"].fillna("?").value_counts().to_dict()
