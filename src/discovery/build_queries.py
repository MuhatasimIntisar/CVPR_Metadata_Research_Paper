"""(1) Build YouTube search queries from the taxonomy's search seeds.

Queries combine action x place x location x style so that collection is driven by
attribute combinations, not random scraping.
"""
from __future__ import annotations

import itertools
import random


def build_queries(taxonomy: dict, max_queries: int, seed: int = 0) -> list[str]:
    s = taxonomy["search_seeds"]
    combos = list(itertools.product(s["actions"], s["places"], s["locations"], s["styles"]))
    random.Random(seed).shuffle(combos)  # deterministic, but spreads coverage when capped
    return [f"{a} {p} {loc} {style}" for a, p, loc, style in combos[:max_queries]]
