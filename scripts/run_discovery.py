"""Stage 1: build search queries from the taxonomy and add candidate video IDs to the manifest.

    python scripts/run_discovery.py --env test_env
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.common import manifest as mf  # noqa: E402
from src.common.config import add_env_arg, load_config  # noqa: E402
from src.discovery.build_queries import build_queries  # noqa: E402
from src.discovery.search import search  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_env_arg(ap)
    args = ap.parse_args()
    cfg = load_config(args.env)
    dc, paths = cfg["discovery"], cfg["paths"]

    queries = build_queries(cfg["taxonomy"], dc["max_queries"])
    print(f"=== Discovery (env={args.env}, data_root={paths.root}) ===")
    rows = []
    for q in queries:
        found = search(q, dc["results_per_query"], dc["min_duration_s"], dc["max_duration_s"])
        print(f"  '{q}': {len(found)} candidates")
        rows += found

    paths.candidates.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(paths.candidates, index=False)
    df, n_new = mf.add_candidates(mf.load(paths.manifest), rows)
    mf.save(df, paths.manifest)
    print(f"\n{n_new} new candidates added. Manifest: {len(df)} videos "
          f"({df['status'].value_counts().to_dict()})")


if __name__ == "__main__":
    main()
