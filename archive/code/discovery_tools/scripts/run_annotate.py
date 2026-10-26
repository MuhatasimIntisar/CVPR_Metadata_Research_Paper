"""Stage 3: run annotators over complete shards, then fuse their outputs.

    python scripts/run_annotate.py --env test_env                       # all shards, all annotators
    python scripts/run_annotate.py --env live --shard shard-00003        # one shard (used by SLURM arrays)
    python scripts/run_annotate.py --env test_env --annotators action_vlm --force
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src import annotate  # noqa: E402
from src.annotate.fuse import FUSED_VERSION, fuse  # noqa: E402
from src.common import shard_io  # noqa: E402
from src.common.config import add_env_arg, load_config  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_env_arg(ap)
    ap.add_argument("--shard", default=None, help="only this shard id")
    ap.add_argument("--annotators", nargs="*", default=None, help="subset of annotator names")
    ap.add_argument("--force", action="store_true", help="re-run even if output exists")
    args = ap.parse_args()
    cfg = load_config(args.env)
    paths, device = cfg["paths"], cfg["annotation"]["device"]

    shards = shard_io.list_complete_shards(paths.shards)
    if args.shard:
        shards = [s for s in shards if s.name == args.shard]
    if not shards:
        print("No complete shards found. Run ingest first.")
        return
    annotators = annotate.get(args.annotators)

    for sd in shards:
        shots, info = shard_io.read_shots(sd), shard_io.read_info(sd)
        print(f"\n=== Annotate {sd.name}: {len(shots)} kept shots, {len(annotators)} annotators (env={args.env}) ===")
        parts = []
        for a in annotators:
            out = a.output_path(paths.annotations, sd.name)
            if out.exists() and not args.force:
                print(f"    [skip] {a.name} v{a.version}: output exists")
            else:
                df = a.annotate(shots, info, sd, device)
                out.parent.mkdir(parents=True, exist_ok=True)
                df.to_parquet(out, index=False)
            parts.append(pd.read_parquet(out))
        fuse(parts, paths.annotations / "fused" / FUSED_VERSION / f"{sd.name}.parquet")


if __name__ == "__main__":
    main()
