"""Stage 2: ingest candidate videos into shards (download, segment, filter, cut clips, delete raw).

    python scripts/run_ingest.py --env test_env              # one shard
    python scripts/run_ingest.py --env test_env --shards 3   # three shards in a row
    python scripts/run_ingest.py --env test_env --max-videos 1
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.config import add_env_arg, load_config  # noqa: E402
from src.ingest.pipeline import ingest_shard  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_env_arg(ap)
    ap.add_argument("--shards", type=int, default=1, help="number of shards to build")
    ap.add_argument("--max-videos", type=int, default=None, help="override videos per shard")
    args = ap.parse_args()
    cfg = load_config(args.env)
    for _ in range(args.shards):
        if ingest_shard(cfg, args.max_videos) is None:
            break


if __name__ == "__main__":
    main()
