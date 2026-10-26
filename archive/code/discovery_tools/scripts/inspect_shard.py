"""Look inside a shard: shot table, discard reasons, and optionally extract a few clips/keyframes to view.

    python scripts/inspect_shard.py --env test_env                       # latest shard
    python scripts/inspect_shard.py --env test_env --extract 5           # also unpack 5 clips + keyframes
"""
import argparse
import sys
import tarfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.common import shard_io  # noqa: E402
from src.common.config import add_env_arg, load_config  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_env_arg(ap)
    ap.add_argument("--shard", default=None)
    ap.add_argument("--extract", type=int, default=0, help="unpack N clips + keyframes to data/<env>/inspect/")
    args = ap.parse_args()
    paths = load_config(args.env)["paths"]

    shards = shard_io.list_complete_shards(paths.shards)
    if not shards:
        print("No complete shards.")
        return
    sd = next((s for s in shards if s.name == args.shard), shards[-1]) if args.shard else shards[-1]
    shots, info = shard_io.read_shots(sd, kept_only=False), shard_io.read_info(sd)

    pd.set_option("display.width", 200, "display.max_colwidth", 50)
    print(f"=== {sd.name} ===")
    cols = [c for c in ["id", "title", "duration", "upload_date", "channel", "caption_source"] if c in info]
    print(info[cols].to_string(index=False))
    print(f"\nShots: {len(shots)} total, {int(shots['kept'].sum())} kept")
    print("Discard reasons:", shots.loc[~shots["kept"], "discard_reason"].value_counts().to_dict())
    print("\n", shots[["shot_id", "start_s", "end_s", "kept", "discard_reason", "caption_text"]].head(30).to_string(index=False))

    if args.extract:
        out = paths.root / "inspect" / sd.name
        out.mkdir(parents=True, exist_ok=True)
        for tar_name in ("clips.tar", "keyframes.tar"):
            with tarfile.open(sd / tar_name) as tf:
                for m in tf.getmembers()[: args.extract]:
                    try:
                        tf.extract(m, out, filter="data")
                    except TypeError:  # older Python without tar filters
                        tf.extract(m, out)
        print(f"\nExtracted {args.extract} clips + keyframes to {out}")


if __name__ == "__main__":
    main()
