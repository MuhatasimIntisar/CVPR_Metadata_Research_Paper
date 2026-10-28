#!/usr/bin/env python3
"""Print the video_ids of shard INDEX of COUNT (space-separated), for SLURM job arrays.

Valid manifest videos are sorted by video file size (a proxy for duration) and dealt greedily to the shard with the
least total so far, so shards finish at about the same time. Deterministic: every array task computes the same split.

  python scripts/shard_videos.py --config configs/cluster.yaml --index 0 --count 4
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from video_metadata.config import load_config  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", nargs="+", default=["configs/cluster.yaml"])
    ap.add_argument("--index", type=int, required=True)
    ap.add_argument("--count", type=int, required=True)
    a = ap.parse_args(argv)
    if not 0 <= a.index < a.count:
        sys.exit(f"--index must be in [0, {a.count})")
    cfg = load_config(*a.config)
    man = json.loads(Path(cfg["paths"]["manifest"]).read_text(encoding="utf-8"))
    root = Path(man.get("corpus_root") or cfg["paths"]["corpus_root"])
    vids = []
    for v in man["videos"]:
        if v.get("status") != "ok":
            continue
        f = root / v["item_dir"] / (v.get("files") or {}).get("video", "video.mp4")
        vids.append((f.stat().st_size if f.exists() else 0, v["video_id"]))
    vids.sort(key=lambda x: (-x[0], x[1]))
    load = [0] * a.count
    shards = [[] for _ in range(a.count)]
    for size, vid in vids:
        i = min(range(a.count), key=lambda k: (load[k], len(shards[k]), k))
        load[i] += size
        shards[i].append(vid)
    print(" ".join(sorted(shards[a.index])))


if __name__ == "__main__":
    main()
