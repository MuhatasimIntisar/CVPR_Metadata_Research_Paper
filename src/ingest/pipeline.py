"""(2) Ingest one shard: download -> segment -> keyframes -> filters -> clips -> captions -> write shard.

The manifest is only updated after the shard is fully written, so a crash mid-shard simply
re-processes those videos on the next run. Raw downloads are deleted once the shard is written.
"""
from __future__ import annotations

import shutil
import time

import pandas as pd

from src.common import manifest as mf
from src.common import shard_io
from src.ingest import captions, clips, filters, keyframes, segment
from src.ingest.download import download

SHOT_COLUMNS = ["shot_id", "video_id", "shot_idx", "start_s", "end_s", "duration_s", "kept",
                "discard_reason", "phash", "caption_text"]


def ingest_shard(cfg: dict, max_videos: int | None = None) -> str | None:
    paths, ic = cfg["paths"], cfg["ingest"]
    df = mf.load(paths.manifest)
    pending = df[df["status"] == "candidate"].head(max_videos or ic["videos_per_shard"])
    if pending.empty:
        print("No candidate videos left. Run discovery first.")
        return None

    shard_id = mf.next_shard_id(paths.shards)
    work = paths.tmp / shard_id
    clip_dir, kf_dir = work / "clips", work / "keyframes"
    clip_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== Ingest {shard_id}: {len(pending)} videos  (env={cfg['env_name']}) ===")

    shot_rows, info_rows, done, failed = [], [], [], []
    for i, row in enumerate(pending.itertuples(), 1):
        vid = row.video_id
        print(f"[{i}/{len(pending)}] {vid}  {str(row.title)[:60]}")
        t0 = time.time()
        try:
            print("      downloading ...")
            d = download(vid, work / "raw" / vid, ic["max_height"])
            dur = float(d["info"].get("duration") or row.duration_s)
            sc = cfg["segmentation"]
            shots = segment.segment(d["video"], dur, sc["threshold"], sc["min_shot_s"], sc["max_shot_s"])
            print(f"      segmented: {len(shots)} shots")
            kc = cfg["keyframes"]
            keyframes.extract(d["video"], shots, kf_dir, vid, kc["max_side"], kc["jpeg_quality"])
            filters.apply_all(shots, cfg)
            captions.attach(shots, d["captions"])
            kept = [s for s in shots if s["kept"]]
            for s in kept:
                clips.cut(d["video"], s, clip_dir / f"{vid}__{s['shot_idx']:04d}.mp4",
                          ic["max_height"], ic["keep_audio"])
            for s in shots:
                s["video_id"], s["shot_id"] = vid, f"{vid}__{s['shot_idx']:04d}"
                shot_rows.append({k: s.get(k) for k in SHOT_COLUMNS})
            info_rows.append({**d["info"], "tags": ",".join(d["info"].get("tags") or []),
                              "categories": ",".join(d["info"].get("categories") or []),
                              "query": row.query})
            done.append((vid, len(shots), len(kept)))
            reasons = pd.Series([s["discard_reason"] for s in shots if not s["kept"]]).value_counts().to_dict()
            print(f"      kept {len(kept)}/{len(shots)} shots  discarded={reasons}  "
                  f"captions={d['info']['caption_source']}  ({time.time() - t0:.0f}s)")
        except Exception as e:  # noqa: BLE001
            print(f"      FAILED: {type(e).__name__}: {e}")
            failed.append((vid, f"{type(e).__name__}: {e}"[:500]))
        if ic["delete_raw"]:
            shutil.rmtree(work / "raw" / vid, ignore_errors=True)
        time.sleep(ic["sleep_between_downloads_s"])

    if not done:
        print("No videos ingested successfully; shard not written.")
    else:
        out = paths.shard_dir(shard_id)
        out.mkdir(parents=True, exist_ok=True)
        shard_io.write_tar(out / "clips.tar", sorted(clip_dir.glob("*.mp4")))
        shard_io.write_tar(out / "keyframes.tar", sorted(kf_dir.glob("*.jpg")))
        pd.DataFrame(shot_rows, columns=SHOT_COLUMNS).to_parquet(out / "shots.parquet", index=False)
        pd.DataFrame(info_rows).to_parquet(out / "info.parquet", index=False)
        shard_io.mark_complete(out)

    for vid, n, k in done:
        df = mf.update(df, vid, status="ingested", shard_id=shard_id, n_shots=n, n_kept=k, error=None)
    for vid, err in failed:
        df = mf.update(df, vid, status="failed", error=err)
    mf.save(df, paths.manifest)
    shutil.rmtree(work, ignore_errors=True)

    if done:
        _summary(paths.shard_dir(shard_id), done, failed)
    return shard_id if done else None


def _summary(out, done, failed) -> None:
    size = sum(f.stat().st_size for f in out.iterdir()) / 1e6
    n_shots, n_kept = sum(d[1] for d in done), sum(d[2] for d in done)
    print(f"\n--- {out.name} written: {len(done)} videos, {n_kept}/{n_shots} shots kept, "
          f"{len(failed)} failed, {size:.1f} MB on disk ---")
    for f in sorted(out.iterdir()):
        print(f"    {f.name:<16} {f.stat().st_size / 1e6:8.2f} MB")
