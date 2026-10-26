"""Stage 1: find candidate videos with randomised, coverage-balanced search queries.

Loops in batches until --n-videos new candidates are added (times the oversample factor, since
some will fail ingest). Each batch: generate queries weighted toward under-filled combos ->
search -> apply caps (per query, per channel, per combo) -> append to manifest + query log.
Safe to stop and re-run: coverage is recomputed from the query log every batch.

    python scripts/run_discovery.py --env test_env --n-videos 10
    python scripts/run_discovery.py --env storage_env --n-videos 1000
    python scripts/run_discovery.py --env test_env --report          # coverage only, no search
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.common import manifest as mf  # noqa: E402
from src.common.config import add_env_arg, load_config  # noqa: E402
from src.discovery.query_generator import (LOG_COLUMNS, QueryGenerator, channel_counts,  # noqa: E402
                                           coverage_counts)
from src.discovery.search import search  # noqa: E402


def load_log(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=LOG_COLUMNS)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_env_arg(ap)
    ap.add_argument("--n-videos", type=int, default=None, help="target number of new candidates")
    ap.add_argument("--report", action="store_true", help="print coverage and exit")
    args = ap.parse_args()
    cfg = load_config(args.env)
    dc, paths = cfg["discovery"], cfg["paths"]
    log_path = paths.root / "discovery" / "query_log.parquet"

    gen = QueryGenerator(cfg["search_pools"], dc["max_per_combo"], dc.get("alpha", 1.0))
    log, man = load_log(log_path), mf.load(paths.manifest)
    print(f"=== Discovery (env={args.env}, data_root={paths.root}) ===")
    print("Coverage before:", gen.coverage_report(coverage_counts(log, man)))
    if args.report:
        return

    target = int((args.n_videos or dc["n_videos"]) * dc["oversample"])
    added_total, batch_idx = 0, int(log["batch_id"].max()) + 1 if len(log) else 0
    for _ in range(dc["max_batches"]):
        if added_total >= target:
            break
        counts, chans = coverage_counts(log, man), channel_counts(log)
        seed = dc["seed"] * 100_000 + batch_idx
        queries = gen.generate(dc["queries_per_batch"], counts, seed)
        if not queries:
            print("All combos at ceiling; stopping.")
            break

        rows, new_cands, batch_combo = [], [], {}
        known = set(man["video_id"]) | set(log["video_id"])
        for q in queries:
            try:
                found = search(q.text, dc["results_per_query"], dc["min_duration_s"], dc["max_duration_s"])
            except Exception as e:  # noqa: BLE001
                print(f"  ! search failed '{q.text}': {type(e).__name__}")
                continue
            n_added = 0
            for r in found:
                ch = r.get("channel") or "?"
                combo_n = counts.get(q.combo, 0) + batch_combo.get(q.combo, 0)
                ok = (r["video_id"] not in known and chans.get(ch, 0) < dc["max_per_channel"]
                      and combo_n < dc["max_per_combo"] and added_total + len(new_cands) < target)
                rows.append({"video_id": r["video_id"], "batch_id": batch_idx, "seed": seed, "query": q.text,
                             "combo": q.combo, "channel": ch, "added": ok})
                if ok:
                    known.add(r["video_id"])
                    chans[ch] = chans.get(ch, 0) + 1
                    batch_combo[q.combo] = batch_combo.get(q.combo, 0) + 1
                    new_cands.append({k: r[k] for k in ("video_id", "query", "title", "duration_s")})
                    n_added += 1
            print(f"  [b{batch_idx}] {q.combo:<32} '{q.text}': {len(found)} found, {n_added} added")
            time.sleep(dc.get("sleep_between_searches_s", 1))

        log = pd.concat([log, pd.DataFrame(rows, columns=LOG_COLUMNS)], ignore_index=True)
        man, n_new = mf.add_candidates(man, new_cands)
        added_total += n_new
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log.to_parquet(log_path, index=False)  # saved every batch -> safe to interrupt
        mf.save(man, paths.manifest)
        print(f"  batch {batch_idx}: +{n_new} candidates ({added_total}/{target})")
        batch_idx += 1

    print("\nCoverage after: ", gen.coverage_report(coverage_counts(log, man)))
    print(f"Manifest: {len(man)} videos {man['status'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
