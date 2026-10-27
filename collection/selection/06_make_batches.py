#!/usr/bin/env python3
"""Stage 6 - split the downloader list into stratified download batches (no downloads).

Reads 05_final/corpus_manifest_<version>.csv (checked against its .sha256; rows are converted to the
ia_download.py layout here, so an Excel-re-saved download list cannot leak into the batches) and
05_final/human_audit.csv; writes
<output_root>/06_batches/:
  batch_01/batch_01.csv ...       downloader-compatible lists (select = x), one folder per batch so the
                                   manifest.csv that ia_download.py writes next to each list is never overwritten
  batch_NN_audit/...               the videos in the human-audit sample, always in the LAST batches
  batch_NN/corpus_paths.txt        the batch's item folders relative to corpus/ (for tar / copying to the cluster)
  batches_index.csv                identifier -> batch, category, minutes, MB, audit flag
  batches_summary.json             per batch: videos, hours, GB, categories, stations

Videos outside the audit sample go first. Each group is split into the fewest batches of at most --batch-size
videos with near-equal sizes; within a group every primary category is dealt round-robin across the batches in
order of decreasing duration, so each batch has the same category mix (+-1 video) and similar hours. Deterministic: rerunning
after --apply-audit (new version) keeps the non-audit batches identical and only rebuilds the audit batches.

  python collection/selection/06_make_batches.py
  python collection/selection/06_make_batches.py --version v2 --batch-size 50
"""
from __future__ import annotations

import argparse
import math
from collections import Counter, defaultdict

import common as C


_DL = []


def _downloader():
    if _DL:
        return _DL[0]
    import importlib.util
    spec = importlib.util.spec_from_file_location("ia_download", C.REPO / "collection" / "ia_download.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _DL.append(mod)
    return mod


def short_dir(ident):
    return _downloader().short_dir(ident)


def split(rows, size):
    """Stratified assignment to ceil(n/size) batches: categories (largest first), each sorted by decreasing
    duration, are laid end to end and dealt round-robin, so every category and the batch sizes differ by at most
    one video between batches and long videos are spread out."""
    if not rows:
        return []
    nb = math.ceil(len(rows) / size)
    by_cat = defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r)
    seq = []
    for cat in sorted(by_cat, key=lambda c: (-len(by_cat[c]), c)):
        seq += sorted(by_cat[cat], key=lambda r: (-C.to_float(r.get("duration_min"), 0), r["identifier"]))
    batches = [[] for _ in range(nb)]
    for i, r in enumerate(seq):
        batches[i % nb].append(r)
    for bt in batches:
        bt.sort(key=lambda r: r["identifier"])
    return batches


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--out-root", default=None)
    ap.add_argument("--version", default="v1", help="which corpus_manifest_<version>.csv to split")
    ap.add_argument("--batch-size", type=int, default=50)
    ap.add_argument("--audit", default=None, help="audit csv (default: 05_final/human_audit.csv)")
    a = ap.parse_args(argv)
    cfg = C.load_config(a.config)
    d5 = C.stage_dir(cfg, 5, a.out_root)
    man_p = C.require(d5 / f"corpus_manifest_{a.version}.csv", "stage-5 manifest")
    sha_p = d5 / f"corpus_manifest_{a.version}.sha256"
    if sha_p.exists():
        changed = [n for n in C.verify_sha256_file(sha_p) if n.endswith(f"corpus_manifest_{a.version}.csv")]
        if changed:
            raise C.StageError(f"{man_p.name} no longer matches its SHA-256 (was it re-saved, e.g. by Excel?). "
                               "Re-run 05_report_and_freeze.py to regenerate it.")
    fields = C.downloader_fields()
    prof_p = C.stage_dir(cfg, 3, a.out_root) / "metadata_profiles.csv"
    prof = {r["identifier"]: r for r in C.read_csv(prof_p)} if prof_p.exists() else {}
    # manifest values win; columns the manifest does not carry (description, subjects, ...) come from stage 3
    rows = [C.download_row({**prof.get(r["identifier"], {}), **{k: v for k, v in r.items() if v != ""}}, fields)
            for r in C.read_csv(man_p)]
    if len({r["identifier"] for r in rows}) != len(rows):
        raise C.StageError(f"{man_p}: duplicate identifiers")
    dl_p = man_p
    audit_p = C.rpath(a.audit) if a.audit else d5 / "human_audit.csv"
    audit_ids = {r["identifier"] for r in C.read_csv(C.require(audit_p, "human audit csv"))}
    # a reviewer may have replaced audit videos; their replacements are audit-related too
    mj = d5 / f"corpus_manifest_{a.version}.json"
    if mj.exists():
        audit_ids |= {x.get("in") for x in C.read_json(mj).get("audit_changes", []) if x.get("in")}
    main_rows = [r for r in rows if r["identifier"] not in audit_ids]
    aud_rows = [r for r in rows if r["identifier"] in audit_ids]
    groups = split(main_rows, a.batch_size) + [None] + split(aud_rows, a.batch_size)

    out = C.rpath(a.out_root or cfg["paths"]["output_root"]) / "06_batches"
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("batch_*"):  # rebuild cleanly (an old manifest.csv inside a downloaded batch is kept)
        for f in old.glob("batch_*.csv"):
            f.unlink()
    index, summary, n = [], [], 0
    is_audit = False
    for g in groups:
        if g is None:
            is_audit = True
            continue
        n += 1
        name = f"batch_{n:02d}" + ("_audit" if is_audit else "")
        bd = out / name
        bd.mkdir(exist_ok=True)
        C.write_csv(bd / f"{name}.csv", [{**r, "select": "x"} for r in g], fields)
        # item folders under corpus/ (same naming as ia_download.py), for copying one batch to the cluster
        (bd / "corpus_paths.txt").write_text("".join(f"{r['category']}/{short_dir(r['identifier'])}\n" for r in g),
                                            encoding="utf-8")
        h = sum(C.to_float(r.get("duration_min"), 0) for r in g) / 60
        gb = sum(C.to_float(r.get("video_mb"), 0) for r in g) / 1024
        summary.append({"batch": name, "videos": len(g), "audit": is_audit, "hours": round(h, 1), "gb": round(gb, 1),
                        "categories": dict(Counter(r["category"] for r in g)),
                        "stations": len({r.get("creator") or r.get("station") for r in g}),
                        "csv": str((bd / f"{name}.csv").relative_to(C.REPO)) if C.REPO in bd.parents else str(bd / f"{name}.csv")})
        for r in g:
            index.append({"identifier": r["identifier"], "batch": name, "audit_sample": int(is_audit),
                          "category": r["category"], "duration_min": r.get("duration_min"), "video_mb": r.get("video_mb"),
                          "title": r.get("title")})
    ids = [x["identifier"] for x in index]
    if len(ids) != len(rows) or len(set(ids)) != len(rows):
        raise C.StageError("internal error: batches do not cover the list exactly once")
    C.write_csv(out / "batches_index.csv", index)
    C.write_json(out / "batches_summary.json", {"created": C.now_iso(), "source": str(dl_p),
                                                "source_sha256": C.sha256_file(dl_p), "batch_size": a.batch_size,
                                                "audit_csv": str(audit_p), "videos": len(rows),
                                                "audit_videos": len(aud_rows), "batches": summary})
    C.log(f"{len(rows)} videos -> {sum(1 for s in summary if not s['audit'])} batches first + "
          f"{sum(1 for s in summary if s['audit'])} audit batches last")
    for s in summary:
        C.log(f"  {s['batch']:16} {s['videos']:3} videos  {s['hours']:5.1f} h  {s['gb']:5.1f} GB  {s['stations']:3} stations")
    C.log(f"wrote {out}")


if __name__ == "__main__":
    main()
