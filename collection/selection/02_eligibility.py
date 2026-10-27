#!/usr/bin/env python3
"""Stage 2 - technical eligibility from metadata (offline by default; optional live IA availability check).

Reads 01_consolidation/candidate_master.csv, writes <output_root>/02_eligibility/:
  eligible_candidates.csv    every consolidated candidate with eligibility_status (eligible | conditional |
                             excluded), eligibility_reasons (why excluded) and eligibility_flags (caveats)
  eligibility_report.json    counts, MP4 / caption / duration / size / resolution distributions, hours and GB
  availability_cache.json    only when --check-availability is used (reused on later runs)

  python collection/selection/02_eligibility.py                                  # offline, collected metadata only
  python collection/selection/02_eligibility.py --check-availability --ids-from collection/selection/outputs/04_selection/selected_1000.csv
  python collection/selection/02_eligibility.py --check-availability --limit 50  # a sample
Network checks are optional and never required for the initial selection. Files are not downloaded; integrity is
verified later by collection/ia_download.py when it downloads (MD5).
"""
from __future__ import annotations

import argparse
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import common as C

IDENT_RX = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{1,99}$")
META = "https://archive.org/metadata/"
UA = "tvm-selection/1.0 (academic research dataset; metadata availability check)"


def check_row(m, ec, cache):
    """-> (status, reasons, flags, details). reasons = why excluded; flags = caveats that do not exclude."""
    reasons, flags, det = [], [], {}
    ident = (m.get("identifier") or "").strip()
    if not IDENT_RX.match(ident):
        reasons.append("invalid_identifier")
    vf = (m.get("video_file") or "").strip()
    ext = ("." + vf.rsplit(".", 1)[-1].lower()) if "." in vf else ""
    if not vf:
        reasons.append("no_video_file")
    elif ext not in ec["allowed_extensions"]:
        flags.append(f"non_preferred_format:{ext or 'none'}")
    dur = C.to_float(m.get("duration_min"))
    if dur is None or dur <= 0:
        reasons.append("invalid_duration")
    elif dur < ec["min_minutes"] or dur > ec["max_minutes"]:
        reasons.append(f"duration_outside_{ec['min_minutes']:g}-{ec['max_minutes']:g}_min")
    mb = C.to_float(m.get("video_mb"))
    if mb is None or mb <= 0:
        flags.append("unknown_size")
    elif mb > ec["hard_max_mb"]:
        reasons.append(f"size_over_{ec['hard_max_mb']}_mb")
    elif mb > ec["soft_max_mb"]:
        flags.append(f"size_over_{ec['soft_max_mb']}_mb")
    if m.get("has_captions") != "yes" or not (m.get("captions") or "").strip():
        (reasons if ec.get("captions_required") else flags).append("no_captions")
    if not (m.get("title") or "").strip():
        reasons.append("no_title")
    if not (m.get("details_url") or "").strip():
        flags.append("no_source_url")
    if not (m.get("creator") or "").strip():
        flags.append("no_creator")
    if not C.full_date(m.get("date")):
        flags.append("no_exact_date" if C.year_of(m.get("date")) else "no_date")
    if m.get("date_source") == "uploaded":
        flags.append("date_is_upload_date")
    if not (m.get("description") or "").strip():
        flags.append("no_description")
    if not (m.get("video_md5") or "").strip():
        flags.append("no_checksum")
    if int(m.get("suspected_reupload_pairs") or 0):
        flags.append("suspected_reupload")
    flags.append("no_rights_field")  # the discovery CSVs carry no licence / rights field
    live = cache.get(ident)
    det["resolution"] = ""
    if live:
        det["availability_checked"] = live.get("checked")
        if not live.get("ok"):
            reasons.append(f"unavailable:{live.get('reason')}")
        else:
            if live.get("video_present") is False:
                reasons.append("video_file_missing_at_check")
            if live.get("size_matches") is False:
                flags.append("size_changed_since_discovery")
            if live.get("md5_matches") is False:
                flags.append("md5_changed_since_discovery")
            if live.get("width") and live.get("height"):
                det["resolution"] = f"{live['width']}x{live['height']}"
    status = "excluded" if reasons else ("conditional" if [f for f in flags if f != "no_rights_field"] else "eligible")
    return status, reasons, flags, det


# ------------------------------------------------------------------ optional live check
def fetch_meta(ident, ac):
    url = META + urllib.parse.quote(ident)
    err = None
    for attempt in range(ac["retries"] + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=ac["timeout_s"]) as r:
                return json.loads(r.read()), None
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, ConnectionError, OSError) as e:
            err = f"{type(e).__name__}: {e}"
            time.sleep(1.5 * (attempt + 1))
    return None, err


def live_check(m, ac):
    time.sleep(ac.get("delay_s", 0))
    item, err = fetch_meta(m["identifier"], ac)
    res = {"checked": C.now_iso()}
    if item is None:
        return {**res, "ok": False, "reason": f"request_failed ({err})"[:200]}
    if not item or "metadata" not in item:
        return {**res, "ok": False, "reason": "no_metadata (item missing or dark)"}
    if str(item["metadata"].get("access-restricted-item", "")).lower() == "true" or item.get("is_dark"):
        return {**res, "ok": False, "reason": "restricted"}
    files = {f.get("name"): f for f in item.get("files", [])}
    v = files.get(m.get("video_file"))
    res.update(ok=True, video_present=v is not None,
               captions_present=any(n and n.lower().endswith((".srt", ".vtt")) for n in files))
    if v:
        mb = C.to_float(m.get("video_mb"))
        if v.get("size") and mb:
            res["size_matches"] = abs(int(v["size"]) / 1048576 - mb) < 0.2
        if v.get("md5") and m.get("video_md5"):
            res["md5_matches"] = v["md5"] == m["video_md5"]
        res["width"], res["height"] = v.get("width"), v.get("height")
    return res


def run_checks(rows, ac, cache, cache_path, force):
    todo = [m for m in rows if force or m["identifier"] not in cache]
    C.log(f"live availability check: {len(todo)} items ({len(rows) - len(todo)} cached), concurrency {ac['concurrency']}")
    done = 0
    with ThreadPoolExecutor(max_workers=max(1, int(ac["concurrency"]))) as ex:
        futs = {ex.submit(live_check, m, ac): m["identifier"] for m in todo}
        for f in as_completed(futs):
            cache[futs[f]] = f.result()
            done += 1
            if done % 25 == 0 or done == len(todo):
                C.write_json(cache_path, cache)  # incremental: an interrupted check keeps its progress
                C.log(f"  {done}/{len(todo)}")
    return cache


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--out-root", default=None)
    ap.add_argument("--check-availability", action="store_true", help="query archive.org metadata (optional)")
    ap.add_argument("--ids-from", help="csv with an identifier column: only check these (e.g. selected_1000.csv)")
    ap.add_argument("--limit", type=int, help="check at most N items")
    ap.add_argument("--force", action="store_true", help="re-check items already in the cache")
    a = ap.parse_args(argv)
    cfg = C.load_config(a.config)
    ec, ac = cfg["eligibility"], cfg["availability"]
    src = C.require(C.stage_dir(cfg, 1, a.out_root) / "candidate_master.csv", "stage-1 output")
    out = C.stage_dir(cfg, 2, a.out_root)
    rows = C.read_csv(src)
    C.validate_columns(rows, ["identifier", "title", "duration_min", "video_file", "consolidation_status"], str(src))
    cache_path = out / "availability_cache.json"
    cache = C.read_json(cache_path) if cache_path.exists() else {}
    if a.check_availability:
        pool = [m for m in rows if m["consolidation_status"] == "kept"]
        if a.ids_from:
            want = {r["identifier"] for r in C.read_csv(C.require(C.rpath(a.ids_from), "--ids-from csv"))}
            pool = [m for m in pool if m["identifier"] in want]
        if a.limit:
            pool = pool[:a.limit]
        cache = run_checks(pool, ac, cache, cache_path, a.force)
        C.write_json(cache_path, cache)

    for m in rows:
        if m["consolidation_status"] != "kept":
            m.update(eligibility_status="excluded", eligibility_reasons=f"consolidation:{m.get('exclusion_reason')}",
                     eligibility_flags="", resolution="", availability_checked="")
            continue
        st, reasons, flags, det = check_row(m, ec, cache)
        m.update(eligibility_status=st, eligibility_reasons=reasons, eligibility_flags=flags,
                 resolution=det.get("resolution", ""), availability_checked=det.get("availability_checked", ""))
    fields = list(rows[0].keys())
    C.write_csv(out / "eligible_candidates.csv", rows, fields)

    live = [m for m in rows if m["consolidation_status"] == "kept"]
    usable = [m for m in live if m["eligibility_status"] != "excluded"]
    mins = sorted(C.to_float(m["duration_min"], 0) for m in usable)
    bands = Counter(C.duration_band(x, cfg["profiling"]["duration_bands"]) for x in mins)
    report = {
        "stage": "02_eligibility", "created": C.now_iso(), "input": str(src), "input_sha256": C.sha256_file(src),
        "consolidated_candidates": len(live),
        "status_counts": dict(Counter(m["eligibility_status"] for m in live)),
        "exclusion_reasons": dict(Counter(r for m in live for r in C.split_list(m["eligibility_reasons"]))),
        "flags": dict(Counter(f for m in live for f in C.split_list(m["eligibility_flags"]))),
        "missing_mp4": sum(1 for m in live if not (m.get("video_file") or "").lower().endswith(".mp4")),
        "captions": {"with": sum(m.get("has_captions") == "yes" for m in usable), "usable_total": len(usable)},
        "duration_min_usable": {"min": mins[0] if mins else None, "median": mins[len(mins) // 2] if mins else None,
                                "max": mins[-1] if mins else None, "bands": dict(bands)},
        "resolution": dict(Counter(m["resolution"] or "unknown (not in discovery metadata)" for m in usable)),
        "usable_hours": round(sum(mins) / 60, 1),
        "usable_reported_gb": round(sum(C.to_float(m.get("video_mb"), 0) for m in usable) / 1024, 1),
        "availability": {"checked": sum(1 for m in live if m["availability_checked"]), "of": len(live),
                         "cache": str(cache_path) if cache else None,
                         "note": "statuses use collected discovery metadata; live checks are optional"},
        "rules": ec,
    }
    C.write_json(out / "eligibility_report.json", report)
    C.log(f"{len(live)} candidates: {report['status_counts']}; usable {len(usable)} = {report['usable_hours']} h, "
          f"{report['usable_reported_gb']} GB; exclusions {report['exclusion_reasons']}")
    C.log(f"wrote {out}")


if __name__ == "__main__":
    main()
