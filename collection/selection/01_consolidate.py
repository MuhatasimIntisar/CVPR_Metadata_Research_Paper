#!/usr/bin/env python3
"""Stage 1 - consolidation and deduplication of the discovery rounds (metadata only, no network).

Reads the discovery CSVs listed in selection_config.json (never modifies them) and writes to
<output_root>/01_consolidation/:
  candidate_master.csv       one row per unique IA identifier: all original columns + discovery provenance,
                             multi-label category evidence, consolidation_status / exclusion_reason
  duplicate_groups.csv       identifiers found in several rounds (merged) and identical video MD5s (one kept)
  suspected_reuploads.csv    probable re-uploads (normalised title + station + duration); flagged, never removed
  pilot_exclusions.csv       the pilot videos, from the pilot manifest / corpus folders
  consolidation_report.json  counts for the paper

  python collection/selection/01_consolidate.py
  python collection/selection/01_consolidate.py --round search_3=collection/rounds/other.csv
"""
from __future__ import annotations

import argparse
import difflib
from collections import Counter, defaultdict

import common as C

REQUIRED = ["category", "title", "duration_min", "identifier", "video_file"]
NON_CONFLICT = {"select", "category", "matched_terms", "neighbours_1d", "found_via", "weekday"}


def load_rounds(cfg, overrides):
    rounds = [dict(r) for r in cfg["paths"]["rounds"]]
    for o in overrides or []:
        name, _, path = o.partition("=")
        if not path:
            raise C.StageError(f"--round expects name=path, got {o!r}")
        hit = [r for r in rounds if r["name"] == name]
        if hit:
            hit[0]["path"] = path
        else:
            rounds.append({"name": name, "path": path})
    out = []
    for order, r in enumerate(rounds):
        p = C.require(C.rpath(r["path"]), f"discovery CSV for {r['name']}")
        rows = C.read_csv(p)
        if not rows:
            raise C.StageError(f"{p} has no rows")
        C.validate_columns(rows, REQUIRED, str(p))
        for i, row in enumerate(rows):
            row["_round"], row["_round_order"], row["_row"] = r["name"], order, i
        out.append({"name": r["name"], "path": str(p), "sha256": C.sha256_file(p), "rows": rows,
                    "columns": list(rows[0].keys())})
    return out


def pilot_ids(cfg):
    """identifier -> (title, source) from pilot manifests (csv with identifier) and corpus folders (files.json)."""
    found = {}
    for src in cfg["paths"].get("pilot_sources", []):
        p = C.rpath(src)
        if p.is_file() and p.suffix.lower() == ".csv":
            for r in C.read_csv(p):
                if r.get("identifier"):
                    found.setdefault(r["identifier"].strip(), (r.get("title", ""), str(src)))
        elif p.is_dir():
            for fj in sorted(p.rglob("files.json")):
                try:
                    d = C.read_json(fj)
                except Exception:  # noqa: BLE001
                    continue
                if d.get("identifier"):
                    found.setdefault(d["identifier"].strip(), ("", str(fj.relative_to(C.REPO)) if C.REPO in fj.parents else str(fj)))
        elif not p.exists():
            C.log(f"  note: pilot source {src} not found (skipped)")
    return found


def completeness(r):
    return sum(1 for k, v in r.items() if not k.startswith("_") and str(v).strip())


def merge_records(rounds):
    by_id = defaultdict(list)
    for rd in rounds:
        for r in rd["rows"]:
            ident = (r.get("identifier") or "").strip()
            if ident:
                by_id[ident].append(r)
    fields = list(dict.fromkeys(k for rd in rounds for k in rd["columns"]))
    merged, dup_groups = {}, []
    for ident, recs in by_id.items():
        recs.sort(key=lambda r: (r["_round_order"], r["_row"]))
        base = max(recs, key=lambda r: (completeness(r), -r["_round_order"]))
        m = {k: base.get(k, "") for k in fields}
        m["discovery_rounds"] = list(dict.fromkeys(r["_round"] for r in recs))
        m["n_discovery_records"] = len(recs)
        m["discovery_categories"] = list(dict.fromkeys(r["category"] for r in recs if r.get("category")))
        m["discovery_matched_terms"] = list(dict.fromkeys(t for r in recs for t in C.split_list(r.get("matched_terms"))))
        m["primary_category"] = recs[0].get("category", "")
        dt_ = defaultdict(list)
        for r in recs:
            dt_[r.get("category", "")] += C.split_list(r.get("matched_terms"))
        m["_discovery_terms"] = {k: list(dict.fromkeys(v)) for k, v in dt_.items()}
        conflicts = [k for k in fields if k not in NON_CONFLICT and len({str(r.get(k, "")).strip() for r in recs
                                                                        if str(r.get(k, "")).strip()}) > 1]
        m["field_conflicts"] = conflicts
        m["select"] = ""
        merged[ident] = m
        if len(recs) > 1:
            dup_groups.append({"duplicate_type": "same_identifier", "identifier": ident, "role": "merged",
                               "n_records": len(recs), "rounds": m["discovery_rounds"], "video_md5": m.get("video_md5", ""),
                               "title": m.get("title", ""), "evidence": f"identifier found {len(recs)} times"})
    return merged, dup_groups, fields


def add_categories(m, compiled):
    ev = C.match_categories(m.get("title", ""), m.get("subjects", ""), compiled)
    rem = list(dict.fromkeys(e["category"] for e in ev))
    disc = [c for c in m["discovery_categories"] if c in compiled]
    for c in disc:  # the discovery search itself is evidence (IA's own query matched, incl. its stemming)
        for t in (m["_discovery_terms"].get(c) or ["(search query)"]):
            ev.append({"category": c, "term": t, "field": "discovery"})
    cats = [c for c in compiled if c in rem or c in disc]
    m["categories"] = cats
    m["n_categories"] = len(cats)
    m["category_evidence"] = C.evidence_str(ev)
    m["rematch_added_categories"] = [c for c in rem if c not in disc]
    m["discovery_not_rematched"] = [c for c in disc if c not in rem]
    for c in compiled:
        m[f"cat_{c}"] = int(c in cats)


def md5_duplicates(merged):
    groups = defaultdict(list)
    for ident, m in merged.items():
        if m["consolidation_status"] == "kept" and (m.get("video_md5") or "").strip():
            groups[m["video_md5"].strip().lower()].append(ident)
    rows, n = [], 0
    for md5, ids in sorted(groups.items()):
        if len(ids) < 2:
            continue
        n += 1

        def rank(i):
            m = merged[i]
            return (0 if m.get("has_captions") == "yes" else 1, -len(m["discovery_rounds"]),
                    C.full_date(m.get("date")) or C.full_date("9999-12-31"), i)
        keep, *drop = sorted(ids, key=rank)
        for i in [keep] + drop:
            m = merged[i]
            if i != keep:
                m["consolidation_status"] = "excluded"
                m["exclusion_reason"] = f"exact_md5_duplicate_of:{keep}"
            rows.append({"duplicate_type": "same_md5", "group_id": f"md5_{n:04d}", "identifier": i,
                         "role": "kept" if i == keep else "excluded", "n_records": m["n_discovery_records"],
                         "rounds": m["discovery_rounds"], "video_md5": md5, "title": m.get("title", ""),
                         "evidence": "identical IA video MD5; kept by captions > rounds > date > identifier"})
    return rows, n


def suspected_reuploads(merged, rc):
    """Same channel + near-identical normalised title + duration within tolerance, and no differing episode /
    date tokens (different episodes are never duplicates). Cross-channel: identical normalised title + duration."""
    items = [(i, m) for i, m in merged.items() if m["consolidation_status"] == "kept"]
    dur = {i: (C.to_float(m.get("duration_min")) or 0) * 60 for i, m in items}
    tnorm = {i: C.fold(m.get("title")) for i, m in items}
    epi = {i: C.episode_tokens(m.get("title")) for i, m in items}
    tol = lambda a, b: max(rc["duration_tolerance_s"], rc["duration_tolerance_frac"] * max(dur[a], dur[b]))  # noqa: E731
    pairs = {}
    by_ch = defaultdict(list)
    for i, m in items:
        by_ch[C.channel_key(m.get("creator"), m.get("station"))].append(i)
    for ch, ids in by_ch.items():
        ids.sort(key=lambda i: dur[i])
        for x in range(len(ids)):
            for y in range(x + 1, len(ids)):
                a, b = ids[x], ids[y]
                if dur[b] - dur[a] > tol(a, b):
                    break
                if epi[a] != epi[b] and epi[a] and epi[b]:
                    continue
                sim = difflib.SequenceMatcher(None, tnorm[a], tnorm[b]).ratio()
                if sim >= rc["title_similarity"]:
                    pairs[tuple(sorted((a, b)))] = (sim, True)
    if rc.get("cross_station", True):
        by_title = defaultdict(list)
        for i, _ in items:
            if len(tnorm[i]) >= rc.get("cross_station_min_title_chars", 12):
                by_title[tnorm[i]].append(i)
        for ids in by_title.values():
            for x in range(len(ids)):
                for y in range(x + 1, len(ids)):
                    a, b = sorted((ids[x], ids[y]))
                    if (a, b) not in pairs and abs(dur[a] - dur[b]) <= tol(a, b):
                        pairs[(a, b)] = (1.0, False)
    rows = []
    for (a, b), (sim, same_ch) in sorted(pairs.items()):
        ma, mb = merged[a], merged[b]
        dd = abs(dur[a] - dur[b])
        conf = "high" if same_ch and sim >= 0.98 and dd <= 10 else "medium"
        rows.append({"identifier_a": a, "identifier_b": b, "channel_a": ma.get("creator") or ma.get("station"),
                     "channel_b": mb.get("creator") or mb.get("station"), "title_a": ma.get("title"),
                     "title_b": mb.get("title"), "duration_min_a": ma.get("duration_min"),
                     "duration_min_b": mb.get("duration_min"), "duration_diff_s": round(dd, 1),
                     "title_similarity": round(sim, 4), "same_channel": int(same_ch), "date_a": ma.get("date"),
                     "date_b": mb.get("date"), "confidence": conf,
                     "evidence": ("same channel" if same_ch else "different channels, identical normalised title")
                                 + f"; title similarity {sim:.3f}; duration difference {dd:.0f} s; no differing episode/date tokens"})
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None, help="selection_config.json (default: next to this script)")
    ap.add_argument("--out-root", default=None, help="override paths.output_root")
    ap.add_argument("--round", action="append", help="name=path: replace or add a discovery CSV")
    ap.add_argument("--categories-config", default=None, help="override paths.categories_config")
    a = ap.parse_args(argv)
    cfg = C.load_config(a.config)
    out = C.stage_dir(cfg, 1, a.out_root)
    cat_path = C.require(C.rpath(a.categories_config or cfg["paths"]["categories_config"]), "categories config")
    compiled = C.compile_categories(C.read_json(cat_path))
    missing = [c for c in C.CATEGORIES if c not in compiled]
    if missing:
        C.log(f"  note: categories not in {cat_path.name}: {missing}")

    rounds = load_rounds(cfg, a.round)
    C.log("rounds: " + ", ".join(f"{r['name']}={len(r['rows'])}" for r in rounds))
    merged, dup_rows, fields = merge_records(rounds)
    for m in merged.values():
        m["consolidation_status"], m["exclusion_reason"] = "kept", ""
        m["channel_key"] = C.channel_key(m.get("creator"), m.get("station"))
        add_categories(m, compiled)

    pilots = pilot_ids(cfg)
    pilot_rows = []
    for ident, (title, src) in sorted(pilots.items()):
        hit = ident in merged
        if hit:
            merged[ident]["consolidation_status"] = "excluded"
            merged[ident]["exclusion_reason"] = "pilot_video"
        pilot_rows.append({"identifier": ident, "title": title or (merged[ident].get("title") if hit else ""),
                           "source": src, "found_in_candidates": int(hit),
                           "reason": "pilot video (excluded from the main collection)"})
    md5_rows, n_md5 = md5_duplicates(merged)
    reup = suspected_reuploads(merged, cfg["consolidation"]["reupload"])
    flagged = Counter()
    for r in reup:
        flagged[r["identifier_a"]] += 1
        flagged[r["identifier_b"]] += 1
    for i, m in merged.items():
        m["suspected_reupload_pairs"] = flagged.get(i, 0)

    extra = ["primary_category", "categories", "n_categories", "category_evidence", "rematch_added_categories",
             "discovery_not_rematched"] + [f"cat_{c}" for c in compiled] + [
             "channel_key", "discovery_rounds", "n_discovery_records", "discovery_categories",
             "discovery_matched_terms", "field_conflicts", "suspected_reupload_pairs", "consolidation_status",
             "exclusion_reason"]
    master_fields = fields + [f for f in extra if f not in fields]
    rows = [merged[i] for i in sorted(merged)]
    C.write_csv(out / "candidate_master.csv", rows, master_fields)
    C.write_csv(out / "duplicate_groups.csv", dup_rows + md5_rows,
                ["duplicate_type", "group_id", "identifier", "role", "n_records", "rounds", "video_md5", "title", "evidence"])
    C.write_csv(out / "suspected_reuploads.csv", reup,
                ["identifier_a", "identifier_b", "channel_a", "channel_b", "title_a", "title_b", "duration_min_a",
                 "duration_min_b", "duration_diff_s", "title_similarity", "same_channel", "date_a", "date_b",
                 "confidence", "evidence"])
    C.write_csv(out / "pilot_exclusions.csv", pilot_rows, ["identifier", "title", "source", "found_in_candidates", "reason"])

    kept = [m for m in rows if m["consolidation_status"] == "kept"]
    hours = lambda rs: round(sum(C.to_float(r.get("duration_min"), 0) for r in rs) / 60, 1)  # noqa: E731
    gb = lambda rs: round(sum(C.to_float(r.get("video_mb"), 0) for r in rs) / 1024, 1)  # noqa: E731
    report = {
        "stage": "01_consolidation", "created": C.now_iso(), "config": cfg["_path"], "categories_config": str(cat_path),
        "categories_config_sha256": C.sha256_file(cat_path), "code": C.code_version(),
        "inputs": [{"name": r["name"], "path": r["path"], "sha256": r["sha256"], "records": len(r["rows"]),
                    "unique_identifiers": len({x["identifier"] for x in r["rows"]}),
                    "discovery_category_counts": dict(Counter(x["category"] for x in r["rows"]))} for r in rounds],
        "records_total": sum(len(r["rows"]) for r in rounds),
        "unique_identifiers": len(merged),
        "duplicate_records_merged": sum(len(r["rows"]) for r in rounds) - len(merged),
        "identifiers_in_multiple_rounds": len(dup_rows),
        "exact_md5_duplicate_groups": n_md5,
        "exact_md5_duplicates_excluded": sum(1 for r in md5_rows if r["role"] == "excluded"),
        "pilot_identifiers_known": len(pilots),
        "pilot_videos_excluded": sum(r["found_in_candidates"] for r in pilot_rows),
        "suspected_reupload_pairs": len(reup),
        "suspected_reupload_pairs_by_confidence": dict(Counter(r["confidence"] for r in reup)),
        "videos_in_suspected_reupload_pairs": len(flagged),
        "kept": len(kept), "excluded": len(rows) - len(kept),
        "exclusion_reasons": dict(Counter(m["exclusion_reason"].split(":")[0] for m in rows if m["exclusion_reason"])),
        "kept_hours": hours(kept), "kept_reported_gb": gb(kept),
        "categories_kept": {
            "discovery_label_counts": dict(Counter(m["primary_category"] for m in kept)),
            "multilabel_counts": {c: sum(m[f"cat_{c}"] for m in kept) for c in compiled},
            "labels_per_video": dict(sorted(Counter(m["n_categories"] for m in kept).items())),
            "videos_with_rematch_added_labels": sum(1 for m in kept if m["rematch_added_categories"]),
            "discovery_labels_not_reproduced_by_rematch": sum(1 for m in kept if m["discovery_not_rematched"]),
            "note": "category labels are keyword evidence (title / IA subject tags / discovery query), not verified content"},
        "field_conflicts": dict(Counter(f for m in rows for f in m["field_conflicts"])),
    }
    C.write_json(out / "consolidation_report.json", report)
    C.log(f"{report['records_total']} records -> {report['unique_identifiers']} unique; {report['kept']} kept "
          f"({report['kept_hours']} h, {report['kept_reported_gb']} GB); excluded {report['exclusion_reasons']}; "
          f"{len(reup)} suspected re-upload pairs flagged")
    C.log(f"wrote {out}")


if __name__ == "__main__":
    main()
