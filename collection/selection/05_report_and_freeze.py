#!/usr/bin/env python3
"""Stage 5 - audit sample, final report and versioned (provisionally) frozen manifest. Never downloads anything.

Reads stages 1-4 and writes <output_root>/05_final/:
  human_audit.csv               stratified review sample with empty reviewer columns
  corpus_manifest_v1.csv        research manifest: exactly N identifiers, sorted by identifier
  corpus_manifest_v1.json       same videos + provenance (seed, config, input hashes, code version, freeze status)
  corpus_manifest_v1.sha256     SHA-256 of the manifest CSV (and of the downloader list)
  download/download_list_v1.csv downloader-compatible list (select = x, ia_download.py columns)
  reserve_manifest.csv          ranked replacements
  final_selection_report.json   the collection-selection report for the paper
  plots/                        optional static figures (matplotlib; skipped if unavailable)

  python collection/selection/05_report_and_freeze.py
  python collection/selection/05_report_and_freeze.py --apply-audit collection/selection/outputs/05_final/human_audit.csv --version v2
Status is 'provisionally_frozen' until --audit-complete is given AND every selected video has a successful live
availability check (02_eligibility.py --check-availability --ids-from ...); then 'frozen'.
"""
from __future__ import annotations

import argparse
import random
from collections import Counter, defaultdict

import common as C
import metrics as M

AUDIT_COLS = ["reviewer_decision", "corrected_categories", "exclusion_reason", "replacement_identifier",
              "replacement_reason", "reviewer_notes"]
MANIFEST_COLS = ["identifier", "title", "details_url", "station_id", "station_name", "creator", "date", "year",
                 "quarter", "duration_min", "duration_band", "video_mb", "has_captions", "language", "primary_category",
                 "categories", "category_evidence", "series_id", "series_confidence", "topic_cluster", "topic_terms",
                 "country", "region", "region_source", "discovery_rounds", "eligibility_status", "eligibility_flags",
                 "resolution", "availability_checked", "selection_rank", "selection_phase", "selection_gain_total",
                 "video_file", "video_md5", "captions"]


def load_downloader_fields():
    """CANDIDATE_FIELDS from collection/ia_download.py (parsed, not imported, so nothing runs)."""
    import ast
    src = (C.REPO / "collection" / "ia_download.py").read_text(encoding="utf-8")
    for node in ast.parse(src).body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "CANDIDATE_FIELDS" for t in node.targets):
            return ast.literal_eval(node.value)
    raise C.StageError("CANDIDATE_FIELDS not found in collection/ia_download.py")


def audit_sample(sel, prof_by_id, reup, ac, seed):
    rng = random.Random(seed)
    picked = defaultdict(list)

    def take(reason, rows, k):
        rows = list(rows)
        rng.shuffle(rows)
        for r in rows[:k]:
            picked[r["identifier"]].append(reason)
    for c in C.CATEGORIES:
        take(f"category:{c}", [r for r in sel if c in C.split_list(r.get("categories"))], ac["per_category"])
    st = Counter(r["station_id"] for r in sel)
    for s, _ in st.most_common(ac["stations"] // 2):
        take(f"large_station:{s}", [r for r in sel if r["station_id"] == s], 1)
    take("small_station", [r for r in sel if st[r["station_id"]] == 1], ac["stations"] - ac["stations"] // 2)
    take("multilabel_3plus", [r for r in sel if int(r.get("n_categories") or 0) >= 3], ac["multilabel"])
    fam = defaultdict(list)
    for r in sel:
        if r.get("series_id"):
            fam[r["series_id"]].append(r)
    groups = [g for g in fam.values() if len(g) >= 2]
    rng.shuffle(groups)
    for g in groups[:ac["series_groups"]]:
        for r in g[:4]:
            picked[r["identifier"]].append(f"same_series_group:{r['series_id']}")
    take("suspected_reupload", [r for r in sel if int(r.get("suspected_reupload_pairs") or 0)], ac["reuploads"])
    by_d = sorted(sel, key=lambda r: C.to_float(r.get("duration_min"), 0))
    for r in by_d[:ac["extremes"] // 2]:
        picked[r["identifier"]].append("shortest")
    for r in by_d[-(ac["extremes"] - ac["extremes"] // 2):]:
        picked[r["identifier"]].append("longest")
    inc = [r for r in sel if any(f in C.split_list(r.get("eligibility_flags")) for f in
                                 ("no_description", "no_date", "no_exact_date", "no_creator", "no_captions"))]
    take("incomplete_metadata", inc, ac["incomplete"])
    rare = sorted(sel, key=lambda r: -C.to_float(r.get("sel_gain_coverage"), 0))
    for r in rare[:ac["rare"]]:
        picked[r["identifier"]].append("rare_coverage (highest coverage gain at pick)")
    out = []
    for i, reasons in sorted(picked.items()):
        r = prof_by_id[i]
        partners = [p for p in reup if i in (p["identifier_a"], p["identifier_b"])]
        out.append({"identifier": i, "audit_reasons": list(dict.fromkeys(reasons)), "title": r.get("title"),
                    "details_url": r.get("details_url"), "station_name": r.get("station_name"), "date": r.get("date"),
                    "duration_min": r.get("duration_min"), "primary_category": r.get("primary_category"),
                    "categories": r.get("categories"), "category_evidence": r.get("category_evidence"),
                    "series_id": r.get("series_id"), "eligibility_flags": r.get("eligibility_flags"),
                    "reupload_partners": [p["identifier_b"] if p["identifier_a"] == i else p["identifier_a"] for p in partners],
                    "description": (r.get("description") or "")[:300], **{c: "" for c in AUDIT_COLS}})
    return out


def apply_audit(sel, reserve, audit_path):
    """reviewer_decision 'exclude' or 'replace' removes a video; replacement_identifier (from the reserve) or the
    best reserve with the same primary category takes its place. corrected_categories overrides labels."""
    rows = C.read_csv(C.require(audit_path, "reviewed audit csv"))
    C.validate_columns(rows, ["identifier"] + AUDIT_COLS, str(audit_path))
    sel = {r["identifier"]: r for r in sel}
    res = {r["identifier"]: r for r in reserve}
    log = []
    for a in rows:
        dec = (a.get("reviewer_decision") or "").strip().lower()
        i = a["identifier"]
        if i not in sel:
            continue
        if a.get("corrected_categories"):
            sel[i]["categories"] = a["corrected_categories"]
            sel[i]["category_evidence"] = (sel[i].get("category_evidence") or "") + " | reviewer:corrected"
            log.append({"identifier": i, "action": "categories_corrected", "detail": a["corrected_categories"]})
        if dec in ("exclude", "replace", "remove", "drop"):
            rep = a.get("replacement_identifier", "").strip()
            if not rep or rep not in res or rep in sel:
                cands = [r for r in reserve if r["identifier"] not in sel and r["identifier"] not in {x.get("in") for x in log}]
                same = [r for r in cands if r.get("primary_category") == sel[i].get("primary_category")]
                rep = (same or cands or [{}])[0].get("identifier")
            if not rep:
                raise C.StageError("reserve exhausted while applying the audit")
            log.append({"identifier": i, "action": "replaced", "in": rep,
                        "detail": a.get("exclusion_reason") or a.get("replacement_reason") or dec})
            del sel[i]
            sel[rep] = dict(res[rep], selection_phase="audit_replacement", selection_rank="")
    return list(sel.values()), log


def plots(pool, sel, out):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        C.log(f"  plots skipped ({e})")
        return []
    out.mkdir(parents=True, exist_ok=True)
    made = []
    try:
        fig, ax = plt.subplots(figsize=(9, 3.5))
        x = range(len(C.CATEGORIES))
        ax.bar([i - 0.2 for i in x], [sum(int(r.get(f"cat_{c}") or 0) for r in pool) for c in C.CATEGORIES], 0.4, label="eligible pool")
        ax.bar([i + 0.2 for i in x], [sum(int(r.get(f"cat_{c}") or 0) for r in sel) for c in C.CATEGORIES], 0.4, label="selected")
        ax.set_xticks(list(x), C.CATEGORIES, rotation=30, ha="right")
        ax.set_ylabel("videos (multi-label)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / "categories.png", dpi=130)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.hist([C.to_float(r.get("duration_min"), 0) for r in pool], bins=30, alpha=0.5, label="eligible pool")
        ax.hist([C.to_float(r.get("duration_min"), 0) for r in sel], bins=30, alpha=0.7, label="selected")
        ax.set_xlabel("minutes")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / "durations.png", dpi=130)
        plt.close(fig)
        made = ["categories.png", "durations.png"]
    except Exception as e:  # noqa: BLE001
        C.log(f"  plots skipped ({e})")
    return made


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--out-root", default=None)
    ap.add_argument("--version", default="v1", help="manifest version tag (v1, v2 ...)")
    ap.add_argument("--apply-audit", help="reviewed human_audit.csv: apply exclusions / replacements / corrections")
    ap.add_argument("--audit-complete", action="store_true", help="declare the human audit finished")
    ap.add_argument("--no-plots", action="store_true")
    a = ap.parse_args(argv)
    cfg = C.load_config(a.config)
    seed, sc = int(cfg["seed"]), cfg["selection"]
    N = int(sc["target"])
    d1, d2, d3, d4 = (C.stage_dir(cfg, n, a.out_root) for n in (1, 2, 3, 4))
    out = C.stage_dir(cfg, 5, a.out_root)
    sel_p = C.require(d4 / f"selected_{N}.csv", "stage-4 selection")
    sel = C.read_csv(sel_p)
    reserve = C.read_csv(C.require(d4 / "reserve_candidates.csv", "stage-4 reserve"))
    prof = C.read_csv(C.require(d3 / "metadata_profiles.csv", "stage-3 profiles"))
    pairs = C.read_csv(C.require(d3 / "related_video_pairs.csv", "stage-3 pairs"))
    reup = C.read_csv(d1 / "suspected_reuploads.csv") if (d1 / "suspected_reuploads.csv").exists() else []
    for r in sel:  # stage-4 columns are prefixed sel_
        r["selection_rank"], r["selection_phase"], r["selection_gain_total"] = r.get("sel_rank"), r.get("sel_phase"), r.get("sel_gain_total")
    audit_log = []
    if a.apply_audit:
        sel, audit_log = apply_audit(sel, reserve, C.rpath(a.apply_audit))
        C.log(f"audit applied: {Counter(x['action'] for x in audit_log)}")

    ids = [r["identifier"] for r in sel]
    if len(ids) != N or len(set(ids)) != N:
        raise C.StageError(f"selection has {len(ids)} rows / {len(set(ids))} distinct identifiers, expected exactly {N}")
    pilots = {r["identifier"] for r in C.read_csv(d1 / "pilot_exclusions.csv")} if (d1 / "pilot_exclusions.csv").exists() else set()
    if pilots & set(ids):
        raise C.StageError(f"pilot videos in the selection: {sorted(pilots & set(ids))}")
    sel.sort(key=lambda r: r["identifier"])

    if not a.apply_audit:
        audit = audit_sample(sel, {r["identifier"]: r for r in sel}, reup, cfg["audit"], seed)
        C.write_csv(out / "human_audit.csv", audit, list(audit[0].keys()) if audit else ["identifier"] + AUDIT_COLS)
    v = a.version
    man_p = C.write_csv(out / f"corpus_manifest_{v}.csv", sel, MANIFEST_COLS)
    dl_fields = load_downloader_fields()
    dl = []
    for r in sel:
        row = {k: r.get(k, "") for k in dl_fields}
        row["select"] = "x"
        row["category"] = r.get("primary_category") or (C.split_list(r.get("categories")) or ["uncategorised"])[0]
        for need in ("identifier", "video_file"):
            if not row.get(need):
                raise C.StageError(f"{r['identifier']}: missing {need} needed by ia_download.py")
        dl.append(row)
    dl_p = C.write_csv(out / "download" / f"download_list_{v}.csv", dl, dl_fields)
    C.write_csv(out / "reserve_manifest.csv", [r for r in reserve if r["identifier"] not in set(ids)],
                ["reserve_rank", "identifier", "title", "details_url", "primary_category", "categories", "station_name",
                 "date", "duration_min", "video_mb", "has_captions", "series_id", "reserve_gain_total", "video_file",
                 "video_md5", "captions"])

    cache_p = d2 / "availability_cache.json"
    cache = C.read_json(cache_p) if cache_p.exists() else {}
    live_ok = sum(1 for i in ids if cache.get(i, {}).get("ok") and cache[i].get("video_present") is not False)
    status = "frozen" if (a.audit_complete and live_ok == N) else "provisionally_frozen"
    pending = ([] if a.audit_complete else ["human audit not declared complete (--audit-complete)"]) + \
              ([] if live_ok == N else [f"live availability verified for {live_ok}/{N} selected videos"])
    man_sha, dl_sha = C.sha256_file(man_p), C.sha256_file(dl_p)
    (out / f"corpus_manifest_{v}.sha256").write_text(f"{man_sha}  corpus_manifest_{v}.csv\n"
                                                     f"{dl_sha}  download/download_list_{v}.csv\n", encoding="utf-8")

    r1 = C.read_json(C.require(d1 / "consolidation_report.json", "stage-1 report"))
    r2 = C.read_json(C.require(d2 / "eligibility_report.json", "stage-2 report"))
    r3 = C.read_json(C.require(d3 / "metadata_diversity_report.json", "stage-3 report"))
    r4 = C.read_json(C.require(d4 / "selection_report.json", "stage-4 report"))
    sel_m = M.set_metrics(sel, pairs, pool_rows=prof)
    rnd = r4["strategies"].get("stratified_random", {}).get("metrics")
    provenance = {"version": v, "status": status, "pending": pending, "created": C.now_iso(), "seed": seed,
                  "config_path": cfg["_path"], "config_sha256": C.sha256_json({k: x for k, x in cfg.items() if k != "_path"}),
                  "config": {k: x for k, x in cfg.items() if k != "_path"}, "code": C.code_version(),
                  "inputs": r1["inputs"], "categories_config_sha256": r1.get("categories_config_sha256"),
                  "manifest_csv": man_p.name, "manifest_sha256": man_sha, "download_list": f"download/{dl_p.name}",
                  "download_list_sha256": dl_sha, "audit_changes": audit_log}
    C.write_json(out / f"corpus_manifest_{v}.json", {**provenance, "n_videos": N,
                                                       "videos": [{k: r.get(k, "") for k in MANIFEST_COLS} for r in sel]})
    rel_pairs = sel_m["related_pairs_retained"]
    report = {
        "title": "TVM corpus selection report (metadata-only)", **{k: provenance[k] for k in
                                                                    ("version", "status", "pending", "created", "seed", "config_sha256", "code", "manifest_sha256", "download_list_sha256")},
        "1_discovery_and_consolidation": {k: r1[k] for k in (
            "inputs", "records_total", "unique_identifiers", "duplicate_records_merged", "identifiers_in_multiple_rounds",
            "exact_md5_duplicate_groups", "exact_md5_duplicates_excluded", "suspected_reupload_pairs",
            "suspected_reupload_pairs_by_confidence", "pilot_videos_excluded", "kept", "kept_hours", "kept_reported_gb",
            "categories_kept")},
        "2_eligibility": {k: r2[k] for k in ("status_counts", "exclusion_reasons", "flags", "missing_mp4", "captions",
                                             "duration_min_usable", "resolution", "usable_hours", "usable_reported_gb",
                                             "availability")},
        "3_diversity_eligible_pool": {k: r3[k] for k in ("categories", "stations", "programme_families", "geography",
                                                         "dates", "durations")} | {
            "topic_clusters": [{k: t[k] for k in ("topic_cluster", "n_videos", "top_terms")} for t in r3["topics"]["clusters_detail"]]},
        "4_retrieval_difficulty_selected": {
            "families_with_multiple_selected": sel_m["families_with_multiple_selected"],
            "videos_in_multi_episode_families": sel_m["videos_in_multi_episode_families"],
            "same_series_pairs": rel_pairs.get("same_series", 0),
            "same_station_adjacent_date_pairs": rel_pairs.get("same_station_near_date", 0),
            "topically_similar_pairs": rel_pairs.get("similar_topic", 0),
            "same_event_pairs": rel_pairs.get("same_event", 0),
            "possible_reupload_pairs": rel_pairs.get("possible_reupload", 0),
            "related_cluster_size_distribution": sel_m["related_cluster_sizes"],
            "note": "potential retrieval distractors from metadata, not verified hard negatives"},
        "5_selection_outcome": {"selected": N, "total_hours": sel_m["total_hours"],
                                "estimated_storage_gb": sel_m["estimated_storage_gb"], "caption_share": sel_m["caption_share"],
                                "metrics": sel_m, "eligible_pool_metrics": r3["set_metrics"],
                                "stratified_random_metrics": rnd, "strategy_comparison": r4["strategies"],
                                "final_strategy": r4["final_strategy"], "selection_config": r4["selection_config"],
                                "reserve": len(reserve)},
    }
    report["plots"] = [] if a.no_plots else plots(prof, sel, out / "plots")
    C.write_json(out / "final_selection_report.json", report)
    C.log(f"{status}: {N} videos, {sel_m['total_hours']} h, {sel_m['estimated_storage_gb']} GB; manifest sha256 {man_sha[:16]}...")
    if pending:
        C.log("pending: " + "; ".join(pending))
    C.log(f"downloader list: {dl_p}")
    C.log(f"wrote {out}")


if __name__ == "__main__":
    main()
