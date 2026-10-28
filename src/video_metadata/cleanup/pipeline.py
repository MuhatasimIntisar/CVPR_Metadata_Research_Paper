"""Cleanup stage: rule-based fixes to the Level-2 metadata before human review (no models, seconds per video).

Reads (read-only): level2/semantic_metadata.json, the Level-1 OCR / diarisation / observations, the correction
outputs (to recompute the observed block of merged moments) and the archive metadata. Writes outputs/<id>/cleanup/:
  clean_metadata.json    the cleaned document (schema vmp-2.2: v2.1 + cleanup fields, docs/metadata_schema.md)
  cleanup_log.jsonl      one line per change: rule, moment, what, from, to, why
  cleanup_summary.json   changes per rule; review reasons before / after
  cleanup_report.md      the Level-2 report of the cleaned document
and, when paths.metadata_root is set, the dataset copy <original video name>_metadata.json/.md (replacing the
Level-2 copy). The Level-2 output itself is never modified, so the rules can always be changed and rerun.
"""
from __future__ import annotations

import copy
import datetime as dt
from collections import Counter
from pathlib import Path

from ..utils.io import read_json, read_jsonl, write_json, write_jsonl
from .rules import Ctx, apply_all

SCHEMA_VERSION = "vmp-2.2"


def run(cfg, manifest, entry, vdir: Path, log):
    from ..level2.pipeline import link_entities, observed_block, ocr_filter, report
    vid = entry["video_id"]
    l2 = read_json(vdir / "level2" / "semantic_metadata.json")
    if not l2:
        raise RuntimeError("no Level-2 metadata; run the level2 stage first")
    raw = vdir / "level1" / "raw"
    ocr = read_json(raw / "ocr.json") or {}
    obs = read_jsonl(vdir / "level1" / "observations.jsonl")
    corr = {o["step"]: o for o in read_jsonl(vdir / "correction" / "corrected_observations.jsonl")}
    cands = (read_json(vdir / "correction" / "candidate_moments.json") or {}).get("moments", [])
    filt = ocr_filter(ocr)

    def observed(a, b):
        return observed_block(a, b, obs, corr, cands, filt) if obs else {}
    ctx = Ctx(cfg["cleanup"], read_json(vdir / "source" / "source_metadata.json"),
              read_json(vdir / "source" / "transcript.json"), ocr, read_json(raw / "speakers.json"), observed)
    doc = copy.deepcopy(l2)
    before = Counter(r.split(":")[0] for m in doc["moments"] for r in m["annotation"].get("review_reasons") or [])
    n_review0 = sum(m["annotation"].get("needs_review", False) for m in doc["moments"])
    apply_all(doc, ctx)
    for m in doc["moments"]:
        for e in m.get("entities") or []:
            e.pop("video_entity_id", None)
    doc["entity_registry"] = link_entities(vid, doc["moments"])
    doc["schema_version"] = SCHEMA_VERSION
    rules = Counter(x["rule"] for x in ctx.log)
    doc["annotation_set"]["status"] = "automatic + rule-based cleanup (not gold)"
    doc["annotation_set"]["cleanup"] = {"version": cfg["cleanup"]["version"], "rule_changes": dict(rules),
                                        "applied_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}
    after = Counter(r.split(":")[0] for m in doc["moments"] for r in m["annotation"]["review_reasons"])
    n_review1 = sum(m["annotation"]["needs_review"] for m in doc["moments"])
    summary = {"video_id": vid, "cleanup_version": cfg["cleanup"]["version"], "changes_per_rule": dict(rules),
               "moments": {"before": len(l2["moments"]), "after": len(doc["moments"])},
               "needs_review": {"before": n_review0, "after": n_review1},
               "review_reasons_before": dict(before), "review_reasons_after": dict(after),
               "suppressed": sum(len(m["annotation"]["review_reasons_suppressed"]) for m in doc["moments"])}
    out = vdir / "cleanup"
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "clean_metadata.json", doc)
    write_jsonl(out / "cleanup_log.jsonl", ctx.log)
    write_json(out / "cleanup_summary.json", summary)
    D = read_jsonl(vdir / "level2" / "semantic_boundary_decisions.jsonl")
    stats = read_json(vdir / "level2" / "run_stats.json") or {}
    report(doc, D, stats, out / "cleanup_report.md")
    if cfg["paths"].get("metadata_root"):
        man = read_json(vdir / "source" / "source_manifest.json") or {"inputs": {}}
        inp = man.get("inputs") or {}
        stem = Path((inp.get("original_names") or {}).get(inp.get("video")) or vid).stem
        md = Path(cfg["paths"]["metadata_root"])
        md.mkdir(parents=True, exist_ok=True)
        write_json(md / f"{stem}_metadata.json", doc)
        report(doc, D, stats, md / f"{stem}_metadata.md")
    log.info(f"{len(ctx.log)} changes {dict(rules)}; moments {len(l2['moments'])} -> {len(doc['moments'])}; "
             f"needs review {n_review0} -> {n_review1} ({summary['suppressed']} reasons suppressed)")
