#!/usr/bin/env python3
"""Write metadata/retrieval/ (tvm-retrieval-1) and metadata/full/ from full documents that already exist (no rerun).

  python scripts/export_retrieval.py outputs/batch01/metadata metadata
  python scripts/export_retrieval.py ../outputs ../metadata       # on the cluster: every outputs/<id>/cleanup/

IN is either a folder of full <video>_metadata.json documents or an outputs root (reads */cleanup/clean_metadata.json).
Categories come from collection/selection/outputs/06_batches/batches_index.csv when it exists.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from video_metadata.cleanup.export import retrieval_view  # noqa: E402
from video_metadata.cleanup.pipeline import export  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inp")
    ap.add_argument("out")
    a = ap.parse_args(argv)
    inp, out = Path(a.inp), Path(a.out)
    files = sorted(inp.glob("*/cleanup/clean_metadata.json")) or sorted(inp.glob("*_metadata.json"))
    idx = ROOT / "collection" / "selection" / "outputs" / "06_batches" / "batches_index.csv"
    cat = {}
    if idx.exists():
        with open(idx, encoding="utf-8-sig", newline="") as f:
            cat = {r["identifier"]: r["category"] for r in csv.DictReader(f)}
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for p in files:
        doc = json.loads(p.read_text(encoding="utf-8"))
        if "moments" not in doc or doc.get("schema") == "tvm-retrieval-1":
            continue
        src = (doc.get("video") or {}).get("source") or {}
        view = retrieval_view(doc, category=cat.get(src.get("archive_identifier")))
        if p.name.endswith("_metadata.json"):
            stem = p.name[:-len("_metadata.json")]
        else:  # outputs/<id>/cleanup/clean_metadata.json: name it after the original video file, as the pipeline does
            man = p.parent.parent / "source" / "source_manifest.json"
            inp_ = (json.loads(man.read_text(encoding="utf-8")).get("inputs") or {}) if man.exists() else {}
            stem = Path((inp_.get("original_names") or {}).get(inp_.get("video")) or p.parent.parent.name).stem
        vdir = p.parent.parent if p.name == "clean_metadata.json" else None
        D, stats = [], {}
        if vdir is not None:  # boundary decisions and run stats for the full report
            l2 = vdir / "level2"
            if (l2 / "semantic_boundary_decisions.jsonl").exists():
                D = [json.loads(x) for x in (l2 / "semantic_boundary_decisions.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
            if (l2 / "run_stats.json").exists():
                stats = json.loads((l2 / "run_stats.json").read_text(encoding="utf-8"))
        export(out, stem, doc, view, D, stats)
        n += 1
    print(f"{n} videos -> {out}/retrieval and {out}/full")


if __name__ == "__main__":
    main()
