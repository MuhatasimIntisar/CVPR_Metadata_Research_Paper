#!/usr/bin/env python3
"""Compare Level-2 models on the same videos (after level2 + cleanup ran in separate outputs roots).

  python scripts/ab_compare.py q4b=../outputs q9b=../outputs_ab/q9b q27b=../outputs_ab/q27b --ids $AB_IDS --out ../ab_report

Writes to --out:
  ab_summary.csv     per model x video: segments, moments, needs review, cleanup name fixes, names removed, shot overrides,
                     generation failures, Level-2 seconds
  ab_spotcheck.csv   blind sheet: random time points; for each, the moment covering it from every model under a shuffled
                     letter (A/B/C) - description, people, shot scale. Fill the score columns by hand.
  ab_key.csv         which letter is which model (open only after scoring)
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path


def load(root, vid):
    d = Path(root) / vid
    doc = json.loads((d / "cleanup" / "clean_metadata.json").read_text(encoding="utf-8"))
    summ = json.loads((d / "cleanup" / "cleanup_summary.json").read_text(encoding="utf-8"))
    rs = d / "level2" / "run_stats.json"
    stats = json.loads(rs.read_text(encoding="utf-8")) if rs.exists() else {}
    log = [json.loads(x) for x in (d / "cleanup" / "cleanup_log.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    return doc, summ, stats, log


def at(doc, t):
    for m in doc["moments"]:
        if m["start_s"] <= t < m["end_s"]:
            return m
    return None


def people(m):
    return "; ".join((f"{e['name']}: " if e.get("name") else "") + (e.get("description") or e.get("type", ""))
                     for e in m.get("entities") or [] if e.get("type") in ("person", "person_group"))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("models", nargs="+", help="tag=outputs_root")
    ap.add_argument("--ids", nargs="+", required=True)
    ap.add_argument("--out", default="../ab_report")
    ap.add_argument("--points", type=int, default=6, help="spot-check time points per video")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args(argv)
    models = dict(x.split("=", 1) for x in a.models)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows, docs = [], {}
    for tag, root in models.items():
        for vid in a.ids:
            try:
                doc, s, st, log = load(root, vid)
            except FileNotFoundError:
                rows.append({"model": tag, "video_id": vid, "status": "missing"})
                continue
            docs[(tag, vid)] = doc
            M = doc["moments"]
            ch = Counter(x["rule"] for x in log)
            rows.append({"model": tag, "video_id": vid, "status": "ok", "segments": len(doc["segments"]), "moments": len(M),
                         "needs_review": sum(m["annotation"]["needs_review"] for m in M),
                         "needs_review_share": round(sum(m["annotation"]["needs_review"] for m in M) / max(len(M), 1), 3),
                         "cleanup_name_fixes": ch.get("appearance", 0),
                         "names_given_to_several": sum(1 for m in M for r in m["annotation"]["review_reasons"] if "removed from all" in r),
                         "shot_overrides": ch.get("shot", 0),
                         "generation_failed": sum(m["annotation"].get("generation_status") != "ok" for m in M),
                         "named_people": sum(1 for m in M for e in m["entities"] if e.get("name") and e.get("type") == "person"),
                         "level2_s": st.get("total_s")})
    keys = sorted({k for r in rows for k in r}, key=lambda k: ["model", "video_id", "status"].index(k) if k in ("model", "video_id", "status") else 3)
    with open(out / "ab_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    rng = random.Random(a.seed)
    tags = list(models)
    sheet, key = [], []
    for vid in a.ids:
        have = [t for t in tags if (t, vid) in docs]
        if len(have) < 2:
            continue
        dur = min(docs[(t, vid)]["moments"][-1]["end_s"] for t in have)
        for k in range(a.points):
            t = round(rng.uniform(0, dur), 1)
            order = have[:]
            rng.shuffle(order)
            row = {"video_id": vid, "time_s": t, "time": f"{int(t // 60)}:{int(t % 60):02d}"}
            for letter, tag in zip("ABCDE", order):
                m = at(docs[(tag, vid)], t)
                row[f"{letter}_span"] = f"{m['start_s']:g}-{m['end_s']:g}" if m else ""
                row[f"{letter}_description"] = (m or {}).get("description") or ""
                row[f"{letter}_people"] = people(m) if m else ""
                row[f"{letter}_shot"] = ((m or {}).get("attributes") or {}).get("shot_scale") or ""
                for crit in ("correct_0_2", "names_ok_y_n", "shot_ok_y_n"):
                    row[f"{letter}_{crit}"] = ""
                key.append({"video_id": vid, "time_s": t, "letter": letter, "model": tag})
            sheet.append(row)
    if sheet:
        with open(out / "ab_spotcheck.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(sheet[0]))
            w.writeheader()
            w.writerows(sheet)
        with open(out / "ab_key.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(key[0]))
            w.writeheader()
            w.writerows(key)
    tot = {}
    for r in rows:
        if r.get("status") == "ok":
            t = tot.setdefault(r["model"], Counter())
            t.update({k: r[k] for k in ("moments", "needs_review", "cleanup_name_fixes", "names_given_to_several",
                                        "shot_overrides", "generation_failed") if isinstance(r.get(k), int)})
            t["level2_s"] += r.get("level2_s") or 0
    for m, t in tot.items():
        print(f"{m:6} moments {t['moments']:4}  needs review {t['needs_review']:3} ({t['needs_review'] / max(t['moments'], 1):.0%})  "
              f"name fixes {t['cleanup_name_fixes']:3}  names on several {t['names_given_to_several']:3}  "
              f"shot overrides {t['shot_overrides']:3}  failed {t['generation_failed']}  level2 {t['level2_s'] / 60:.0f} min")
    print(f"wrote {out}/ab_summary.csv" + (f", ab_spotcheck.csv ({len(sheet)} points), ab_key.csv" if sheet else ""))


if __name__ == "__main__":
    main()
