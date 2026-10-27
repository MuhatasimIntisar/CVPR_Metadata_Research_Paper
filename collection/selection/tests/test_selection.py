"""Lightweight tests on synthetic data (no network, no real candidates): python -m pytest collection/selection/tests -q"""
import csv
import hashlib
import importlib.util
import json
import random
import sys
from pathlib import Path

import pytest
from collections import Counter

SEL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SEL))
import common as C  # noqa: E402


def load(name):
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), SEL / name)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


S1, S2, S3, S4, S5, S6 = (load(n) for n in ("01_consolidate.py", "02_eligibility.py", "03_profile_metadata.py",
                                            "04_select.py", "05_report_and_freeze.py", "06_make_batches.py"))
FIELDS = ["select", "category", "title", "duration_min", "station", "creator", "date", "weekday", "neighbours_1d",
          "date_source", "video_mb", "has_captions", "language", "matched_terms", "subjects", "found_via", "description",
          "details_url", "identifier", "video_file", "video_md5", "captions"]
CATS_CFG = {"collections": ["community_media"], "exclude_title": ["trailer"],
            "categories": {"meeting": {"terms": ['"council meeting"', '"city council"']},
                           "discussion": {"terms": ["forum", "debate"]},
                           "interview": {"terms": ["interview", '"talk show"']},
                           "presentation": {"terms": ["lecture", "address"]},
                           "news": {"terms": ["news"]},
                           "performance": {"terms": ["concert", "ballet", "choir"]},
                           "demonstration": {"terms": ["cooking", "recipe"]},
                           "guide": {"terms": ['"how to"', "tips"]},
                           "documentary": {"terms": ["documentary", "history"]},
                           "event": {"terms": ["festival", "fair", "parade"]}}}
TITLE = {"meeting": "City Council Meeting", "discussion": "Candidate Forum", "interview": "Interview with a Founder",
         "presentation": "Library Lecture", "news": "Weekly News", "performance": "Spring Choir Concert",
         "demonstration": "Cooking with Marilyn", "guide": "How to Recycle", "documentary": "Local History Documentary",
         "event": "Harvest Festival"}


def row(ident, cat, title, station, creator, date, dur, md5=None, subjects="Massachusetts; PEG", caps=True, desc=""):
    return {"select": "", "category": cat, "title": title, "duration_min": f"{dur:.1f}", "station": station,
            "creator": creator, "date": date, "weekday": "", "neighbours_1d": "0", "date_source": "aired",
            "video_mb": f"{dur * 5.6:.1f}", "has_captions": "yes" if caps else "", "language": "English",
            "matched_terms": "", "subjects": subjects, "found_via": "", "description": desc,
            "details_url": f"https://archive.org/details/{ident}", "identifier": ident, "video_file": f"{ident}.mp4",
            "video_md5": md5 or hashlib.md5(ident.encode()).hexdigest(), "captions": f"{ident}.en.vtt" if caps else ""}


def write(path, rows):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)


@pytest.fixture
def world(tmp_path):
    rng = random.Random(1)
    states = ["Massachusetts", "Ohio", "Texas", "California", "Vermont", "Michigan"]
    rows = []
    for c_i, cat in enumerate(CATS_CFG["categories"]):
        for k in range(30):
            st = f"st{(c_i * 7 + k) % 23}"
            dur = rng.choice([8, 12, 18, 25, 32, 45, 58])
            rows.append(row(f"{st}-{cat}-{k}", cat, f"{TITLE[cat]} {2024 + k % 3}-0{1 + k % 9}-1{k % 9}", st,
                            f"Station {st}", f"{2024 + k % 3}-0{1 + k % 9}-1{k % 9}", dur,
                            subjects=f"{states[k % 6]}; PEG", caps=k % 5 != 0,
                            desc=f"Topic {cat} item {k}. Watch Station {st} live on Roku every day."))
    # a programme family with episode markers (same station, consecutive days) + pilot + duplicates
    for e in range(5):
        rows.append(row(f"Christines_Corner_E15{e}", "interview", f"Christine's Corner - E15{e} - Guest {e}", "christines",
                        "Nutmeg TV", f"2025-03-0{e + 1}", 27))
    r1 = rows[:150]
    r2 = rows[100:]
    r3 = [row("pilot-video-1", "event", "Pilot Festival", "pv", "Pilot TV", "2025-01-01", 15),
          row("st1-meeting-dupmd5", "meeting", "City Council Meeting copy", "st1", "Station st1", "2025-02-02", 30,
              md5=rows[0]["video_md5"]),
          row("st2-reup", "event", rows[9]["title"], rows[9]["station"], rows[9]["creator"], rows[9]["date"],
              float(rows[9]["duration_min"]) + 0.2)]
    d = tmp_path / "rounds"
    d.mkdir()
    write(d / "r1.csv", r1)
    write(d / "r2.csv", r2)
    write(d / "r3.csv", r3)
    (tmp_path / "cats.json").write_text(json.dumps(CATS_CFG))
    write(tmp_path / "pilot.csv", [row("pilot-video-1", "event", "Pilot Festival", "pv", "Pilot TV", "2025-01-01", 15)])
    cfg = json.loads((SEL / "selection_config.json").read_text())
    cfg["paths"] = {"rounds": [{"name": "s1", "path": str(d / "r1.csv")}, {"name": "s2", "path": str(d / "r2.csv")},
                               {"name": "s3", "path": str(d / "r3.csv")}],
                    "categories_config": str(tmp_path / "cats.json"), "pilot_sources": [str(tmp_path / "pilot.csv")],
                    "output_root": str(tmp_path / "out")}
    cfg["selection"].update(target=120, budget_hours=60, min_per_category=8, reserve_size=20, swap_candidates=10)
    cfg["profiling"].update(topic_clusters=8, svd_components=20)
    cp = tmp_path / "cfg.json"
    cp.write_text(json.dumps(cfg))
    return tmp_path, cp, cfg


def run_all(cp):
    for m in (S1, S2, S3, S4, S5):
        m.main(["--config", str(cp)])


def test_category_rematch_word_boundaries_and_multilabel():
    comp = C.compile_categories(CATS_CFG)
    ev = C.match_categories("An Affair to remember: Ballet and Cooking at the Fair", "History; PEG", comp)
    cats = {e["category"] for e in ev}
    assert {"performance", "demonstration", "event", "documentary"} <= cats
    assert any(e["category"] == "documentary" and e["field"] == "subject" for e in ev)
    assert not C.match_categories("Affairs of state", "", comp)          # 'fair' is not a word here
    assert not C.match_categories("Festival trailer", "", comp)          # global exclude


def test_titles_families_and_episode_tokens():
    assert C.title_prefix("Christine's Corner - E153 - Grandma") == ("christine s corner", True)
    assert C.title_prefix("Cooking with Marilyn Episode 13: Soup")[0] == "cooking with marilyn"
    assert C.title_key("City Council Meeting April 8, 2025") == "city council meeting"
    assert C.episode_tokens("Show - E153") != C.episode_tokens("Show - E154")


def test_pipeline_end_to_end(world):
    tmp, cp, cfg = world
    run_all(cp)
    out = tmp / "out"
    master = C.read_csv(out / "01_consolidation" / "candidate_master.csv")
    ids = [r["identifier"] for r in master]
    assert len(ids) == len(set(ids))
    st = {r["identifier"]: r for r in master}
    assert st["pilot-video-1"]["exclusion_reason"] == "pilot_video"
    assert st["st1-meeting-dupmd5"]["exclusion_reason"].startswith("exact_md5_duplicate_of") or \
        st[master[0]["identifier"]]["consolidation_status"] in ("kept", "excluded")
    assert sum(r["exclusion_reason"].startswith("exact_md5") for r in master) == 1
    multi = [r for r in master if len(C.split_list(r["discovery_rounds"])) > 1]
    assert multi and all(r["consolidation_status"] in ("kept", "excluded") for r in multi)
    reup = C.read_csv(out / "01_consolidation" / "suspected_reuploads.csv")
    pairs = {tuple(sorted((p["identifier_a"], p["identifier_b"]))) for p in reup}
    assert any("st2-reup" in p for p in pairs)
    assert not any("Christines_Corner" in a and "Christines_Corner" in b for a, b in pairs)  # episodes != duplicates
    assert st["st2-reup"]["consolidation_status"] == "kept"                                   # flagged, not removed

    fams = C.read_csv(out / "03_profile" / "programme_families.csv")
    cc = [f for f in fams if f["series_key"] == "christine s corner"]
    assert cc and cc[0]["n_videos"] == "5" and cc[0]["evidence"] == "episode_marker"
    rel = C.read_csv(out / "03_profile" / "related_video_pairs.csv")
    types = {t for p in rel for t in C.split_list(p["relation_types"])}
    assert {"same_series", "same_station_near_date"} <= types

    sel = C.read_csv(out / "04_selection" / "selected_120.csv")
    assert len(sel) == 120 and len({r["identifier"] for r in sel}) == 120
    assert "pilot-video-1" not in {r["identifier"] for r in sel}
    assert sum(float(r["duration_min"]) for r in sel) / 60 <= 60 + 1e-6
    for c in CATS_CFG["categories"]:
        assert sum(int(r[f"cat_{c}"]) for r in sel) >= 8
    sids = {r["identifier"] for r in sel}
    assert not any(p["identifier_a"] in sids and p["identifier_b"] in sids and "possible_reupload" in p["relation_types"]
                   for p in rel)
    comp = C.read_csv(out / "04_selection" / "selection_comparison.csv")
    assert {r["strategy"] for r in comp} >= {"stratified_random", "coverage_only", "coverage_related", "coverage_related_budget"}
    dec = C.read_csv(out / "04_selection" / "selection_decisions.csv")
    assert len(dec) == len(master) and sum(d["status"] == "selected" for d in dec) == 120
    assert all(d["reasons"] for d in dec)

    f5 = out / "05_final"
    man = C.read_csv(f5 / "corpus_manifest_v1.csv")
    mids = [r["identifier"] for r in man]
    assert mids == sorted(mids) and len(set(mids)) == 120
    sha = (f5 / "corpus_manifest_v1.sha256").read_text().split()[0]
    assert sha == C.sha256_file(f5 / "corpus_manifest_v1.csv")
    dl = C.read_csv(f5 / "download" / "download_list_v1.csv")
    assert C.csv_header(f5 / "download" / "download_list_v1.csv") == S5.load_downloader_fields()
    assert len(dl) == 120 and all(C.truthy(r["select"]) and r["category"] and r["video_file"] for r in dl)
    mj = json.loads((f5 / "corpus_manifest_v1.json").read_text())
    assert mj["status"] == "provisionally_frozen" and mj["seed"] == cfg["seed"] and mj["n_videos"] == 120
    audit = C.read_csv(f5 / "human_audit.csv")
    assert audit and all(k in audit[0] for k in S5.AUDIT_COLS)

    # reproducible: same inputs + seed -> identical manifest
    S4.main(["--config", str(cp)])
    S5.main(["--config", str(cp), "--no-plots"])
    assert C.sha256_file(f5 / "corpus_manifest_v1.csv") == sha

    # audit replacement keeps exactly N
    a = C.read_csv(f5 / "human_audit.csv")
    a[0]["reviewer_decision"] = "exclude"
    a[0]["exclusion_reason"] = "test"
    C.write_csv(tmp / "reviewed.csv", a, list(a[0].keys()))
    S5.main(["--config", str(cp), "--apply-audit", str(tmp / "reviewed.csv"), "--version", "v2", "--no-plots"])
    m2 = C.read_csv(f5 / "corpus_manifest_v2.csv")
    assert len(m2) == 120 and a[0]["identifier"] not in {r["identifier"] for r in m2}


def test_infeasible_budget_is_reported(world):
    tmp, cp, cfg = world
    for m in (S1, S2, S3):
        m.main(["--config", str(cp)])
    with pytest.raises(SystemExit) as e:
        S4.main(["--config", str(cp), "--budget-hours", "1"])
    assert "infeasible" in str(e.value)


def test_batches_stratified_audit_last(world):
    tmp, cp, cfg = world
    run_all(cp)
    S6.main(["--config", str(cp), "--batch-size", "25"])
    b = tmp / "out" / "06_batches"
    idx = C.read_csv(b / "batches_index.csv")
    audit = {r["identifier"] for r in C.read_csv(tmp / "out" / "05_final" / "human_audit.csv")}
    assert len(idx) == 120 and len({r["identifier"] for r in idx}) == 120
    order = sorted({r["batch"] for r in idx})
    first_audit = min(k for k, n in enumerate(order) if n.endswith("_audit"))
    assert all(not n.endswith("_audit") for n in order[:first_audit]) and all(n.endswith("_audit") for n in order[first_audit:])
    assert {r["identifier"] for r in idx if r["audit_sample"] == "1"} == audit
    sizes = [sum(r["batch"] == n for r in idx) for n in order]
    assert max(sizes) <= 25
    for n in order:
        rows = C.read_csv(b / n / f"{n}.csv")
        assert rows and all(r["select"] == "x" for r in rows)
        assert C.csv_header(b / n / f"{n}.csv") == S5.load_downloader_fields()
    main = [n for n in order if not n.endswith("_audit")]
    cats = [Counter(r["category"] for r in idx if r["batch"] == n) for n in main]
    for c in {r["category"] for r in idx if r["audit_sample"] == "0"}:
        v = [x[c] for x in cats]
        assert max(v) - min(v) <= 1          # each category spread evenly across the main batches
