"""Compare raw segmenter output across videos (stdlib only; reads data/segments/*/).

    python scripts/compare_segmenters.py                       # all videos under data/segments
    python scripts/compare_segmenters.py --tol 1.0 --out data/segments/comparison.json

Per video and segmenter: boundaries per minute plus segmenter-specific diagnostics.
Agreement: for segmenter A vs B, the share of A's boundaries with a B boundary within --tol seconds.
"""
import argparse
import bisect
import glob
import json
import os
import statistics as st
from collections import Counter

SEGS = ["shots", "scenes", "speakers", "silence", "audio_events", "ocr", "topic", "camera_motion"]


def load(d, n):
    p = os.path.join(d, n + ".json")
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def times(doc, pred=None):
    if not doc:
        return []
    return sorted(b["t"] for b in doc.get("boundaries", []) if pred is None or pred(b))


def near_share(a, b, tol):
    if not a or not b:
        return None
    hit = 0
    for t in a:
        i = bisect.bisect_left(b, t)
        if any(0 <= j < len(b) and abs(b[j] - t) <= tol for j in (i - 1, i)):
            hit += 1
    return round(hit / len(a), 2)


def video_stats(d, tol):
    v = load(d, "video")
    mins = v["duration"] / 60
    s = {"video": os.path.basename(d.rstrip("/\\")), "minutes": round(mins, 1),
         "title": (v.get("archive") or {}).get("title", "")[:70]}
    docs = {n: load(d, n) for n in SEGS + ["asr"]}

    sh = docs["shots"]
    if sh:
        durs = [x["duration"] for x in sh["shots"]]
        s["shots"] = {"per_min": round(len(sh["boundaries"]) / mins, 2), "median_shot_s": round(st.median(durs), 1),
                      "longest_shot_s": round(max(durs), 1),
                      "low_conf_share": round(sum(b["conf"] < 0.3 for b in sh["boundaries"]) / max(len(sh["boundaries"]), 1), 2)}
    sc = docs["scenes"]
    if sc:
        sims = [x["best_sim"] for x in sc["per_shot"][1:]]
        s["scenes"] = {"n_scenes": len(sc["scenes"]), "n_shots": len(sc["per_shot"]),
                       "sim_median": round(st.median(sims), 2) if sims else None}
    sp = docs["speakers"]
    if sp:
        turns = sp["boundaries"]
        s["speakers"] = {"n_speakers": sp["stats"]["n_speakers"], "turns_per_min": round(len(turns) / mins, 2),
                         "held_3s_share": round(sum(t["new_run_s"] >= 3 for t in turns) / max(len(turns), 1), 2),
                         "median_run_s": round(st.median([r["end"] - r["start"] for r in sp["runs"]]), 1) if sp["runs"] else None}
    si = docs["silence"]
    if si:
        d_ = [x["duration"] for x in si["silences"]]
        s["silence"] = {"per_min": round(len(d_) / mins, 2), "over_1s": sum(x >= 1 for x in d_),
                        "max_s": round(max(d_), 1) if d_ else 0}
    ae = docs["audio_events"]
    if ae:
        c = Counter(b["group"] for b in ae["boundaries"] if b["kind"].endswith("_start"))
        s["audio_events"] = {"active_share": {k: v_ for k, v_ in ae["stats"]["share_of_windows_active"].items() if v_ > 0},
                             "starts_per_min": {k: round(n / mins, 2) for k, n in c.items()}}
    oc = docs["ocr"]
    if oc:
        b = oc["boundaries"]
        s["ocr"] = {"per_min": round(len(b) / mins, 2), "lower_third": sum(x["lower_third"] for x in b),
                    "overlay": oc.get("static_overlay", [])[:5],
                    "with_text_share": round(oc["stats"]["samples_with_text"] / max(oc["stats"]["n_samples"], 1), 2)}
    tp = docs["topic"]
    if tp:
        s["topic"] = {"per_min": round(len(tp["boundaries"]) / mins, 2)}
    cm = docs["camera_motion"]
    if cm:
        runs = cm["runs"]
        s["camera_motion"] = {"label_share": cm["stats"]["label_share"],
                              "changes_per_min": round(len(cm["boundaries"]) / mins, 2),
                              "moving_runs_over_1s": sum(r["label"] != "static" and r["end"] - r["start"] >= 1 for r in runs)}
    a = docs["asr"]
    if a:
        s["asr"] = {"language": a["stats"]["language"], "words_per_min": round(a["stats"]["n_words"] / mins),
                    "speech_coverage": a["stats"]["speech_coverage"], "cleanup": a["stats"].get("cleanup")}

    T = {"cut": times(sh), "scene": times(sc), "turn": times(sp), "turn_held": times(sp, lambda b: b["new_run_s"] >= 3),
         "topic": times(tp), "ocr": times(oc), "ocr_lower": times(oc, lambda b: b.get("lower_third")),
         "silence": times(si), "applause": times(ae, lambda b: b["group"] in ("applause", "cheering") and b["kind"].endswith("start")),
         "music_edge": times(ae, lambda b: b["group"] == "music"), "camera": times(cm)}
    pairs = [("turn", "cut"), ("turn_held", "cut"), ("cut", "turn"), ("topic", "turn"), ("topic", "cut"),
             ("topic", "silence"), ("ocr_lower", "turn"), ("ocr", "cut"), ("scene", "turn"), ("applause", "cut"),
             ("music_edge", "cut"), ("camera", "cut")]
    s["agreement"] = {f"{a}->{b}": near_share(T[a], T[b], tol) for a, b in pairs if T[a] and T[b]}
    s["counts"] = {k: len(v_) for k, v_ in T.items()}
    return s


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=os.path.join("data", "segments"))
    ap.add_argument("--tol", type=float, default=1.0, help="seconds for two boundaries to count as agreeing")
    ap.add_argument("--out", default=None, help="write the full comparison as JSON")
    args = ap.parse_args()
    dirs = sorted(d for d in glob.glob(os.path.join(args.root, "*")) if os.path.exists(os.path.join(d, "video.json")))
    rows = [video_stats(d, args.tol) for d in dirs]

    hdr = f"{'video':34} {'min':>5} {'cut/m':>6} {'shot':>5} {'scn':>4} {'spk':>3} {'turn/m':>6} {'held':>4} " \
          f"{'top/m':>5} {'ocr/m':>5} {'sil/m':>5} {'cam/m':>5} {'static':>6}"
    print(hdr)
    for r in rows:
        g = lambda k, f, dflt="-": (r.get(k) or {}).get(f, dflt)
        print(f"{r['video'][:34]:34} {r['minutes']:>5} {g('shots','per_min'):>6} {g('shots','median_shot_s'):>5} "
              f"{g('scenes','n_scenes'):>4} {g('speakers','n_speakers'):>3} {g('speakers','turns_per_min'):>6} "
              f"{g('speakers','held_3s_share'):>4} {g('topic','per_min'):>5} {g('ocr','per_min'):>5} "
              f"{g('silence','per_min'):>5} {g('camera_motion','changes_per_min'):>5} "
              f"{(r.get('camera_motion') or {}).get('label_share', {}).get('static', '-'):>6}")
    print(f"\nAgreement (share of A's boundaries with a B boundary within {args.tol}s):")
    keys = sorted({k for r in rows for k in r["agreement"]})
    print(f"{'video':34} " + " ".join(f"{k:>14}" for k in keys))
    for r in rows:
        print(f"{r['video'][:34]:34} " + " ".join(f"{str(r['agreement'].get(k, '-')):>14}" for k in keys))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=1)
        print(f"\nFull comparison -> {args.out}")


if __name__ == "__main__":
    main()
