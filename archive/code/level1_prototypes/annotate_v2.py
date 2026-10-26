"""Level-1 v2 atomic annotations: independent, multi-label, observable-only descriptions of 15 s intervals.

    python -m level1.annotate_v2 --video-id detvde-Chocolate_Ballerina_Founder_Chane_5b4f44 [--variant masked|raw]

Inputs: data/level1/<id>/v2/ (detect_v2.py), data/level1/<id>/ (YOLO track/pose, VAD), data/segments/<id>/
(diarisation, audio events, OCR, shots, silence). numpy + matplotlib only; deterministic.
Outputs (data/level1/<id>/v2/): atomic_annotations_v2.jsonl, activity_distributions_v2.csv,
environment_distributions_v2.csv, annotation_conflicts_v2.csv, annotation_timeline_v2.png,
annotation_quality_report.md. No splitting or merging of intervals.

Scores are within-head softmax similarities (relative, NOT calibrated probabilities). Each field reports
model score, temporal coverage, number of observations, stability, contradictions and provenance separately.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

W = 15.0
R = {  # fixed rules (no annotations involved)
    "support_topk": 2, "support_rel": 1.5,   # a clip supports an activity: in its top-2 and score >= 1.5 x uniform
    "present_min_clips": 2,                  # present: >= 2 supporting clips (>= ~4 s) ...
    "strong_rel": 3.0,                       # ... or 1 clip scoring >= 3 x uniform; otherwise 'weak'
    "person_conf": 0.4, "obj_conf": 0.4,
    "black_luma": 20.0, "black_std": 8.0,
    "ast_on": 0.3, "speaker_min_s": 1.0,
    "lower_third_y": 0.65, "lower_third_w": 0.30,
}
LIVE, GRAPHIC, BLACK = 0, 1, 2  # indices into the presentation head
FOCUS = [(180, 210, "short dance/performance insert"), (420, 495, "mixed performance and studio footage"),
         (720, 870, "sustained performance-related content"), (880, 915, "late dance/visual transition")]


def load(p):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def r3(x):
    return None if x is None else round(float(x), 3)


def spans(ts, half):
    """Merge clip centres into [start, end] spans (adjacent clips joined)."""
    out = []
    for t in sorted(ts):
        if out and t - half <= out[-1][1] + 1e-6:
            out[-1][1] = r3(t + half)
        else:
            out.append([r3(t - half), r3(t + half)])
    return out


class Data:
    def __init__(self, root, vid, variant):
        l1, seg = root / "data" / "level1" / vid, root / "data" / "segments" / vid
        self.v2 = l1 / "v2"
        self.meta = load(self.v2 / "detect_v2_meta.json")
        H = self.meta["heads"]
        self.ACT, self.ENV, self.PRES = H["activity"], H["environment"], H["presentation"]
        self.variant = variant
        x = dict(np.load(self.v2 / "xclip_v2.npz"))
        c = dict(np.load(self.v2 / "clip_v2.npz"))
        self.xt, self.xs, self.xs_other = x["times"], x[variant], x["raw" if variant == "masked" else "masked"]
        self.ct = c["times"]
        o = "raw" if variant == "masked" else "masked"
        self.env, self.env_o = c[f"env_{variant}"], c[f"env_{o}"]
        self.pres, self.pres_o = c[f"pres_{variant}"], c[f"pres_{o}"]
        self.emb = c[f"emb_{variant}"]
        self.cmask = c["mask_share"]
        f = dict(np.load(self.v2 / "frame_stats.npz"))
        self.ft, self.fl, self.fsd = f["t"], f["luma_mean"], f["luma_std"]
        self.static_mask_share = self.meta["mask"]["static_mask_share"]
        self.track, self.pose, self.vad = load(l1 / "yolo_track.json"), load(l1 / "yolo_pose.json"), load(l1 / "vad.json")
        self.spk, self.ast = load(seg / "speakers.json"), load(seg / "audio_events.json")
        self.ocr, self.shots = load(seg / "ocr.json"), load(seg / "shots.json")
        self.silence = load(seg / "silence.json")
        self.duration = float((load(seg / "video.json") or {}).get("duration") or self.vad["duration_s"])
        self.persons = {}  # frame t -> person count
        for fr in self.track["frames"]:
            self.persons[fr["t"]] = sum(1 for d in fr["dets"] if d[0] == "person" and d[1] >= R["person_conf"])
        self.pt = np.array(sorted(self.persons))

    def black(self, a, b):
        m = (self.ft >= a) & (self.ft < b)
        return bool(m.any() and ((self.fl[m] < R["black_luma"]) & (self.fsd[m] < R["black_std"])).mean() >= 0.5)

    def person_near(self, t, half=1.0):
        m = (self.pt >= t - half) & (self.pt <= t + half)
        return int(max((self.persons[x] for x in self.pt[m]), default=0)), int(m.sum())

    def pres_near(self, t):
        i = int(np.argmin(np.abs(self.ct - t)))
        return int(self.pres[i].argmax()), float(self.pres[i].max()), float(abs(self.ct[i] - t))


def activity_block(D, a, b, scores, ACT):
    K = len(ACT)
    m = (D.xt >= a) & (D.xt < b)
    obs, valid_idx = [], []
    for i in np.where(m)[0]:
        t = float(D.xt[i])
        npers, nfr = D.person_near(t)
        pi, ps, _ = D.pres_near(t)
        # presentation is descriptive only: the uncalibrated head never vetoes an observation
        reason = None
        if D.black(t - 1, t + 1):
            reason = "black frame (luminance)"
        elif nfr and npers == 0:
            reason = "no person detected"
        top = np.argsort(scores[i])[::-1][:R["support_topk"]]
        obs.append({"t": r3(t), "valid": reason is None, "not_observed_reason": reason, "persons": npers,
                    "presentation": D.PRES[pi], "presentation_score": r3(ps),
                    "top": [[ACT[k], r3(scores[i][k])] for k in top]})
        if reason is None:
            valid_idx.append(i)
    present, scores_mean = [], {}
    if valid_idx:
        S = scores[valid_idx]
        scores_mean = {ACT[k]: r3(S[:, k].mean()) for k in np.argsort(S.mean(0))[::-1][:5]}
        for k in range(K):
            sup = [j for j, i in enumerate(valid_idx)
                   if k in np.argsort(scores[i])[::-1][:R["support_topk"]] and scores[i][k] >= R["support_rel"] / K]
            if not sup:
                continue
            sc = S[sup, k]
            n = len(sup)
            status = "present" if n >= R["present_min_clips"] or sc.max() >= R["strong_rel"] / K else "weak"
            ts = [float(D.xt[valid_idx[j]]) for j in sup]
            consecutive = any(b2 - a2 <= 2.01 for a2, b2 in zip(ts, ts[1:]))
            present.append({"label": ACT[k], "status": status, "n_supporting_clips": n,
                            "fraction_of_valid_clips": r3(n / len(valid_idx)), "max_score": r3(sc.max()),
                            "mean_score_when_supported": r3(sc.mean()), "time_spans_s": spans(ts, 1.0),
                            "stable_across_samples": n == 1 or consecutive})
        present.sort(key=lambda x: (-x["n_supporting_clips"], -x["max_score"]))
    n = int(m.sum())
    status = "observed" if valid_idx else ("unknown" if n == 0 else "not_observed")
    reasons = Counter(o["not_observed_reason"] for o in obs if o["not_observed_reason"])
    return {"status": status, "n_clips": n, "n_valid_clips": len(valid_idx), "temporal_coverage": r3(len(valid_idx) / n) if n else 0,
            "not_observed_reasons": dict(reasons), "activities": present, "mean_scores_top5": scores_mean,
            "observations": obs}


def environment_block(D, a, b, env, pres):
    m = (D.ct >= a) & (D.ct < b)
    obs, valid = [], []
    for i in np.where(m)[0]:
        t = float(D.ct[i])
        pi = int(pres[i].argmax())
        npers, _ = D.person_near(t, 1.0)
        reason = ("black frame (luminance)" if D.black(t - 1, t + 1) else
                  f"no person and presentation: {D.PRES[pi]}" if pi in (GRAPHIC, BLACK) and npers == 0 else None)
        k = int(env[i].argmax())
        obs.append({"t": r3(t), "label": D.ENV[k] if reason is None else "unknown", "score": r3(env[i][k]),
                    "presentation": D.PRES[pi], "presentation_score": r3(pres[i][pi]), "not_observed_reason": reason})
        if reason is None:
            valid.append(i)
    n = int(m.sum())
    if not valid:
        return {"status": "unknown", "label": "unknown", "n_observations": n, "n_valid": 0, "temporal_coverage": 0,
                "observations": obs}
    E = env[valid]
    mean = E.mean(0)
    tops = [D.ENV[int(e.argmax())] for e in E]
    k = int(mean.argmax())
    return {"status": "observed", "label": D.ENV[k], "mean_score": r3(mean[k]),
            "distribution_top5": {D.ENV[j]: r3(mean[j]) for j in np.argsort(mean)[::-1][:5]},
            "n_observations": n, "n_valid": len(valid), "temporal_coverage": r3(len(valid) / n),
            "stable": len(set(tops)) == 1, "labels_observed": dict(Counter(tops)),
            "contradictory": len(set(tops)) > 1, "observations": obs}


def interval(D, idx, a, b, prev_emb):
    rec = {"interval": idx, "start_s": r3(a), "end_s": r3(b), "variant": D.variant}
    rec["activity"] = activity_block(D, a, b, D.xs, D.ACT)
    rec["activity"]["provenance"] = f"{D.meta['models']['xclip']} 8 frames / 2 s, {D.variant} frames; validity from YOLO persons + presentation head + luminance"
    rec["environment"] = environment_block(D, a, b, D.env, D.pres)
    rec["environment"]["provenance"] = f"{D.meta['models']['clip']} every {D.meta['sampling']['clip_every_s']:g} s, {D.variant}; valid only on live footage"
    m = (D.ct >= a) & (D.ct < b)
    pc = Counter(D.PRES[int(p.argmax())] for p in D.pres[m])
    ocr_s = [s for s in (D.ocr or {}).get("samples", []) if a <= s["t"] < b]
    lower = [s for s in ocr_s if any(it["box"][1] >= R["lower_third_y"] and it["box"][2] - it["box"][0] >= R["lower_third_w"]
                                     for it in s["items"])]
    texts = []
    for s in ocr_s:
        for it in s["items"]:
            if it.get("conf", 1) >= 0.5 and it["text"] not in texts:
                texts.append(it["text"])
    fm = (D.ft >= a) & (D.ft < b)
    rec["presentation"] = {"fractions": {k: r3(v / max(1, m.sum())) for k, v in pc.items()},
                           "black_frame_share": r3(((D.fl[fm] < R["black_luma"]) & (D.fsd[fm] < R["black_std"])).mean()) if fm.any() else None,
                           "provenance": "CLIP presentation head + 4 fps luminance"}
    rec["overlays"] = {"static_overlay_mask_share": r3(D.static_mask_share),
                       "text_present_share": r3(np.mean([bool(s["items"]) for s in ocr_s])) if ocr_s else None,
                       "lower_third_share": r3(len(lower) / len(ocr_s)) if ocr_s else None,
                       "ocr_texts": texts[:20], "provenance": "static-overlay mask (detect_v2) + EasyOCR (segments/ocr.json)"}
    # people, objects, gestures (YOLO / YOLO-pose)
    frs = [f for f in D.track["frames"] if a <= f["t"] < b]
    counts = [D.persons[f["t"]] for f in frs]
    objs = {}
    for f in frs:
        for c, conf, _tid, *_ in f["dets"]:
            if c != "person" and conf >= R["obj_conf"]:
                o = objs.setdefault(c, {"frames": 0, "first_t": f["t"], "last_t": f["t"], "max_conf": 0})
                o["frames"] += 1
                o["last_t"] = f["t"]
                o["max_conf"] = max(o["max_conf"], conf)
    tracks = {}
    for f in D.track["frames"]:
        for c, conf, tid, *_ in f["dets"]:
            if c == "person" and conf >= R["person_conf"] and tid >= 0:
                tracks.setdefault(tid, []).append(f["t"])
    gest = []
    for f in D.pose["frames"]:
        if a <= f["t"] < b and any(p["kpts"][w][2] >= 0.5 and p["kpts"][s][2] >= 0.5 and p["kpts"][w][1] < p["kpts"][s][1] - 0.02
                                   for p in f["persons"] for w, s in ((9, 5), (10, 6))):
            gest.append(f["t"])
    rec["people"] = {"n_frames": len(frs), "count_median": r3(np.median(counts)) if counts else None,
                     "count_max": max(counts) if counts else None,
                     "count_distribution": dict(Counter(min(c, 6) for c in counts)) if counts else {},
                     "count_series": [[f["t"], D.persons[f["t"]]] for f in frs[::4]],
                     "arrivals": sorted(tid for tid, ts in tracks.items() if a <= ts[0] < b and len(ts) >= 2),
                     "departures": sorted(tid for tid, ts in tracks.items() if a <= ts[-1] < b and len(ts) >= 2),
                     "hand_raised_times": gest, "provenance": "YOLO11s + ByteTrack (2 fps); YOLO11n-pose"}
    rec["objects"] = {c: {"frame_share": r3(o["frames"] / max(1, len(frs))), "first_t": o["first_t"], "last_t": o["last_t"],
                          "max_conf": r3(o["max_conf"])} for c, o in sorted(objs.items(), key=lambda kv: -kv[1]["frames"])}
    # audio (independent of visual activity)
    sp = sum(max(0.0, min(b, s["end"]) - max(a, s["start"])) for s in D.vad["speech"])
    talk = {}
    for run in (D.spk or {}).get("runs", []):
        o = min(b, run["end"]) - max(a, run["start"])
        if o > 0:
            talk[run["speaker"]] = r3(talk.get(run["speaker"], 0) + o)
    win = [w for w in (D.ast or {}).get("windows", []) if a <= w["t"] < b]
    ev = {g: r3(np.mean([w["groups"].get(g, 0) >= R["ast_on"] for w in win])) if win else None
          for g in ("music", "applause", "cheering", "laughter", "whistle", "bell_gavel")}
    sil = sum(max(0.0, min(b, s["end"]) - max(a, s["start"])) for s in (D.silence or {}).get("silences", []))
    rec["audio"] = {"speech_present_share": r3(sp / (b - a)),
                    "active_speakers": {k: v for k, v in sorted(talk.items()) if v >= R["speaker_min_s"]},
                    "music_present_share": ev.pop("music"), "event_shares": ev, "silence_share": r3(sil / (b - a)),
                    "audio_visual_speaker_association": "not available",
                    "provenance": "Silero VAD; ECAPA diarisation; AST AudioSet (2 s windows, >= 0.3); ffmpeg silencedetect"}
    rec["shots"] = {"cuts_s": [r3(x["t"]) for x in (D.shots or {}).get("boundaries", []) if a <= x["t"] < b]}
    e = D.emb[m].mean(0) if m.any() else None
    e = None if e is None else e / np.linalg.norm(e)
    rec["visual_continuity_to_previous"] = r3(float(e @ prev_emb)) if e is not None and prev_emb is not None else None
    # conflicts / co-occurrence
    conf = []
    if rec["environment"].get("contradictory"):
        conf.append({"field": "environment", "type": "different environments within interval",
                     "detail": rec["environment"]["labels_observed"]})
    acts = [x for x in rec["activity"]["activities"] if x["status"] == "present"]
    if len(acts) > 1:
        conf.append({"field": "activity", "type": "multiple activities present (co-occurring or sequential)",
                     "detail": {x["label"]: x["time_spans_s"] for x in acts}})
    unstable = [x["label"] for x in rec["activity"]["activities"] if not x["stable_across_samples"]]
    if unstable:
        conf.append({"field": "activity", "type": "isolated non-consecutive support (unstable)", "detail": unstable})
    if len(pc) > 1:
        conf.append({"field": "presentation", "type": "mixed live / graphic / black within interval", "detail": dict(pc)})
    if rec["activity"]["status"] == "observed" and acts and rec["audio"]["speech_present_share"] > 0.5 and \
            not any(x["label"].startswith("talking") for x in acts):
        conf.append({"field": "audio_vs_activity", "type": "speech present but no talking activity seen (voice-over?)",
                     "detail": [x["label"] for x in acts]})
    rec["conflicts"] = conf
    return rec, e


def other_variant_summary(D, a, b):
    """Activity + environment under the other frame variant (for the masked vs raw comparison)."""
    act = activity_block(D, a, b, D.xs_other, D.ACT)
    env = environment_block(D, a, b, D.env_o, D.pres_o)
    return act, env


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video-id", required=True)
    ap.add_argument("--root", default=".")
    ap.add_argument("--variant", default="masked", choices=["masked", "raw"])
    a = ap.parse_args()
    D = Data(Path(a.root), a.video_id, a.variant)
    out = D.v2
    n = int(np.ceil(D.duration / W))
    bounds = [(i * W, min((i + 1) * W, D.duration)) for i in range(n)]
    if len(bounds) > 1 and bounds[-1][1] - bounds[-1][0] < W / 3:
        bounds[-2] = (bounds[-2][0], bounds[-1][1])
        bounds.pop()
    recs, prev = [], None
    act_rows, env_rows, conf_rows = [], [], []
    cmp = {"act_present_same": 0, "act_present_diff": 0, "env_same": 0, "env_diff": 0, "intervals": len(bounds)}
    for i, (s, e) in enumerate(bounds):
        rec, prev = interval(D, i, s, e, prev)
        recs.append(rec)
        o_act, o_env = other_variant_summary(D, s, e)
        mine = {x["label"]: x for x in rec["activity"]["activities"]}
        theirs = {x["label"]: x for x in o_act["activities"]}
        if {k for k, v in mine.items() if v["status"] == "present"} == {k for k, v in theirs.items() if v["status"] == "present"}:
            cmp["act_present_same"] += 1
        else:
            cmp["act_present_diff"] += 1
        cmp["env_same" if rec["environment"]["label"] == o_env["label"] else "env_diff"] += 1
        for lab in D.ACT:
            x, y = mine.get(lab), theirs.get(lab)
            if x or y:
                act_rows.append({"interval": i, "start_s": s, "label": lab,
                                 f"{a.variant}_status": x["status"] if x else "not_observed",
                                 f"{a.variant}_n_clips": x["n_supporting_clips"] if x else 0,
                                 f"{a.variant}_fraction": x["fraction_of_valid_clips"] if x else 0,
                                 f"{a.variant}_max_score": x["max_score"] if x else "",
                                 f"{a.variant}_spans": json.dumps(x["time_spans_s"]) if x else "",
                                 "other_variant_status": y["status"] if y else "not_observed",
                                 "other_variant_n_clips": y["n_supporting_clips"] if y else 0,
                                 "valid_clips": rec["activity"]["n_valid_clips"], "clips": rec["activity"]["n_clips"]})
        dist = rec["environment"].get("distribution_top5", {})
        mm = (D.ct >= s) & (D.ct < e)
        for j, lab in enumerate(D.ENV):
            if lab in dist or (mm.any() and D.env_o[mm][:, j].mean() >= 0.1):
                env_rows.append({"interval": i, "start_s": s, "label": lab, f"{a.variant}_mean_score_valid": dist.get(lab, ""),
                                 "other_variant_mean_score_all": r3(D.env_o[mm][:, j].mean()) if mm.any() else "",
                                 "is_top": rec["environment"]["label"] == lab, "coverage": rec["environment"]["temporal_coverage"],
                                 "stable": rec["environment"].get("stable", ""), "other_variant_top": o_env["label"]})
        for c in rec["conflicts"]:
            conf_rows.append({"interval": i, "start_s": s, **c, "detail": json.dumps(c["detail"])})
    with open(out / "atomic_annotations_v2.jsonl", "w", encoding="utf-8") as f:
        f.writelines(json.dumps(x, ensure_ascii=False) + "\n" for x in recs)
    for name, rows in (("activity_distributions_v2.csv", act_rows), ("environment_distributions_v2.csv", env_rows),
                       ("annotation_conflicts_v2.csv", conf_rows)):
        keys = []
        for rr in rows:
            keys += [k for k in rr if k not in keys]
        with open(out / name, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys or ["none"])
            w.writeheader()
            w.writerows(rows)
    plot(D, recs, out / "annotation_timeline_v2.png")
    report(D, recs, cmp, conf_rows, out / "annotation_quality_report.md")
    print(f"{len(recs)} intervals; masked-vs-raw: present-activity sets equal in {cmp['act_present_same']}/{len(recs)}, "
          f"environment equal in {cmp['env_same']}/{len(recs)}; {len(conf_rows)} conflict rows -> {out}")


def fmt_acts(rec):
    xs = rec["activity"]["activities"]
    if rec["activity"]["status"] != "observed":
        return f"{rec['activity']['status']} {rec['activity']['not_observed_reasons']}"
    return "; ".join(f"{x['label']} [{x['status']}, {x['n_supporting_clips']}/{rec['activity']['n_valid_clips']} clips, "
                     f"max {x['max_score']}, {x['time_spans_s']}]" for x in xs) or "no activity supported"


def report(D, recs, cmp, conf_rows, path):
    K = len(D.ACT)
    L = ["# Level-1 v2 annotation quality report", "",
         f"Video duration {D.duration:.1f} s, {len(recs)} intervals of 15 s, variant used: **{D.variant}** frames.", "",
         "## Taxonomy (independent heads)", "",
         f"- Physical activity (X-CLIP, {K} labels, multi-label per interval): {', '.join(D.ACT)}",
         f"- Physical environment (CLIP, {len(D.ENV)} labels; unknown when no live footage): {', '.join(D.ENV)}",
         f"- Presentation (CLIP, separate head): {', '.join(D.PRES)}",
         "- Overlays: static overlay mask + OCR text / lower-third presence (own fields)",
         "- People/objects: YOLO11s + ByteTrack, YOLO11n-pose. Audio: VAD, diarisation, AST events (own fields).",
         "- Removed: 'on-screen text or graphics' and 'scenery with no people' from activity; title cards, black and "
         "graphic frames from environment; scene clustering (replaced by a continuous visual-continuity value).", "",
         f"Activity rule: a 2 s clip supports a label if it is in the clip's top-{R['support_topk']} and scores >= "
         f"{R['support_rel']}/{K}; 'present' = >= {R['present_min_clips']} supporting clips or one clip >= {R['strong_rel']}/{K}; "
         "otherwise 'weak'. Clips are 'not observed' when black, not live footage, or no person is detected. "
         "Scores are within-head softmax similarities, not calibrated probabilities.", "",
         "## Masked vs unmasked frames", "",
         f"- Static overlay mask covers {D.static_mask_share:.1%} of the frame; OCR text boxes are masked per frame in addition.",
         f"- Present-activity sets identical in {cmp['act_present_same']}/{cmp['intervals']} intervals; "
         f"environment label identical in {cmp['env_same']}/{cmp['intervals']}.",
         f"- X-CLIP top-1 label identical on {np.mean(D.xs.argmax(1) == D.xs_other.argmax(1)):.1%} of clips; "
         f"CLIP environment top-1 identical on {np.mean(D.env.argmax(1) == D.env_o.argmax(1)):.1%} of frames; "
         f"presentation top-1 identical on {np.mean(D.pres.argmax(1) == D.pres_o.argmax(1)):.1%}.", "",
         "## Presentation head reliability", "",
         f"- Presentation top-1 = live footage on {np.mean(D.pres.argmax(1) == LIVE):.1%} of CLIP frames; YOLO detects >= 1 person "
         f"on {np.mean([D.person_near(float(t))[0] > 0 for t in D.ct]):.1%} of the same frames.",
         f"- Frames labelled graphic/black/split/photo while YOLO sees a person: "
         f"{np.mean([(int(p.argmax()) != LIVE) and D.person_near(float(t))[0] > 0 for t, p in zip(D.ct, D.pres)]):.1%}.",
         "- Because this head is uncalibrated and unstable, it is stored as metadata only and never gates activity "
         "or environment (gates: luminance black frame; no person detected).", ""]
    dj = D.ACT.index("dancing")
    for lo, hi, name in FOCUS:
        m = (D.xt >= lo) & (D.xt < hi)
        if m.any():
            L.append(f"- {lo}-{hi} s ({name}): mean 'dancing' score {D.xs[m, dj].mean():.3f} ({D.variant}) vs "
                     f"{D.xs_other[m, dj].mean():.3f} (other); clips with dancing top-1: {int((D.xs[m].argmax(1) == dj).sum())} vs "
                     f"{int((D.xs_other[m].argmax(1) == dj).sum())} of {int(m.sum())}")
    tally = Counter(c["type"] for c in conf_rows)
    L += ["", "## Conflicts and co-occurrence (all intervals)", ""] + [f"- {k}: {v}" for k, v in tally.most_common()]
    L += ["", "## Focus intervals", ""]
    for lo, hi, name in FOCUS:
        L += [f"### {lo}-{hi} s: {name}", "",
              "| interval | activities (status, clips, max score, spans) | environment (coverage, stable) | presentation | people med/max | speech / music | text/lower-third |",
              "|---|---|---|---|---|---|---|"]
        for rec in recs:
            if rec["end_s"] <= lo or rec["start_s"] >= hi:
                continue
            env = rec["environment"]
            L.append(f"| {rec['start_s']:g}-{rec['end_s']:g} | {fmt_acts(rec)} | {env['label']} "
                     f"({env['temporal_coverage']}, {env.get('stable', '-')}) | {rec['presentation']['fractions']} | "
                     f"{rec['people']['count_median']}/{rec['people']['count_max']} | {rec['audio']['speech_present_share']} / "
                     f"{rec['audio']['music_present_share']} | {rec['overlays']['text_present_share']}/{rec['overlays']['lower_third_share']} |")
        L += ["", "Clip-level observations (t: top labels, or reason not observed):", ""]
        for rec in recs:
            for o in rec["activity"]["observations"]:
                if lo <= o["t"] < hi:
                    L.append(f"- {o['t']:.0f}s: " + (", ".join(f"{l} {s}" for l, s in o["top"]) if o["valid"]
                                                       else f"not observed ({o['not_observed_reason']})"))
        L.append("")
    L += ["## All intervals (compact)", "", "| i | t | activities present | environment | live share | people med | speech | music |",
          "|---|---|---|---|---|---|---|---|"]
    for rec in recs:
        acts = ", ".join(f"{x['label']}({x['n_supporting_clips']})" for x in rec["activity"]["activities"] if x["status"] == "present")
        L.append(f"| {rec['interval']} | {rec['start_s']:g} | {acts or rec['activity']['status']} | {rec['environment']['label']} | "
                 f"{rec['presentation']['fractions'].get(D.PRES[LIVE], 0)} | {rec['people']['count_median']} | "
                 f"{rec['audio']['speech_present_share']} | {rec['audio']['music_present_share']} |")
    path.write_text("\n".join(L), encoding="utf-8")


def plot(D, recs, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(6, 1, figsize=(16, 13), sharex=True, gridspec_kw={"height_ratios": [0.6, 1.4, 3.2, 0.9, 0.9, 0.9]})
    T = D.duration
    cm = plt.get_cmap("tab20")
    # presentation
    pcol = {k: cm(i * 2) for i, k in enumerate(D.PRES)}
    for t, p in zip(D.ct, D.pres):
        ax[0].axvspan(t - 2.5, t + 2.5, color=pcol[D.PRES[int(p.argmax())]], lw=0)
    ax[0].legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for c in pcol.values()], labels=[k[:28] for k in pcol],
                 fontsize=6, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.9))
    ax[0].set_yticks([])
    ax[0].set_ylabel("presentation", rotation=0, ha="right", va="center", fontsize=8)
    # environment (valid frames only; unknown left white)
    ecol = {k: cm(i % 20) for i, k in enumerate(D.ENV)}
    used = set()
    for t, e, p in zip(D.ct, D.env, D.pres):
        if not D.black(t - 1, t + 1) and not (int(p.argmax()) in (GRAPHIC, BLACK) and D.person_near(float(t))[0] == 0):
            lab = D.ENV[int(e.argmax())]
            used.add(lab)
            ax[1].axvspan(t - 2.5, t + 2.5, color=ecol[lab], lw=0)
    ax[1].legend(handles=[plt.Rectangle((0, 0), 1, 1, color=ecol[k]) for k in sorted(used)], labels=sorted(used),
                 fontsize=6, ncol=4, loc="upper left")
    ax[1].set_yticks([])
    ax[1].set_ylabel("environment\n(observable frames)", rotation=0, ha="right", va="center", fontsize=8)
    # activity heatmap (score relative to uniform), invalid clips greyed
    K = len(D.ACT)
    H = np.log2(np.maximum(D.xs * K, 1e-3)).T
    ext = [float(D.xt[0]) - 1, float(D.xt[-1]) + 1, K - 0.5, -0.5]
    ax[2].imshow(np.clip(H, -2, 3), aspect="auto", cmap="magma", extent=ext, interpolation="nearest")
    for rec in recs:
        for o in rec["activity"]["observations"]:
            if not o["valid"]:
                ax[2].axvspan(o["t"] - 1, o["t"] + 1, color="white", alpha=0.6, lw=0)
            for x in rec["activity"]["activities"]:
                if x["status"] == "present":
                    for s, e in x["time_spans_s"]:
                        ax[2].plot([s, e], [D.ACT.index(x["label"])] * 2, color="cyan", lw=2)
    ax[2].set_yticks(range(K), D.ACT, fontsize=7)
    ax[2].set_ylabel("activity score\n(log2 x uniform)\ncyan = present", rotation=0, ha="right", va="center", fontsize=8)
    # people
    ax[3].step(D.pt, [D.persons[t] for t in D.pt], where="mid", lw=0.7, color="black")
    ax[3].set_ylabel("persons\n(YOLO)", rotation=0, ha="right", va="center", fontsize=8)
    # audio
    sp = np.zeros(int(np.ceil(T)))
    for s in D.vad["speech"]:
        sp[int(s["start"]):int(np.ceil(s["end"]))] = 1
    ax[4].fill_between(np.arange(len(sp)), 0, sp, step="post", color="tab:blue", alpha=0.5, label="speech (VAD)")
    mt = [(w["t"], w["groups"].get("music", 0)) for w in (D.ast or {}).get("windows", [])]
    if mt:
        ax[4].plot(*zip(*mt), color="tab:orange", lw=0.7, label="music (AST)")
    ax[4].legend(fontsize=7, loc="upper right")
    ax[4].set_ylabel("audio", rotation=0, ha="right", va="center", fontsize=8)
    # overlays
    ot = [(s["t"], len(s["items"]) > 0) for s in (D.ocr or {}).get("samples", [])]
    if ot:
        ax[5].fill_between([x for x, _ in ot], 0, [int(y) for _, y in ot], step="mid", color="grey", alpha=0.6, label="on-screen text (OCR)")
    ax[5].legend(fontsize=7, loc="upper right")
    ax[5].set_ylabel("overlays", rotation=0, ha="right", va="center", fontsize=8)
    for a_ in ax:
        for lo, hi, _ in FOCUS:
            a_.axvspan(lo, hi, color="yellow", alpha=0.12, lw=0)
        for k in range(1, len(recs)):
            a_.axvline(recs[k]["start_s"], color="grey", lw=0.3)
    ax[-1].set_xlim(0, T)
    ax[-1].xaxis.set_major_formatter(lambda v, _p=None: f"{int(v // 60)}:{int(v % 60):02d}")
    ax[-1].set_xlabel(f"time (m:ss). Thin grey = 15 s intervals; yellow = focus windows. Variant: {D.variant} frames.")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
