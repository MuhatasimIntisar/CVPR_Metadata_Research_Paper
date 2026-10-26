"""Fine-grained Level-1 observations (2 s steps) and 15 s window summaries, built from raw detector outputs only.

Every field keeps model scores (not just labels) and its provenance; nothing is collapsed to a single dominant
label at this stage. Presentation-head labels are kept as descriptive metadata only (they are unreliable).
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np

from ..utils.io import read_json, write_jsonl


def r(x, n=3):
    return None if x is None else round(float(x), n)


class Raw:
    def __init__(self, raw: Path, cfg: dict):
        self.cfg = cfg["level1"]
        meta = read_json(raw / "detect_v2_meta.json")
        self.ACT, self.ENV, self.PRES = (meta["heads"][k] for k in ("activity", "environment", "presentation"))
        x, c, f = (dict(np.load(raw / n)) for n in ("xclip_v2.npz", "clip_v2.npz", "frame_stats.npz"))
        self.xt, self.xs, self.xs_raw = x["times"].astype(float), x["masked"], x["raw"]
        self.ct, self.env, self.pres, self.emb = c["times"].astype(float), c["env_masked"], c["pres_masked"], c["emb_masked"]
        e = self.emb / np.linalg.norm(self.emb, axis=1, keepdims=True)
        self.cont = np.r_[np.nan, np.sum(e[1:] * e[:-1], 1)]  # cosine to previous CLIP frame
        self.ft, self.fl, self.fsd = f["t"], f["luma_mean"], f["luma_std"]
        self.track = read_json(raw / "yolo_track.json")
        self.pose = read_json(raw / "yolo_pose.json")
        self.vad = (read_json(raw / "vad.json") or {}).get("speech", [])
        self.spk = (read_json(raw / "speakers.json") or {}).get("runs", [])
        ast = (read_json(raw / "audio_events.json") or {}).get("windows", [])
        self.at = np.array([w["t"] for w in ast])
        self.ag = ast
        self.cuts = [b["t"] for b in (read_json(raw / "shots.json") or {}).get("boundaries", [])]
        self.ocr = (read_json(raw / "ocr.json") or {}).get("samples", [])
        self.motion = (read_json(raw / "camera_motion.json") or {}).get("samples", [])
        self.sil = (read_json(raw / "silence.json") or {}).get("silences", [])
        self.pt = np.array([fr["t"] for fr in self.track["frames"]]) if self.track else np.array([])
        self.topic = topic_curve(raw, self.xt, self.cfg["topics"])


def topic_curve(raw: Path, grid, tc):
    """1 - cos between word-weighted mean chunk embeddings in [t-C, t) and [t, t+C); None without enough words."""
    p = raw / "transcript_topics.npz"
    out = [(None, 0, 0)] * len(grid)
    if not p.exists():
        return out
    z = np.load(p)
    ct, nw, E = z["times"].astype(float), z["n_words"].astype(float), z["emb"]
    if not len(ct):
        return out
    C, mw = tc["context_s"], tc["min_words"]
    res = []
    for t in grid:
        L = (ct >= t - C) & (ct < t)
        R = (ct >= t) & (ct < t + C)
        wl, wr = nw[L].sum(), nw[R].sum()
        if wl < mw or wr < mw:
            res.append((None, int(wl), int(wr)))
            continue
        a, b = nw[L] @ E[L], nw[R] @ E[R]
        res.append((float(1 - a @ b / (np.linalg.norm(a) * np.linalg.norm(b))), int(wl), int(wr)))
    return res


SHOT_PROV = ("YOLO11n-pose, largest person per frame: ankles visible -> wide (extreme_wide if the person is < "
             "extreme_wide_h of frame height), knees -> medium_wide, hips -> medium, shoulders -> close_up, face only -> "
             "extreme_close_up; heuristic, people only (posters/screens showing people count)")


def shot_scale(persons, sc) -> tuple[str, float | None]:
    """Framing of the main (largest) person in one frame, from which COCO keypoints are visible."""
    ps = [p for p in persons if p["conf"] >= sc["person_conf"] and p["box"][3] - p["box"][1] >= sc["min_box_h"]]
    if not ps:
        return "no_person", None
    p = max(ps, key=lambda q: (q["box"][3] - q["box"][1]) * (q["box"][2] - q["box"][0]))
    k, h = p["kpts"], p["box"][3] - p["box"][1]

    def vis(*ids):
        return any(k[i][2] >= sc["kpt_conf"] and 0 <= k[i][0] <= 1 and 0 <= k[i][1] <= 1 for i in ids)
    if vis(15, 16):
        return ("extreme_wide" if h < sc["extreme_wide_h"] else "wide"), h
    if vis(13, 14):
        return "medium_wide", h
    if vis(11, 12):
        return "medium", h
    if vis(5, 6):
        return "close_up", h
    return "extreme_close_up", h


def overlap(a, b, s, e):
    return max(0.0, min(b, e) - max(a, s))


def step_record(R: Raw, k: int, t: float, words) -> dict:
    cfg = R.cfg
    h = cfg["step_s"] / 2
    a, b = t - h, t + h
    K = len(R.ACT)
    s = R.xs[k]
    o = np.argsort(s)[::-1]
    act = {"scores_top5": [[R.ACT[j], r(s[j])] for j in o[:5]], "top1": R.ACT[o[0]], "top1_rel": r(s[o[0]] * K),
           "top1_margin_ratio": r(s[o[0]] / max(s[o[1]], 1e-9)), "top1_unmasked": R.ACT[int(R.xs_raw[k].argmax())],
           "provenance": "X-CLIP 8 frames / 2 s, overlay-masked frames; softmax within head (relative score)"}
    ci = int(np.argmin(np.abs(R.ct - t)))
    eo = np.argsort(R.env[ci])[::-1]
    env = {"top3": [[R.ENV[j], r(R.env[ci][j])] for j in eo[:3]], "frame_t": r(R.ct[ci]), "frame_index": ci,
           "visual_continuity_to_previous_frame": r(R.cont[ci]) if np.isfinite(R.cont[ci]) else None,
           "provenance": "CLIP every 5 s (nearest frame), overlay-masked; relative score"}
    pj = int(R.pres[ci].argmax())
    fm = (R.ft >= a) & (R.ft < b)
    black = bool(fm.any() and ((R.fl[fm] < cfg["black_luma"]) & (R.fsd[fm] < cfg["black_std"])).mean() >= 0.5)
    counts, tids, objs = [], set(), Counter()
    if R.track:
        for fr in R.track["frames"]:
            if a <= fr["t"] < b:
                ps = [d for d in fr["dets"] if d[0] == "person" and d[1] >= cfg["person_conf"]]
                counts.append(len(ps))
                tids |= {d[2] for d in ps if d[2] >= 0}
                objs.update({d[0] for d in fr["dets"] if d[0] != "person" and d[1] >= cfg["object_conf"]})
    hand, scales, heights = False, [], []
    if R.pose:
        for fr in R.pose["frames"]:
            if a <= fr["t"] < b:
                lab, hh = shot_scale(fr["persons"], cfg["shot_scale"])
                scales.append(lab)
                if hh is not None:
                    heights.append(hh)
                for p in fr["persons"]:
                    kp = p["kpts"]
                    hand |= any(kp[w][2] >= .5 and kp[sh][2] >= .5 and kp[w][1] < kp[sh][1] - .02 for w, sh in ((9, 5), (10, 6)))
    mv = [m for m in R.motion if a <= m["t"] < b]
    talk = Counter()
    for run in R.spk:
        ov = overlap(a, b, run["start"], run["end"])
        if ov > 0:
            talk[run["speaker"]] += ov
    am = [w for w in R.ag if a <= w["t"] < b]
    on = cfg["ast_on"]
    ev = {g: r(np.mean([w["groups"].get(g, 0) >= on for w in am])) if am else None
          for g in ("music", "applause", "cheering", "laughter", "whistle", "bell_gavel")}
    texts = []
    for smp in R.ocr:
        if a - 0.5 <= smp["t"] < b + 0.5:
            texts += [it["text"] for it in smp["items"] if it.get("conf", 1) >= 0.5 and it["text"] not in texts]
    return {
        "step": k, "t": r(t), "start": r(a), "end": r(b),
        "activity": act, "environment": env,
        "presentation": {"label": R.PRES[pj], "score": r(R.pres[ci][pj]), "note": "metadata only (unreliable head)"},
        "black_frame": black,
        "people": {"count_median": r(np.median(counts)) if counts else None, "count_max": max(counts) if counts else None,
                   "track_ids": sorted(tids), "n_frames": len(counts), "provenance": "YOLO11s + ByteTrack, 2 fps"},
        "objects": dict(objs), "hand_raised": hand,
        "shot_scale": {"label": Counter(scales).most_common(1)[0][0] if scales else None,
                       "frames": dict(Counter(scales)), "subject_height_median": r(np.median(heights)) if heights else None,
                       "provenance": SHOT_PROV},
        "motion": {"local_resid_mean": r(np.mean([m["resid"] for m in mv]), 5) if mv else None,
                   "camera": Counter(m["label"] for m in mv).most_common(1)[0][0] if mv else None},
        "audio": {"speech_share": r(sum(overlap(a, b, x["start"], x["end"]) for x in R.vad) / (b - a)),
                  "speakers": {k_: r(v) for k_, v in talk.items()},
                  "silence_share": r(sum(overlap(a, b, x["start"], x["end"]) for x in R.sil) / (b - a)), "events": ev},
        "topic": {"change": r(R.topic[k][0], 4), "words_before": R.topic[k][1], "words_after": R.topic[k][2],
                  "provenance": "Sentence-BERT transcript chunks, 60 s context each side (1 - cosine)"},
        "shot_cuts": [r(c) for c in R.cuts if a <= c < b],
        "ocr_text": texts[:10],
        "transcript": " ".join(w["w"] for w in words if a <= (w["start"] + w["end"]) / 2 < b),
    }


def window_record(steps, i, a, b, ACT, ENV):
    st = [s for s in steps if a <= s["t"] < b]
    act = Counter(s["activity"]["top1"] for s in st if not s["black_frame"])
    env = Counter(s["environment"]["top3"][0][0] for s in st)
    pc = [s["people"]["count_median"] for s in st if s["people"]["count_median"] is not None]
    spk = Counter()
    for s in st:
        spk.update(s["audio"]["speakers"])
    return {"window": i, "start": r(a), "end": r(b), "n_steps": len(st),
            "activity_top1_counts": dict(act.most_common()), "environment_top1_counts": dict(env.most_common()),
            "people_median": r(np.median(pc)) if pc else None, "people_max": max((s["people"]["count_max"] or 0) for s in st) if st else None,
            "track_ids": sorted({t for s in st for t in s["people"]["track_ids"]}),
            "objects": dict(Counter(o for s in st for o in s["objects"]).most_common(10)),
            "speech_share": r(np.mean([s["audio"]["speech_share"] for s in st])) if st else None,
            "music_share": r(np.mean([s["audio"]["events"]["music"] or 0 for s in st])) if st else None,
            "speakers": {k: r(v) for k, v in spk.items()}, "shot_cuts": [c for s in st for c in s["shot_cuts"]],
            "black_share": r(np.mean([s["black_frame"] for s in st])) if st else None,
            "hand_raised_steps": sum(s["hand_raised"] for s in st),
            "shot_scale_counts": dict(Counter(s["shot_scale"]["label"] for s in st if s["shot_scale"]["label"]).most_common()),
            "camera_motion_counts": dict(Counter(s["motion"]["camera"] for s in st if s["motion"]["camera"]).most_common()),
            "ocr_text": list(dict.fromkeys(x for s in st for x in s["ocr_text"]))[:10],
            "transcript": " ".join(s["transcript"] for s in st if s["transcript"]),
            "note": "15 s summary = observation mechanism, not a final retrieval interval"}


def build(cfg, vdir: Path, transcript: dict):
    raw = vdir / "level1" / "raw"
    R = Raw(raw, cfg)
    words = transcript.get("words", [])
    steps = [step_record(R, k, float(t), words) for k, t in enumerate(R.xt)]
    write_jsonl(vdir / "level1" / "observations.jsonl", steps)
    W = cfg["level1"]["window_s"]
    dur = float(max(R.xt[-1] + 1, R.ft[-1])) if len(R.xt) else 0.0
    n = int(np.ceil(dur / W))
    wins = [window_record(steps, i, i * W, min((i + 1) * W, dur), R.ACT, R.ENV) for i in range(n)]
    write_jsonl(vdir / "level1" / "window_summaries.jsonl", wins)
    try:
        plot(steps, vdir / "reports" / "level1_timeline.png", dur)
    except Exception:  # plotting must never fail the stage
        pass
    return len(steps), len(wins)


def plot(steps, path: Path, dur):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(4, 1, figsize=(16, 6.5), sharex=True, gridspec_kw={"height_ratios": [1, 1, .6, .6]})
    cm = plt.get_cmap("tab20")
    for row, key in ((0, lambda s: s["activity"]["top1"]), (1, lambda s: s["environment"]["top3"][0][0])):
        labs = sorted({key(s) for s in steps})
        col = {l: cm(i % 20) for i, l in enumerate(labs)}
        for s in steps:
            ax[row].axvspan(s["start"], s["end"], color="#444" if s["black_frame"] else col[key(s)], lw=0)
        ax[row].legend(handles=[plt.Rectangle((0, 0), 1, 1, color=col[l]) for l in labs], labels=labs, fontsize=6,
                       ncol=6, loc="upper center", bbox_to_anchor=(0.5, 1.45))
        ax[row].set_yticks([])
    ax[0].set_ylabel("activity top-1", rotation=0, ha="right", fontsize=8)
    ax[1].set_ylabel("environment top-1", rotation=0, ha="right", fontsize=8)
    ax[2].step([s["t"] for s in steps], [s["people"]["count_median"] or 0 for s in steps], where="mid", color="k", lw=.8)
    ax[2].set_ylabel("persons", rotation=0, ha="right", fontsize=8)
    ax[3].plot([s["t"] for s in steps], [s["audio"]["speech_share"] for s in steps], lw=.7, label="speech (VAD)")
    ax[3].plot([s["t"] for s in steps], [s["audio"]["events"]["music"] or 0 for s in steps], lw=.7, label="music (AST)")
    ax[3].legend(fontsize=6, loc="upper right")
    for c in [c for s in steps for c in s["shot_cuts"]]:
        ax[3].axvline(c, color="grey", lw=.3)
    ax[-1].set_xlim(0, dur)
    ax[-1].xaxis.set_major_formatter(lambda v, _p=None: f"{int(v // 60)}:{int(v % 60):02d}")
    ax[-1].set_xlabel("time (m:ss); grey ticks = shot cuts; dark = black frames")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
