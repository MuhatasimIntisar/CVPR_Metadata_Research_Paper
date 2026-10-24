"""Training-free moment proposals from Level-1 evidence (no models, no semantics, deterministic).

    python -m level1.moments --video-id detvde-Chocolate_Ballerina_Founder_Chane_5b4f44

Inputs (reused, never modified): data/level1/<id>/v2/{xclip_v2,clip_v2,frame_stats}.npz + detect_v2_meta.json,
data/level1/<id>/{yolo_track,vad}.json, data/segments/<id>/{speakers,audio_events,shots}.json.
Outputs (data/level1/<id>/moments/): moment_proposals.json, moment_boundary_evidence.json, moment_timeline.png,
moment_proposal_report.md.

Method
 1. Evidence tracks on the 2 s X-CLIP grid (finer tracks kept at their own rate):
    activity scores (2 s), environment + CLIP embedding (5 s), person count (0.5 s), speech (VAD), music (AST),
    shot cuts, black frames (luminance).
 2. Activity state per step: scores averaged over the step and its valid neighbours (+-2 s); dominant label if
    its smoothed score >= 2x uniform, else 'uncertain'. Black frames -> 'unobservable'; no person -> 'no person
    visible' (an observable state). Noise rule for 2 s sampling: a state must hold >= 2 steps (4 s); a 1-step
    run between two runs of the same state is bridged; other 1-step runs become 'uncertain'.
 3. Candidate changes, each typed and weighted:
      activity onset/offset/change 1.0 (0.5 within a label family, e.g. talking postures; 0.5 to/from uncertain)
      environment change persisting >= 2 CLIP frames 0.6 | major visual change (CLIP cosine < 0.80) 0.4
      dominant people-group change (>= 2 count bins, 6 s windows) 0.4 | music or speech on/offset 0.3 | cut 0.2
    Evidence within 2 s is pooled (each type counted once); a boundary needs pooled weight >= 1.0.
    So a cut alone never ends a moment, audio alone never does, and continuing speech never blocks a new activity.
 4. Moments are the spans between accepted boundaries. Spans that are mostly uncertain/unobservable, or shorter
    than the 4 s noise floor, are kept and labelled 'ambiguous'/'unobservable' rather than given an activity.
All constants are fixed a priori in PARAM; none were tuned on this video.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

PARAM = {
    "step_s": 2.0, "smooth_steps": 1, "dominant_rel": 2.0, "min_steps": 2,
    "weights": {"activity": 1.0, "activity_family": 0.5, "activity_uncertain": 0.5, "observability": 1.0,
                "environment": 0.6, "visual": 0.4, "people": 0.4, "audio": 0.3, "shot": 0.2},
    "boundary_threshold": 1.0, "pool_s": 2.0,
    "env_persist_frames": 2, "visual_cos": 0.80, "people_window_steps": 3, "people_bin_jump": 2,
    "audio_window_steps": 3, "audio_jump": 0.67, "ast_on": 0.3, "person_conf": 0.4,
    "black_luma": 20.0, "black_std": 8.0, "people_bins": [0, 1, 2, 3, 6],
}
FAMILIES = [{"talking while seated", "talking to the camera", "talking while standing"}]  # taxonomy-level, generic
NO_PERSON, UNCERTAIN, UNOBS = "no person visible", "uncertain", "unobservable"


def load(p):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def r(x, n=3):
    return None if x is None else round(float(x), n)


def family(a, b):
    return any(a in f and b in f for f in FAMILIES)


class Tracks:
    def __init__(self, root, vid):
        l1, seg = root / "data" / "level1" / vid, root / "data" / "segments" / vid
        v2 = l1 / "v2"
        meta = load(v2 / "detect_v2_meta.json")
        self.ACT, self.ENV = meta["heads"]["activity"], meta["heads"]["environment"]
        x, c, f = (dict(np.load(v2 / n)) for n in ("xclip_v2.npz", "clip_v2.npz", "frame_stats.npz"))
        self.t, self.S = x["times"].astype(float), x["masked"]
        self.ct, self.env, self.emb = c["times"].astype(float), c["env_masked"], c["emb_masked"]
        self.ft, self.fl, self.fs = f["t"], f["luma_mean"], f["luma_std"]
        track = load(l1 / "yolo_track.json")
        self.pt = np.array([fr["t"] for fr in track["frames"]])
        self.pc = np.array([sum(1 for d in fr["dets"] if d[0] == "person" and d[1] >= PARAM["person_conf"])
                            for fr in track["frames"]])
        self.vad = load(l1 / "vad.json")["speech"]
        self.spk = (load(seg / "speakers.json") or {}).get("runs", [])
        ast = (load(seg / "audio_events.json") or {}).get("windows", [])
        self.at = np.array([w["t"] for w in ast])
        self.music = np.array([w["groups"].get("music", 0) >= PARAM["ast_on"] for w in ast], float)
        self.cuts = [b["t"] for b in (load(seg / "shots.json") or {}).get("boundaries", [])]
        self.duration = float(max(self.t[-1] + 1, self.ft[-1]))

    def span_mask(self, arr, t, h=1.0):
        return (arr >= t - h) & (arr < t + h)

    def step_obs(self):
        K, h = len(self.ACT), PARAM["step_s"] / 2
        obs = []
        for t in self.t:
            fm = self.span_mask(self.ft, t, h)
            black = bool(fm.any() and ((self.fl[fm] < PARAM["black_luma"]) & (self.fs[fm] < PARAM["black_std"])).mean() >= 0.5)
            pm = self.span_mask(self.pt, t, h)
            persons = int(np.median(self.pc[pm])) if pm.any() else None
            speech = sum(max(0.0, min(t + h, s["end"]) - max(t - h, s["start"])) for s in self.vad) / (2 * h)
            am = self.span_mask(self.at, t, h)
            music = float(self.music[am].mean()) if am.any() else 0.0
            talk = Counter()
            for s in self.spk:
                o = min(t + h, s["end"]) - max(t - h, s["start"])
                if o > 0:
                    talk[s["speaker"]] += o
            ci = int(np.argmin(np.abs(self.ct - t)))
            obs.append({"t": r(t), "black": black, "persons": persons, "speech": r(speech), "music": r(music),
                        "speaker": talk.most_common(1)[0][0] if talk else None,
                        "env": self.ENV[int(self.env[ci].argmax())], "env_t": r(self.ct[ci])})
        valid = np.array([not o["black"] and (o["persons"] or 0) > 0 for o in obs])
        n = len(obs)
        for k, o in enumerate(obs):
            raw = self.S[k]
            top = np.argsort(raw)[::-1][:2]
            o["activity_raw_top2"] = [[self.ACT[j], r(raw[j])] for j in top]
            if o["black"]:
                o["state"], o["rel"] = UNOBS, None
                continue
            if not valid[k]:
                o["state"], o["rel"] = NO_PERSON, None
                continue
            nb = [j for j in range(max(0, k - PARAM["smooth_steps"]), min(n, k + PARAM["smooth_steps"] + 1)) if valid[j]]
            sm = self.S[nb].mean(0)
            j = int(sm.argmax())
            o["rel"] = r(sm[j] * K)
            o["smoothed_top"] = self.ACT[j]
            o["state"] = self.ACT[j] if sm[j] * K >= PARAM["dominant_rel"] else UNCERTAIN
        return obs


def runs(states):
    out = []
    for k, s in enumerate(states):
        if out and out[-1][0] == s:
            out[-1][2] = k
        else:
            out.append([s, k, k])
    return out


def denoise(obs):
    """Noise rule: >= 2 steps to count; 1-step runs between equal neighbours are bridged, others -> uncertain."""
    st = [o["state"] for o in obs]
    log = []
    R = runs(st)
    for i, (s, a, b) in enumerate(R):
        if b - a + 1 >= PARAM["min_steps"] or s in (UNCERTAIN, UNOBS):
            continue
        prev_s = R[i - 1][0] if i else None
        next_s = R[i + 1][0] if i + 1 < len(R) else None
        new = prev_s if prev_s is not None and prev_s == next_s else UNCERTAIN
        log.append({"t": obs[a]["t"], "state": s, "replaced_by": new,
                    "reason": "1-step run bridged between identical states" if new == prev_s else "1-step run (< 4 s): detection noise"})
        for k in range(a, b + 1):
            st[k] = new
    for k, o in enumerate(obs):
        o["denoised_state"] = st[k]
    return log


def candidate_changes(T, obs):
    W, ev = PARAM["weights"], []
    st = [o["denoised_state"] for o in obs]
    R = runs(st)
    for (s0, a0, b0), (s1, a1, b1) in zip(R, R[1:]):
        t = obs[a1]["t"] - PARAM["step_s"] / 2
        if UNOBS in (s0, s1):
            typ, w = "observability", W["observability"]
        elif UNCERTAIN in (s0, s1):
            typ, w = "activity_uncertain", W["activity_uncertain"]
        elif family(s0, s1):
            typ, w = "activity_family", W["activity_family"]
        else:
            typ, w = "activity", W["activity"]
        ev.append({"t": r(t), "type": typ, "weight": w, "from": s0, "to": s1,
                   "before_span": [obs[a0]["t"] - 1, obs[b0]["t"] + 1], "after_span": [obs[a1]["t"] - 1, obs[b1]["t"] + 1],
                   "before_mean_rel": r(np.mean([obs[k]["rel"] for k in range(a0, b0 + 1) if obs[k].get("rel")] or [0])),
                   "after_mean_rel": r(np.mean([obs[k]["rel"] for k in range(a1, b1 + 1) if obs[k].get("rel")] or [0]))})
    # environment (CLIP frames; non-black only), persistence >= 2 frames on the new side
    lab = [T.ENV[int(e.argmax())] for e in T.env]
    ok = [not bool(((T.fl[T.span_mask(T.ft, t)] < PARAM["black_luma"]) &
                    (T.fs[T.span_mask(T.ft, t)] < PARAM["black_std"])).mean() >= 0.5) for t in T.ct]
    idx = [i for i in range(len(lab)) if ok[i]]
    for p, q in zip(idx, idx[1:]):
        if lab[q] != lab[p]:
            nxt = idx[idx.index(q):idx.index(q) + PARAM["env_persist_frames"]]
            persist = len(nxt) == PARAM["env_persist_frames"] and all(lab[j] == lab[q] for j in nxt)
            ev.append({"t": r((T.ct[p] + T.ct[q]) / 2), "type": "environment", "weight": W["environment"] if persist else 0.0,
                       "from": lab[p], "to": lab[q], "persisted_frames": sum(1 for j in nxt if lab[j] == lab[q]),
                       "note": None if persist else "did not persist >= 2 frames (noise)"})
    for p, q in zip(idx, idx[1:]):
        cos = float(T.emb[p] @ T.emb[q] / (np.linalg.norm(T.emb[p]) * np.linalg.norm(T.emb[q])))
        if cos < PARAM["visual_cos"]:
            ev.append({"t": r((T.ct[p] + T.ct[q]) / 2), "type": "visual", "weight": W["visual"], "clip_cosine": r(cos)})
    # people group (ordinal count bins, 6 s windows either side), local maxima only
    bins = [None if o["persons"] is None else int(np.searchsorted(PARAM["people_bins"], o["persons"], side="right") - 1) for o in obs]
    m, cand = PARAM["people_window_steps"], []
    for k in range(m, len(obs) - m + 1):
        a = [b for b in bins[k - m:k] if b is not None]
        b_ = [b for b in bins[k:k + m] if b is not None]
        if a and b_:
            d = abs(np.median(b_) - np.median(a))
            if d >= PARAM["people_bin_jump"]:
                cand.append((k, d, np.median(a), np.median(b_)))
    for k, d, a, b_ in cand:
        if all(d >= d2 for k2, d2, *_ in cand if abs(k2 - k) <= m):
            ev.append({"t": r(obs[k]["t"] - 1), "type": "people", "weight": W["people"],
                       "from_bin_median": r(a), "to_bin_median": r(b_)})
    # audio: music / speech on- or offset (share over 6 s windows)
    m = PARAM["audio_window_steps"]
    for key in ("music", "speech"):
        cand = []
        for k in range(m, len(obs) - m + 1):
            a = np.mean([o[key] for o in obs[k - m:k]])
            b_ = np.mean([o[key] for o in obs[k:k + m]])
            if abs(b_ - a) >= PARAM["audio_jump"]:
                cand.append((k, abs(b_ - a), a, b_))
        for k, d, a, b_ in cand:
            if all(d >= d2 for k2, d2, *_ in cand if abs(k2 - k) <= m):
                ev.append({"t": r(obs[k]["t"] - 1), "type": "audio", "weight": PARAM["weights"]["audio"], "stream": key,
                           "before_share": r(a), "after_share": r(b_)})
    for c in T.cuts:
        ev.append({"t": r(c), "type": "shot", "weight": W["shot"]})
    return sorted(ev, key=lambda e: e["t"])


def pool(ev):
    """Group evidence within pool_s; each type counts once (its max weight); accept if sum >= threshold."""
    groups = []
    for e in ev:
        if e["weight"] <= 0:
            continue
        if groups and e["t"] - groups[-1]["first_t"] <= PARAM["pool_s"]:
            groups[-1]["evidence"].append(e)
        else:
            groups.append({"first_t": e["t"], "evidence": [e]})
    out = []
    for g in groups:
        best = {}
        for e in g["evidence"]:
            key = "activity" if e["type"].startswith("activity") else e["type"]
            if key not in best or e["weight"] > best[key]["weight"]:
                best[key] = e
        score = sum(e["weight"] for e in best.values())
        anchor = max(best.values(), key=lambda e: (e["weight"], e["type"].startswith("activity")))
        t = round(anchor["t"] / PARAM["step_s"]) * PARAM["step_s"]
        ok = score >= PARAM["boundary_threshold"]
        parts = " + ".join("%s %g" % (k, v["weight"]) for k, v in sorted(best.items()))
        out.append({"t": r(t), "score": r(score), "accepted": ok, "types": sorted(best), "anchor_type": anchor["type"],
                    "evidence": g["evidence"],
                    "reason": "%s: pooled evidence %.2f %s %g (%s)" % ("accepted" if ok else "rejected", score,
                                                                      ">=" if ok else "<", PARAM["boundary_threshold"], parts)})
    return out


def build(vid, T, obs, groups):
    acc = [g for g in groups if g["accepted"]]
    cuts = sorted({0.0, *[g["t"] for g in acc], round(T.duration, 2)})
    gmap = {g["t"]: g for g in acc}
    moments = []
    for i, (a, b) in enumerate(zip(cuts, cuts[1:])):
        if b - a <= 0:
            continue
        steps = [o for o in obs if a <= o["t"] < b]
        states = Counter(o["denoised_state"] for o in steps)
        n = max(1, len(steps))
        act = {k: v for k, v in states.items() if k not in (UNCERTAIN, UNOBS)}
        dom, dom_n = (max(act.items(), key=lambda kv: kv[1]) if act else (None, 0))
        unobs, unc = states.get(UNOBS, 0) / n, states.get(UNCERTAIN, 0) / n
        if unobs >= 0.5:
            status = "unobservable"
        elif b - a < PARAM["min_steps"] * PARAM["step_s"] or unc >= 0.5 or dom_n / n < 0.5:
            status = "ambiguous"
        else:
            status = "moment"
        envs = Counter(o["env"] for o in steps if not o["black"])
        rel = [o["rel"] for o in steps if o["denoised_state"] == dom and o.get("rel")]
        best = max(steps, key=lambda o: o["rel"] or 0) if steps else None
        sg, eg = gmap.get(a), gmap.get(b)
        moments.append({
            "moment_id": f"{vid}_m{i:03d}", "video_id": vid, "start_s": r(a), "end_s": r(b), "duration_s": r(b - a),
            "status": status,
            "dominant_activities": [{"label": k, "share_of_steps": r(v / n)} for k, v in states.most_common() if v / n >= 0.15],
            "environment": [{"label": k, "share_of_frames": r(v / max(1, sum(envs.values())))} for k, v in envs.most_common(3)],
            "people_median": r(np.median([o["persons"] for o in steps if o["persons"] is not None])) if steps else None,
            "audio": {"speech_share": r(np.mean([o["speech"] for o in steps])) if steps else None,
                      "music_share": r(np.mean([o["music"] for o in steps])) if steps else None,
                      "speakers": dict(Counter(o["speaker"] for o in steps if o["speaker"]))},
            "supporting_observations": [{"t": o["t"], "state": o["denoised_state"], "raw_top2": o["activity_raw_top2"],
                                         "rel": o.get("rel"), "env": o["env"], "persons": o["persons"],
                                         "speech": o["speech"], "music": o["music"]} for o in steps],
            "start_boundary": ({"t": sg["t"], "score": sg["score"], "types": sg["types"], "reason": sg["reason"]}
                               if sg else {"t": r(a), "reason": "video start"}),
            "end_boundary": ({"t": eg["t"], "score": eg["score"], "types": eg["types"], "reason": eg["reason"]}
                             if eg else {"t": r(b), "reason": "video end"}),
            "uncertainty": {"uncertain_share": r(unc), "unobservable_share": r(unobs),
                            "dominant_share": r(dom_n / n), "mean_rel_score_dominant": r(np.mean(rel)) if rel else None,
                            "environment_labels": len(envs), "weakest_boundary_score": r(min(x["score"] for x in (sg, eg) if x))
                            if (sg or eg) else None},
            "representative_frames_s": sorted({r(a + 1), r((a + b) / 2), r(b - 1)} | ({best["t"]} if best and best.get("rel") else set())),
        })
    return moments


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video-id", required=True)
    ap.add_argument("--root", default=".")
    a = ap.parse_args()
    root = Path(a.root)
    T = Tracks(root, a.video_id)
    obs = T.step_obs()
    noise = denoise(obs)
    ev = candidate_changes(T, obs)
    groups = pool(ev)
    moments = build(a.video_id, T, obs, groups)
    out = root / "data" / "level1" / a.video_id / "moments"
    out.mkdir(parents=True, exist_ok=True)
    (out / "moment_proposals.json").write_text(json.dumps({"video_id": a.video_id, "params": PARAM,
        "label_families": [sorted(f) for f in FAMILIES], "n_moments": len(moments), "moments": moments},
        indent=1, ensure_ascii=False), encoding="utf-8")
    (out / "moment_boundary_evidence.json").write_text(json.dumps({"video_id": a.video_id,
        "pooled_candidates": groups, "activity_noise_suppressed": noise,
        "step_observations": obs}, indent=1, ensure_ascii=False), encoding="utf-8")
    plot(T, obs, groups, moments, out / "moment_timeline.png")
    report(a.video_id, moments, groups, noise, out / "moment_proposal_report.md")
    st = Counter(m["status"] for m in moments)
    print(f"{len(moments)} proposals ({dict(st)}), {sum(g['accepted'] for g in groups)} accepted / {len(groups)} pooled candidates, "
          f"{len(noise)} noise runs -> {out}")


FOCUS = [(180, 210), (420, 495), (720, 870), (880, 915)]


def report(vid, M, G, noise, path):
    d = np.array([m["duration_s"] for m in M if m["status"] == "moment"])
    st = Counter(m["status"] for m in M)
    acc, rej = [g for g in G if g["accepted"]], [g for g in G if not g["accepted"]]
    L = [f"# Moment proposals: {vid}", "",
         f"- {len(M)} spans: {st.get('moment', 0)} moments, {st.get('ambiguous', 0)} ambiguous, {st.get('unobservable', 0)} unobservable.",
         f"- Moment durations (s): min {d.min():g}, median {np.median(d):g}, mean {d.mean():.1f}, max {d.max():g}." if len(d) else "- no moments",
         f"- Pooled candidate changes: {len(G)} ({len(acc)} accepted as boundaries, {len(rej)} rejected).",
         f"- Rejected candidates by evidence: {dict(Counter('+'.join(g['types']) for g in rej).most_common(8))}",
         f"- Accepted boundaries by evidence: {dict(Counter('+'.join(g['types']) for g in acc).most_common(8))}",
         f"- Activity noise runs suppressed (1 step = 2 s): {len(noise)} "
         f"({sum(1 for x in noise if x['replaced_by'] != UNCERTAIN)} bridged, {sum(1 for x in noise if x['replaced_by'] == UNCERTAIN)} -> uncertain).",
         "", "## All proposals", "", "| id | span (s) | dur | status | dominant activities | environment | start evidence | end evidence |",
         "|---|---|---:|---|---|---|---|---|"]
    for m in M:
        acts = ", ".join(f"{x['label']} {x['share_of_steps']}" for x in m["dominant_activities"])
        env = ", ".join(f"{x['label']} {x['share_of_frames']}" for x in m["environment"][:2])
        L.append(f"| {m['moment_id'][-4:]} | {m['start_s']:g}-{m['end_s']:g} | {m['duration_s']:g} | {m['status']} | {acts} | {env} | "
                 f"{'+'.join(m['start_boundary'].get('types', ['start']))} ({m['start_boundary'].get('score', '')}) | "
                 f"{'+'.join(m['end_boundary'].get('types', ['end']))} ({m['end_boundary'].get('score', '')}) |")
    L += ["", "## Diagnostic windows", ""]
    for lo, hi in FOCUS:
        L += [f"### {lo}-{hi} s", ""]
        for m in M:
            if m["end_s"] > lo and m["start_s"] < hi:
                L.append(f"- **{m['start_s']:g}-{m['end_s']:g} s ({m['status']})**: "
                         + ", ".join(f"{x['label']} {x['share_of_steps']}" for x in m["dominant_activities"])
                         + f"; env {m['environment'][0]['label'] if m['environment'] else '-'}; "
                         f"start: {m['start_boundary']['reason']}; end: {m['end_boundary']['reason']}")
        for g in G:
            if lo <= g["t"] <= hi and not g["accepted"]:
                L.append(f"  - rejected candidate at {g['t']:g} s: {g['reason']}")
        L.append("")
    path.write_text("\n".join(L), encoding="utf-8")


def plot(T, obs, G, M, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(5, 1, figsize=(16, 9), sharex=True, gridspec_kw={"height_ratios": [1.2, 0.6, 0.6, 0.6, 1.0]})
    states = sorted({o["denoised_state"] for o in obs})
    cm = plt.get_cmap("tab20")
    col = {s: ("#dddddd" if s == UNCERTAIN else "#555555" if s == UNOBS else cm(i % 20)) for i, s in enumerate(states)}
    for o in obs:
        ax[0].axvspan(o["t"] - 1, o["t"] + 1, color=col[o["denoised_state"]], lw=0)
    ax[0].legend(handles=[plt.Rectangle((0, 0), 1, 1, color=col[s]) for s in states], labels=states, fontsize=6, ncol=6,
                 loc="upper center", bbox_to_anchor=(0.5, 1.6))
    ax[0].set_ylabel("activity\n(2 s, denoised)", rotation=0, ha="right", va="center", fontsize=8)
    envs = sorted({o["env"] for o in obs})
    ecol = {s: cm((i * 3) % 20) for i, s in enumerate(envs)}
    for o in obs:
        ax[1].axvspan(o["t"] - 1, o["t"] + 1, color=ecol[o["env"]], lw=0)
    ax[1].set_ylabel("environment", rotation=0, ha="right", va="center", fontsize=8)
    ax[2].step([o["t"] for o in obs], [o["persons"] or 0 for o in obs], where="mid", color="black", lw=0.8)
    ax[2].set_ylabel("persons", rotation=0, ha="right", va="center", fontsize=8)
    ax[3].plot([o["t"] for o in obs], [o["speech"] for o in obs], lw=0.7, label="speech")
    ax[3].plot([o["t"] for o in obs], [o["music"] for o in obs], lw=0.7, label="music")
    ax[3].legend(fontsize=6, loc="upper right")
    ax[3].set_ylabel("audio", rotation=0, ha="right", va="center", fontsize=8)
    scol = {"moment": "tab:green", "ambiguous": "tab:orange", "unobservable": "tab:gray"}
    for m in M:
        ax[4].axvspan(m["start_s"], m["end_s"], ymin=0.55, ymax=0.95, color=scol[m["status"]], alpha=0.75, lw=0.5, ec="white")
    for g in G:
        ax[4].plot([g["t"]], [0.3], marker="v" if g["accepted"] else "x", color="black" if g["accepted"] else "red", ms=4)
    ax[4].set_ylim(0, 1)
    ax[4].set_yticks([])
    ax[4].set_ylabel("proposals\n(v accepted, x rejected)", rotation=0, ha="right", va="center", fontsize=8)
    for a_ in ax:
        for g in G:
            if g["accepted"]:
                a_.axvline(g["t"], color="black", lw=0.6)
        for lo, hi in FOCUS:
            a_.axvspan(lo, hi, color="yellow", alpha=0.10, lw=0)
    ax[-1].set_xlim(0, T.duration)
    ax[-1].xaxis.set_major_formatter(lambda v, _p=None: f"{int(v // 60)}:{int(v % 60):02d}")
    ax[-1].set_xlabel("time (m:ss). Green = moment, orange = ambiguous, grey = unobservable. Yellow = diagnostic windows.")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
