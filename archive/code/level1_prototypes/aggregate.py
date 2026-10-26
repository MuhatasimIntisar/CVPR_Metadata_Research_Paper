"""Level-1 robust state aggregation and pruning (deterministic; no models; no topics or descriptions).

    python -m level1.aggregate --video-id detvde-Chocolate_Ballerina_Founder_Chane_5b4f44

Inputs: data/level1/<id>/ (detect.py) + data/segments/<id>/ (shots, speakers, audio_events, ocr, camera_motion,
silence). Outputs in data/level1/<id>/:
  atomic_state_summaries.jsonl  one coarse state per 15 s interval (+ all secondary observations)
  stable_state_segments.json    merged stable-state segments with aggregate state and provenance
  state_changes.jsonl           accepted boundaries, with what changed and how long it persisted
  suppressed_changes.jsonl      changes that did not cause a boundary, and why

Primary (boundary-driving) state, compared component by component:
  environment  = CLIP environment distribution (JSD) + object-density bins (ordinal)
  activity     = X-CLIP activity distribution (JSD) + movement-intensity bins (ordinal)
  people       = person-count bins (ordinal)
  speech/audio = per-second audio-state distribution (JSD) + active-speaker-count bins (ordinal)
  scene        = mean CLIP frame embedding (cosine, mapped to [0, 1])
Missing evidence -> component left out (unknown), never counted as disagreement.
Secondary (never boundary-driving, always kept): track arrivals/departures, occlusions, gestures, transient
objects, shot cuts, framing, camera motion, speaker turns, on-screen text changes.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.spatial.distance import jensenshannon

P = {  # all fixed a priori; no reference annotations involved
    "interval_s": 15.0,
    "weights": {"environment": 0.25, "activity": 0.25, "people": 0.15, "speech_audio": 0.15, "scene": 0.20},
    "tau_change": 0.60,      # compatibility with the segment below this -> candidate change
    "tau_keep": 0.70,        # must recover above this to cancel a pending change (hysteresis band 0.60-0.70)
    "tau_strong": 0.35,      # below this vs BOTH segment and previous interval -> immediate boundary
    "persist_intervals": 2,  # a normal change must persist 2 intervals (30 s)
    "max_pending_intervals": 4,
    "scene_cos": [0.70, 0.92],  # CLIP cosine mapped linearly to [0, 1]
    "det_conf": 0.40,
    "people_bins": [0, 1, 2, 3, 6, 16], "people_labels": ["0", "1", "2", "3-5", "6-15", "16+"],
    "object_bins": [0, 1, 3, 7], "object_labels": ["none", "1-2", "3-6", "7+"],
    "motion_bins": [0, 0.0005, 0.0015, 0.004], "motion_labels": ["still", "low", "moderate", "high"],
    "speaker_bins": [0, 1, 2, 3], "speaker_labels": ["0", "1", "2", "3+"],
    "audio_states": ["silence", "speech", "music", "speech+music", "crowd", "other"],
    "ast_on": 0.30, "speaker_min_s": 1.5,
    "framing_bins": [0, 0.35, 0.70], "framing_labels": ["wide", "medium", "close-up"],
    "transient_object_share": 0.20, "stable_object_share": 0.50,
    "gesture_max_s": 5.0, "occlusion_max_s": 4.0, "min_track_s": 1.0,
}


# ---------------------------------------------------------------- helpers
def load(p: Path):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def binidx(v, edges):
    return int(np.searchsorted(edges, v, side="right") - 1)


def hist(vals, n):
    h = np.bincount(np.asarray(vals, int), minlength=n).astype(float)
    return h / h.sum() if h.sum() else None


def ordinal_sim(p, q):
    """1 - earth mover's distance between ordinal bin distributions, normalised to [0, 1]."""
    if p is None or q is None:
        return None
    return 1.0 - float(np.abs(np.cumsum(p) - np.cumsum(q)).sum()) / (len(p) - 1)


def jsd_sim(p, q):
    if p is None or q is None:
        return None
    return 1.0 - float(jensenshannon(p, q, base=2))


def top(dist, labels, k=3):
    if dist is None:
        return None
    o = np.argsort(dist)[::-1][:k]
    return [[labels[i], round(float(dist[i]), 3)] for i in o]


def r(x):
    return round(float(x), 3)


# ---------------------------------------------------------------- per-interval observation
class Obs:
    def __init__(self, l1: Path, seg: Path):
        self.meta = load(l1 / "detect_meta.json") or {}
        self.env_labels = self.meta.get("environments", [])
        self.act_labels = self.meta.get("activities", [])
        self.track = load(l1 / "yolo_track.json")
        self.pose = load(l1 / "yolo_pose.json")
        self.vad = load(l1 / "vad.json")
        z = l1 / "clip_env.npz"
        self.clip = dict(np.load(z)) if z.exists() else None
        z = l1 / "xclip_actions.npz"
        self.xclip = dict(np.load(z)) if z.exists() else None
        self.shots, self.speakers = load(seg / "shots.json"), load(seg / "speakers.json")
        self.ast, self.ocr = load(seg / "audio_events.json"), load(seg / "ocr.json")
        self.motion, self.silence = load(seg / "camera_motion.json"), load(seg / "silence.json")
        self.video = load(seg / "video.json") or {}
        self.duration = float(self.video.get("duration") or (self.vad or {}).get("duration_s") or 0)
        self._tracks = self._track_events()
        self._gestures = self._gesture_events()
        self._audio = self._audio_seconds()

    # --- secondary events over the whole video (kept with timestamps)
    def _track_events(self):
        if not self.track:
            return []
        seen = {}
        for f in self.track["frames"]:
            for c, conf, tid, *_ in f["dets"]:
                if c == "person" and conf >= P["det_conf"] and tid >= 0:
                    seen.setdefault(tid, []).append(f["t"])
        dt = 1.0 / self.track["fps"]
        ev = []
        for tid, ts in seen.items():
            dur = ts[-1] - ts[0] + dt
            if dur < P["min_track_s"]:
                ev.append({"t": ts[0], "type": "isolated_person_detection", "track": tid, "duration_s": r(dur)})
                continue
            ev.append({"t": ts[0], "type": "person_arrival", "track": tid, "visible_s": r(dur)})
            ev.append({"t": ts[-1], "type": "person_departure", "track": tid, "visible_s": r(dur)})
            for a, b in zip(ts, ts[1:]):
                if 1.5 * dt < b - a <= P["occlusion_max_s"]:
                    ev.append({"t": a, "type": "temporary_occlusion", "track": tid, "gap_s": r(b - a)})
        return sorted(ev, key=lambda e: e["t"])

    def _gesture_events(self):
        """Hand raised: a wrist above its shoulder (COCO 9/10 vs 5/6), keypoint conf >= 0.5."""
        if not self.pose:
            return []
        flags = []
        for f in self.pose["frames"]:
            up = False
            for p in f["persons"]:
                k = p["kpts"]
                for w, s in ((9, 5), (10, 6)):
                    if k[w][2] >= 0.5 and k[s][2] >= 0.5 and k[w][1] < k[s][1] - 0.02:
                        up = True
            flags.append((f["t"], up))
        ev, start = [], None
        for t, up in flags + [(1e9, False)]:
            if up and start is None:
                start = t
            elif not up and start is not None:
                d = t - start
                ev.append({"t": start, "type": "hand_raised", "duration_s": r(d),
                           "brief": d <= P["gesture_max_s"]})
                start = None
        return ev

    def _audio_seconds(self):
        """Per-second audio state from VAD (speech), AST (music / crowd / other) and silencedetect."""
        n = int(np.ceil(self.duration))
        if n == 0 or (self.vad is None and self.ast is None):
            return None
        speech = np.zeros(n)
        for s in (self.vad or {}).get("speech", []):
            for i in range(int(s["start"]), min(n, int(np.ceil(s["end"])))):
                speech[i] += max(0.0, min(s["end"], i + 1) - max(s["start"], i))
        music, crowd, other = np.zeros(n), np.zeros(n), np.zeros(n)
        for w in (self.ast or {}).get("windows", []):
            i = int(w["t"])
            if i < n:
                g = w["groups"]
                music[i] = max(music[i], g.get("music", 0))
                crowd[i] = max(crowd[i], g.get("applause", 0), g.get("cheering", 0), g.get("laughter", 0))
                other[i] = max(other[i], g.get("whistle", 0), g.get("bell_gavel", 0))
        silent = np.zeros(n)
        for s in (self.silence or {}).get("silences", []):
            silent[int(s["start"]):int(np.ceil(s["end"]))] = 1
        on = P["ast_on"]
        st = []
        for i in range(n):
            sp, mu = speech[i] >= 0.5, music[i] >= on
            st.append(3 if sp and mu else 1 if sp else 2 if mu else 4 if crowd[i] >= on else 5 if other[i] >= on
                      else 0 if silent[i] else 5)
        return np.array(st)

    # --- one interval
    def summary(self, idx, a, b):
        s = {"interval": idx, "start_s": r(a), "end_s": r(b)}
        prim, sec, unknown = {}, {}, []

        def inside(t):
            return a <= t < b

        # environment (CLIP) + scene embedding
        env = emb = None
        if self.clip is not None:
            m = (self.clip["times"] >= a) & (self.clip["times"] < b)
            if m.any():
                env = self.clip["probs"][m].mean(0)
                e = self.clip["emb"][m].mean(0)
                emb = e / (np.linalg.norm(e) or 1)
        # activity (X-CLIP), time-weighted by clip
        act = None
        if self.xclip is not None:
            m = (self.xclip["times"] >= a) & (self.xclip["times"] < b)
            if m.any():
                act = self.xclip["probs"][m].mean(0)
        # people / objects / framing from YOLO
        people = objects = framing = None
        obj_frames, n_frames = {}, 0
        if self.track:
            pb, ob, fb = [], [], []
            for f in self.track["frames"]:
                if not inside(f["t"]):
                    continue
                n_frames += 1
                ds = [d for d in f["dets"] if d[1] >= P["det_conf"]]
                ps = [d for d in ds if d[0] == "person"]
                pb.append(binidx(len(ps), P["people_bins"]))
                ob.append(binidx(len(ds) - len(ps), P["object_bins"]))
                if ps:
                    fb.append(binidx(max(d[6] - d[4] for d in ps), P["framing_bins"]))
                for c in {d[0] for d in ds if d[0] != "person"}:
                    obj_frames[c] = obj_frames.get(c, 0) + 1
            people, objects = hist(pb, 6), hist(ob, 4)
            framing = hist(fb, 3)
        # movement (local motion residual of optical flow)
        move = cam = None
        if self.motion:
            sm = [x for x in self.motion["samples"] if inside(x["t"])]
            if sm:
                move = hist([binidx(x["resid"], P["motion_bins"]) for x in sm], 4)
                labs = [x["label"] for x in sm]
                cam = {k: r(labs.count(k) / len(labs)) for k in sorted(set(labs))}
        # audio state + speakers
        audio = None
        if self._audio is not None:
            seg = self._audio[int(a):int(np.ceil(b))]
            audio = hist(seg, 6) if len(seg) else None
        spk = None
        talk = {}
        if self.speakers:
            for run in self.speakers["runs"]:
                o = min(b, run["end"]) - max(a, run["start"])
                if o > 0:
                    talk[run["speaker"]] = talk.get(run["speaker"], 0) + o
            n_act = sum(v >= P["speaker_min_s"] for v in talk.values())
            spk = hist([binidx(n_act, P["speaker_bins"])], 4)

        def put(name, dist, labels, extra=None):
            if dist is None:
                unknown.append(name)
                prim[name] = {"status": "unknown"}
                return
            prim[name] = {"dominant": labels[int(np.argmax(dist))], "p": r(dist.max()),
                          "distribution": {l: r(v) for l, v in zip(labels, dist) if v > 0.005}, **(extra or {})}

        put("environment", env, self.env_labels)
        put("activity", act, self.act_labels)
        put("people_count", people, P["people_labels"])
        put("object_density", objects, P["object_labels"])
        put("movement", move, P["motion_labels"])
        put("audio_state", audio, P["audio_states"])
        put("active_speakers", spk, P["speaker_labels"])
        prim["scene_embedding"] = {"status": "ok" if emb is not None else "unknown",
                                   "n_frames": int(((self.clip["times"] >= a) & (self.clip["times"] < b)).sum())
                                   if self.clip is not None else 0}
        # secondary observations (retained, never boundary-driving)
        sec["objects_stable"] = sorted(c for c, k in obj_frames.items() if n_frames and k / n_frames >= P["stable_object_share"])
        sec["objects_transient"] = sorted(c for c, k in obj_frames.items()
                                          if n_frames and k / n_frames < P["transient_object_share"])
        sec["object_frame_share"] = {c: r(k / n_frames) for c, k in sorted(obj_frames.items())} if n_frames else {}
        sec["framing"] = {l: r(v) for l, v in zip(P["framing_labels"], framing)} if framing is not None else None
        sec["camera_motion"] = cam
        sec["shot_cuts"] = [r(x["t"]) for x in (self.shots or {}).get("boundaries", []) if inside(x["t"])]
        sec["speaker_talk_s"] = {k: r(v) for k, v in sorted(talk.items())}
        sec["speaker_turns"] = [r(x["t"]) for x in (self.speakers or {}).get("boundaries", []) if inside(x["t"])]
        sec["track_events"] = [e for e in self._tracks if inside(e["t"])]
        sec["gestures"] = [e for e in self._gestures if inside(e["t"])]
        sec["on_screen_text_changes"] = [{"t": r(x["t"]), "text": x.get("text", "")}
                                         for x in (self.ocr or {}).get("boundaries", []) if inside(x["t"])]
        s.update(primary=prim, secondary=sec, unknown=unknown,
                 raw_counts={"yolo_frames": n_frames, "clip_frames": prim["scene_embedding"]["n_frames"]})
        vec = {"environment": env, "activity": act, "people": people, "objects": objects, "movement": move,
               "audio": audio, "speakers": spk, "emb": emb}
        return s, vec


# ---------------------------------------------------------------- state comparison
def mean_state(vecs):
    out = {}
    for k in vecs[0]:
        xs = [v[k] for v in vecs if v[k] is not None]
        if not xs:
            out[k] = None
        else:
            m = np.mean(xs, 0)
            out[k] = m / (np.linalg.norm(m) or 1) if k == "emb" else m
    return out


def combine(parts):
    """parts: [(weight, sim or None)] -> weighted mean of available parts, or None."""
    av = [(w, s) for w, s in parts if s is not None]
    return sum(w * s for w, s in av) / sum(w for w, _ in av) if av else None


def compat(A, B):
    lo, hi = P["scene_cos"]
    scene = None
    if A["emb"] is not None and B["emb"] is not None:
        scene = float(np.clip((float(A["emb"] @ B["emb"]) - lo) / (hi - lo), 0, 1))
    comp = {
        "environment": combine([(0.75, jsd_sim(A["environment"], B["environment"])),
                                (0.25, ordinal_sim(A["objects"], B["objects"]))]),
        "activity": combine([(0.70, jsd_sim(A["activity"], B["activity"])),
                             (0.30, ordinal_sim(A["movement"], B["movement"]))]),
        "people": ordinal_sim(A["people"], B["people"]),
        "speech_audio": combine([(0.75, jsd_sim(A["audio"], B["audio"])),
                                 (0.25, ordinal_sim(A["speakers"], B["speakers"]))]),
        "scene": scene,
    }
    total = combine([(P["weights"][k], v) for k, v in comp.items()])
    return total, {k: (None if v is None else r(v)) for k, v in comp.items()}


def changed_attributes(A, B, labels):
    """Plain-language list of which observable attributes differ between two states."""
    out = []
    for key, lab, kind in (("environment", "environment", "dist"), ("activity", "activity", "dist"),
                           ("people", "people_count", "ord"), ("objects", "object_density", "ord"),
                           ("movement", "movement", "ord"), ("audio", "audio_state", "dist"),
                           ("speakers", "active_speakers", "ord")):
        p, q = A[key], B[key]
        if p is None or q is None:
            continue
        sim = jsd_sim(p, q) if kind == "dist" else ordinal_sim(p, q)
        L = labels[lab]
        if sim < 0.6 or L[int(np.argmax(p))] != L[int(np.argmax(q))]:
            out.append({"attribute": lab, "from": top(p, L, 2), "to": top(q, L, 2), "similarity": r(sim)})
    if A["emb"] is not None and B["emb"] is not None:
        c = float(A["emb"] @ B["emb"])
        if c < P["scene_cos"][1] - 0.1:
            out.append({"attribute": "scene", "clip_cosine": r(c)})
    return out


# ---------------------------------------------------------------- segmentation
def segment(summ, vecs, labels):
    W, n = P["interval_s"], len(vecs)
    changes, suppressed, log = [], [], []
    segs, cur, pending, confirm = [], [0], [], 0
    seg_start_change = [None]

    def close(next_ids, kind, ca, cp, comp_a, comp_p):
        nonlocal cur
        before, after = mean_state([vecs[j] for j in cur]), mean_state([vecs[j] for j in next_ids])
        ch = {"id": len(changes), "t": summ[next_ids[0]]["start_s"], "type": kind,
              "previous_segment": [summ[cur[0]]["start_s"], summ[cur[-1]]["end_s"]],
              "persisted_s": r(len(next_ids) * W) if kind != "strong_immediate" else 0.0,
              "compat_to_segment": r(ca), "compat_to_previous_interval": None if cp is None else r(cp),
              "components_to_segment": comp_a, "components_to_previous_interval": comp_p,
              "changed": changed_attributes(before, after, labels)}
        names = ", ".join(c["attribute"] for c in ch["changed"]) or "overall state"
        how = ("immediately (strong change against both the segment and the previous interval)"
               if kind == "strong_immediate" else f"after the change persisted {len(next_ids) * W:g} s")
        drift = " Each step was small (gradual drift away from the segment state)." if kind == "drift" else ""
        ch["explanation"] = f"Boundary at {ch['t']:g} s: {names} changed; accepted {how}.{drift}"
        changes.append(ch)
        segs.append((cur, seg_start_change[0]))
        seg_start_change[0] = ch["id"]
        cur = list(next_ids)

    for i in range(1, n):
        agg = mean_state([vecs[j] for j in cur])
        ca, comp_a = compat(agg, vecs[i])
        cp, comp_p = compat(vecs[i - 1], vecs[i])
        log.append({"interval": i, "compat_to_segment": None if ca is None else r(ca),
                    "compat_to_previous": None if cp is None else r(cp), "components_to_segment": comp_a,
                    "components_to_previous": comp_p})
        if ca is None:  # no evidence at all -> unknown, keep the segment going
            (pending if pending else cur).append(i)
            continue
        if pending:
            if ca >= P["tau_keep"]:
                suppressed.append({"t": summ[pending[0]]["start_s"], "type": "transient_state_change",
                                   "reason": f"did not persist: lasted {len(pending) * W:g} s, state returned "
                                             f"(compatibility {ca:.2f} >= {P['tau_keep']})",
                                   "intervals": pending,
                                   "changed": changed_attributes(agg, mean_state([vecs[j] for j in pending]), labels)})
                cur += pending + [i]
                pending, confirm = [], 0
                continue
            pending.append(i)
            confirm += ca < P["tau_change"]
            if confirm >= P["persist_intervals"]:
                drift = all((log[j - 1]["compat_to_previous"] or 0) >= P["tau_keep"] for j in pending)
                close(pending, "drift" if drift else "persistent", ca, cp, comp_a, comp_p)
                pending, confirm = [], 0
            elif len(pending) >= P["max_pending_intervals"]:
                suppressed.append({"t": summ[pending[0]]["start_s"], "type": "unconfirmed_change",
                                   "reason": "stayed in the hysteresis band without persisting", "intervals": pending})
                cur += pending
                pending, confirm = [], 0
            continue
        if ca < P["tau_change"]:
            major = [k for k in ("environment", "activity", "scene")
                     if comp_p.get(k) is not None and comp_p[k] < 0.4]
            if cp is not None and cp < P["tau_strong"] and ca < P["tau_strong"] and major:
                close([i], "strong_immediate", ca, cp, comp_a, comp_p)
            else:
                pending, confirm = [i], 1
                if P["persist_intervals"] <= 1:
                    close(pending, "persistent", ca, cp, comp_a, comp_p)
                    pending, confirm = [], 0
            continue
        cur.append(i)
        low = [k for k, v in comp_p.items() if v is not None and v < 0.5]
        if low:
            suppressed.append({"t": summ[i]["start_s"], "type": "outweighed_component_change",
                               "reason": f"{', '.join(low)} changed vs previous interval, but overall compatibility "
                                         f"with the segment stayed {ca:.2f} >= {P['tau_change']}",
                               "intervals": [i], "components_to_previous": comp_p})
    if pending:
        suppressed.append({"t": summ[pending[0]]["start_s"], "type": "unconfirmed_change_at_end",
                           "reason": "video ended before the change could persist", "intervals": pending})
        cur += pending
    segs.append((cur, seg_start_change[0]))
    return segs, changes, suppressed, log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video-id", required=True)
    ap.add_argument("--root", default=".")
    a = ap.parse_args()
    root = Path(a.root)
    l1, seg = root / "data" / "level1" / a.video_id, root / "data" / "segments" / a.video_id
    t0 = time.time()
    O = Obs(l1, seg)
    labels = {"environment": O.env_labels, "activity": O.act_labels, "people_count": P["people_labels"],
              "object_density": P["object_labels"], "movement": P["motion_labels"],
              "audio_state": P["audio_states"], "active_speakers": P["speaker_labels"]}
    W, n = P["interval_s"], int(np.ceil(O.duration / P["interval_s"]))
    bounds = [(i * W, min((i + 1) * W, O.duration)) for i in range(n)]
    if len(bounds) > 1 and bounds[-1][1] - bounds[-1][0] < W / 3:  # fold a short tail into the last interval
        bounds[-2] = (bounds[-2][0], bounds[-1][1])
        bounds.pop()
    summ, vecs = zip(*[O.summary(i, s, e) for i, (s, e) in enumerate(bounds)])
    summ, vecs = list(summ), list(vecs)
    t_sum = time.time() - t0
    segs, changes, suppressed, log = segment(summ, vecs, labels)
    t_seg = time.time() - t0 - t_sum
    for row in log:
        summ[row["interval"]]["compatibility"] = {k: v for k, v in row.items() if k != "interval"}
    # secondary events are suppressed by design: log them too
    for s in summ:
        for e in s["secondary"]["track_events"] + s["secondary"]["gestures"]:
            suppressed.append({"t": e["t"], "type": e["type"], "reason": "secondary observation (never boundary-driving)",
                               "intervals": [s["interval"]], "detail": e})
        for t in s["secondary"]["shot_cuts"]:
            suppressed.append({"t": t, "type": "shot_cut", "reason": "framing change is secondary", "intervals": [s["interval"]]})
        for o in s["secondary"]["objects_transient"]:
            suppressed.append({"t": s["start_s"], "type": "transient_object", "reason": "isolated/short-lived object detection",
                               "intervals": [s["interval"]], "detail": {"class": o, "frame_share": s["secondary"]["object_frame_share"][o]}})
    suppressed.sort(key=lambda x: x["t"])
    out_segs = []
    for k, (ids, start_change) in enumerate(segs):
        st = mean_state([vecs[j] for j in ids])
        internal = [compat(st, vecs[j])[0] for j in ids]
        internal = [x for x in internal if x is not None]
        secd = {key: [e for j in ids for e in summ[j]["secondary"][key]]
                for key in ("track_events", "gestures", "shot_cuts", "speaker_turns", "on_screen_text_changes")}
        objs = {}
        for j in ids:
            for c in summ[j]["secondary"]["object_frame_share"]:
                objs[c] = objs.get(c, 0) + 1
        out_segs.append({
            "segment": k, "start_s": summ[ids[0]]["start_s"], "end_s": summ[ids[-1]]["end_s"],
            "duration_s": r(summ[ids[-1]]["end_s"] - summ[ids[0]]["start_s"]), "intervals": ids,
            "state": {lab: top(st[key], labels[lab], 3) for key, lab in
                      (("environment", "environment"), ("activity", "activity"), ("people", "people_count"),
                       ("objects", "object_density"), ("movement", "movement"), ("audio", "audio_state"),
                       ("speakers", "active_speakers"))},
            "internal_compatibility": {"mean": r(np.mean(internal)), "min": r(np.min(internal))} if internal else None,
            "started_by_change": start_change,
            "objects_seen_in_n_intervals": dict(sorted(objs.items(), key=lambda x: -x[1])),
            "secondary_observations": secd,
        })
    l1.mkdir(parents=True, exist_ok=True)
    with open(l1 / "atomic_state_summaries.jsonl", "w", encoding="utf-8") as f:
        f.writelines(json.dumps(s, ensure_ascii=False) + "\n" for s in summ)
    with open(l1 / "state_changes.jsonl", "w", encoding="utf-8") as f:
        f.writelines(json.dumps(c, ensure_ascii=False) + "\n" for c in changes)
    with open(l1 / "suppressed_changes.jsonl", "w", encoding="utf-8") as f:
        f.writelines(json.dumps(c, ensure_ascii=False) + "\n" for c in suppressed)
    (l1 / "stable_state_segments.json").write_text(json.dumps({
        "video_id": a.video_id, "duration_s": r(O.duration), "n_intervals": len(summ), "n_segments": len(out_segs),
        "params": P, "detector_timing_s": O.meta.get("timing_s"),
        "aggregation_timing_s": {"summaries": r(t_sum), "segmentation": r(t_seg)},
        "segments": out_segs}, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"{len(summ)} intervals -> {len(out_segs)} stable-state segments, {len(changes)} boundaries, "
          f"{len(suppressed)} suppressed changes ({t_sum + t_seg:.2f} s)")
    for c in changes:
        print(" ", c["explanation"])


if __name__ == "__main__":
    main()
