"""Correction stage: filtered evidence -> weighted candidate changes -> candidate moments (deterministic).

Evidence types and default weights (configs/default.yaml: correction.weights), pooled within pool_s (each type
counted once, by its strongest item); a boundary needs pooled weight >= boundary_threshold (1.0):
  activity            1.0  new activity state persists >= persist_alone_s (6 s)                      (rule A)
  activity_short      0.6  new state persists >= persist_supported_s (4 s): needs environment/visual  (rule A)
  activity_family     0.5  change between variants of one activity (configured label families)
  observability       1.0  entering/leaving a black/blank span >= black_min_s
  environment         0.6  CLIP environment top-1 change persisting >= env_persist_frames
  visual_transition   0.4  CLIP frame discontinuity (cosine < visual_cos); a shot cut alone counts 0.2;
                           cut + discontinuity is ONE piece of evidence (0.4), not two              (rule E)
  people              0.4  person-count bin jump >= people_bin_jump between 6 s windows, away from cuts;
                      0.2  when a shot cut is within one step (tracker identity unreliable)          (rule C)
  audio               0.3  music or speech onset/offset
Uncertain spans never create boundaries (rule D); short certain runs survive as sub-events (rule F); activity
changes inside one moment are kept as evidence for Level 2 (rule G). Human overrides are applied first and
forced/removed boundaries take precedence over automatic decisions.
"""
from __future__ import annotations

import math
from collections import Counter
from pathlib import Path

import numpy as np

from ..utils.io import append_jsonl, read_json, read_jsonl, write_json, write_jsonl
from . import manual_overrides
from .observation_filter import NOPERSON, UNCERTAIN, UNOBS, step_state
from .temporal_consistency import certain_runs, denoise, runs


def r(x, n=3):
    return None if x is None else round(float(x), n)


def family(a, b, fams):
    return any(a in f and b in f for f in fams)


def activity_evidence(states, times, c, step):
    W = c["weights"]
    alone, supp = math.ceil(c["persist_alone_s"] / step), math.ceil(c["persist_supported_s"] / step)
    ev = []
    R = certain_runs(states, (UNCERTAIN,))
    for A, B in zip(R, R[1:]):
        (sa, a0, a1), (sb, b0, b1) = A[:3], B[:3]
        nb = b1 - b0 + 1
        t = times[b0] - step / 2
        gap = b0 - a1 - 1
        base = {"t": r(t), "from": sa, "to": sb, "from_span": [r(times[a0] - step / 2), r(times[a1] + step / 2)],
                "to_span": [r(times[b0] - step / 2), r(times[b1] + step / 2)], "new_state_persists_s": r(nb * step),
                "uncertain_gap_s": r(gap * step)}
        if UNOBS in (sa, sb):
            if (sb == UNOBS and nb * step >= c["black_min_s"]) or (sa == UNOBS and (a1 - a0 + 1) * step >= c["black_min_s"]):
                ev.append({**base, "type": "observability", "weight": W["observability"]})
            else:
                ev.append({**base, "type": "observability", "weight": 0.0, "note": "black span shorter than black_min_s"})
            continue
        if family(sa, sb, c["label_families"]):
            ev.append({**base, "type": "activity_family", "weight": W["activity_family"]})
        elif nb >= alone:
            ev.append({**base, "type": "activity", "weight": W["activity"]})
        elif nb >= supp:
            ev.append({**base, "type": "activity_short", "weight": W["activity_short"],
                       "note": "persists < persist_alone_s: needs environment/visual support"})
    return ev


def environment_evidence(obs, c, raw):
    W, ev = c["weights"], []
    clip = np.load(raw / "clip_v2.npz")
    meta = read_json(raw / "detect_v2_meta.json")
    ENV = meta["heads"]["environment"]
    ct, env, emb = clip["times"].astype(float), clip["env_masked"], clip["emb_masked"]
    black_t = {o["environment"]["frame_index"] for o in obs if o["black_frame"]}
    idx = [i for i in range(len(ct)) if i not in black_t]
    lab = {i: ENV[int(env[i].argmax())] for i in idx}
    n = c["env_persist_frames"]
    for j, (p, q) in enumerate(zip(idx, idx[1:])):
        if lab[q] != lab[p]:
            nxt = idx[j + 1:j + 1 + n]
            ok = len(nxt) == n and all(lab[x] == lab[q] for x in nxt)
            ev.append({"t": r((ct[p] + ct[q]) / 2), "type": "environment", "weight": W["environment"] if ok else 0.0,
                       "from": lab[p], "to": lab[q], "note": None if ok else f"did not persist {n} frames (noise)"})
    e = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    for p, q in zip(idx, idx[1:]):
        cos = float(e[p] @ e[q])
        if cos < c["visual_cos"]:
            ev.append({"t": r((ct[p] + ct[q]) / 2), "type": "visual_transition", "weight": W["visual_transition"],
                       "clip_cosine": r(cos), "frames_s": [r(ct[p]), r(ct[q])]})
    return ev


def people_audio_evidence(obs, c, step, cuts):
    W, ev = c["weights"], []
    edges = c["people_bin_edges"]
    bins = [None if o["people"]["count_median"] is None else int(np.searchsorted(edges, o["people"]["count_median"], side="right") - 1)
            for o in obs]
    m = max(1, int(round(c["people_window_s"] / step)))
    cand = []
    for k in range(m, len(obs) - m + 1):
        a = [b for b in bins[k - m:k] if b is not None]
        b_ = [b for b in bins[k:k + m] if b is not None]
        if a and b_ and abs(np.median(b_) - np.median(a)) >= c["people_bin_jump"]:
            before = {t for o in obs[k - m:k] for t in o["people"]["track_ids"]}
            after = {t for o in obs[k:k + m] for t in o["people"]["track_ids"]}
            cand.append((k, abs(np.median(b_) - np.median(a)), np.median(a), np.median(b_), before, after))
    for k, d, a, b_, before, after in cand:
        if all(d >= d2 for k2, d2, *_ in cand if abs(k2 - k) <= m):
            t = obs[k]["t"] - step / 2
            at_cut = any(abs(cu - t) <= step for cu in cuts)
            new_ids = sorted(after - before)
            ev.append({"t": r(t), "type": "people", "weight": W["people_at_cut"] if at_cut else W["people"],
                       "from_bin_median": r(a), "to_bin_median": r(b_), "new_track_ids": new_ids[:10],
                       "continuing_track_ids": sorted(after & before)[:10],
                       "note": "shot cut within one step: tracker identity unreliable" if at_cut else None})
    m = max(1, int(round(c["audio_window_s"] / step)))
    for key, get in (("music", lambda o: o["audio"]["events"]["music"] or 0), ("speech", lambda o: o["audio"]["speech_share"])):
        vals = [get(o) for o in obs]
        cand = []
        for k in range(m, len(obs) - m + 1):
            a, b_ = np.mean(vals[k - m:k]), np.mean(vals[k:k + m])
            if abs(b_ - a) >= c["audio_jump"]:
                cand.append((k, abs(b_ - a), a, b_))
        for k, d, a, b_ in cand:
            if all(d >= d2 for k2, d2, *_ in cand if abs(k2 - k) <= m):
                ev.append({"t": r(obs[k]["t"] - step / 2), "type": "audio", "weight": c["weights"]["audio"], "stream": key,
                           "before_share": r(a), "after_share": r(b_)})
    for cu in cuts:
        ev.append({"t": r(cu), "type": "visual_transition", "weight": 0.2, "shot_cut": True, "note": "cut alone"})
    return ev


def pool(ev, c, overrides, step):
    """Group evidence within pool_s; one item per type; human forced/removed boundaries override the decision."""
    groups = []
    for e in sorted(ev, key=lambda e: e["t"]):
        if e["weight"] <= 0:
            continue
        if groups and e["t"] - groups[-1]["first_t"] <= c["pool_s"]:
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
        anchor = max(best.values(), key=lambda e: (e["type"].startswith("activity") or e["type"] == "observability", e["weight"]))
        t = round(anchor["t"] / step) * step
        ok = score >= c["boundary_threshold"]
        parts = " + ".join(f"{k} {v['weight']:g}" for k, v in sorted(best.items()))
        out.append({"t": r(t), "score": r(score), "accepted": ok, "decided_by": "rules", "types": sorted(best),
                    "evidence": g["evidence"],
                    "reason": f"{'accepted' if ok else 'rejected'}: pooled evidence {score:.2f} "
                              f"{'>=' if ok else '<'} {c['boundary_threshold']:g} ({parts})"})
    for rm in overrides["removed_boundaries"]:
        for g in out:
            if abs(g["t"] - rm["t"]) <= rm.get("tolerance_s", 2.0) and g["accepted"]:
                g.update(accepted=False, decided_by="human", reason=f"removed by reviewer (edit {rm.get('edit_id')}); was: {g['reason']}")
    for fb in overrides["forced_boundaries"]:
        g = next((g for g in out if abs(g["t"] - fb["t"]) <= 1e-6), None)
        if g:
            g.update(accepted=True, decided_by="human", reason=f"forced by reviewer (edit {fb.get('edit_id')}); was: {g['reason']}")
        else:
            out.append({"t": r(fb["t"]), "score": None, "accepted": True, "decided_by": "human", "types": ["human"],
                        "evidence": [], "reason": f"added by reviewer (edit {fb.get('edit_id')})"})
    return sorted(out, key=lambda g: g["t"])


def build_moments(vid, obs, states, groups, dur, c, step):
    acc = {g["t"]: g for g in groups if g["accepted"]}
    cuts = sorted({0.0, *acc, r(dur)})
    out = []
    for i, (a, b) in enumerate(zip(cuts, cuts[1:])):
        if b <= a:
            continue
        idx = [k for k, o in enumerate(obs) if a <= o["t"] < b]
        st = [states[k] for k in idx]
        n = max(1, len(st))
        cnt = Counter(st)
        certain = {k: v for k, v in cnt.items() if k not in (UNCERTAIN, UNOBS)}
        dom, dom_n = max(certain.items(), key=lambda kv: kv[1]) if certain else (None, 0)
        unobs, unc = cnt.get(UNOBS, 0) / n, cnt.get(UNCERTAIN, 0) / n
        status = ("unobservable" if unobs >= 0.5 else
                  "ambiguous" if b - a < c["min_event_s"] or unc >= 0.5 or dom_n / n < 0.5 else "moment")
        subs = []
        for s, k0, k1 in runs(st):
            if s not in (UNCERTAIN, UNOBS) and (k1 - k0 + 1) * step >= c["min_event_s"]:
                subs.append({"state": s, "start_s": r(obs[idx[k0]]["t"] - step / 2), "end_s": r(obs[idx[k1]]["t"] + step / 2)})
        envs = Counter(obs[k]["environment"]["top3"][0][0] for k in idx if not obs[k]["black_frame"])
        rel = [obs[k]["activity"]["top1_rel"] for k in idx if states[k] == dom]
        best = max(idx, key=lambda k: obs[k]["activity"]["top1_rel"] if states[k] == dom else -1) if idx else None
        sg, eg = acc.get(a), acc.get(b)

        def bnd(g, t, what):
            return ({"t": g["t"], "score": g["score"], "types": g["types"], "decided_by": g["decided_by"], "reason": g["reason"]}
                    if g else {"t": r(t), "types": [what], "reason": f"video {what}"})

        out.append({
            "interval_id": f"c{i:03d}", "video_id": vid, "start_s": r(a), "end_s": r(b), "duration_s": r(b - a), "status": status,
            "activities": [{"state": k, "share_of_steps": r(v / n)} for k, v in cnt.most_common()],
            "sub_events": subs,
            "environment": [{"label": k, "share_of_steps": r(v / max(1, sum(envs.values())))} for k, v in envs.most_common(3)],
            "people_median": r(np.median([obs[k]["people"]["count_median"] for k in idx if obs[k]["people"]["count_median"] is not None]))
            if idx else None,
            "audio": {"speech_share": r(np.mean([obs[k]["audio"]["speech_share"] for k in idx])) if idx else None,
                      "music_share": r(np.mean([obs[k]["audio"]["events"]["music"] or 0 for k in idx])) if idx else None},
            "start_boundary": bnd(sg, a, "start"), "end_boundary": bnd(eg, b, "end"),
            "uncertainty": {"uncertain_share": r(unc), "unobservable_share": r(unobs), "dominant_state": dom,
                            "dominant_share": r(dom_n / n), "mean_top1_rel_dominant": r(np.mean(rel)) if rel else None,
                            "n_environment_labels": len(envs)},
            "representative_frames_s": sorted({r(a + 1), r((a + b) / 2), r(b - 1)} | ({obs[best]["t"]} if best is not None and dom else set())),
            "source_steps": [idx[0], idx[-1]] if idx else [],
        })
    return out


def run(cfg, entry, vdir: Path, log):
    c, step = cfg["correction"], cfg["level1"]["step_s"]
    vid = entry["video_id"]
    obs = read_jsonl(vdir / "level1" / "observations.jsonl")
    if not obs:
        raise RuntimeError("no Level-1 observations; run level1 first")
    raw = vdir / "level1" / "raw"
    out = vdir / "correction"
    out.mkdir(parents=True, exist_ok=True)
    clog = []
    overrides = manual_overrides.load(vdir)
    times = [o["t"] for o in obs]
    first = [step_state(o, c) for o in obs]
    states = manual_overrides.apply_activity([s for s, _ in first], times, overrides, clog)
    states = denoise(states, math.ceil(c["min_event_s"] / step), clog, times)
    states = manual_overrides.apply_activity(states, times, overrides, clog)  # human edits win over denoising
    cuts = [cu for o in obs for cu in o["shot_cuts"]]
    ev = activity_evidence(states, times, c, step) + environment_evidence(obs, c, raw) + people_audio_evidence(obs, c, step, cuts)
    groups = pool(ev, c, overrides, step)
    dur = float((read_json(vdir / "source" / "source_metadata.json") or {}).get("duration_s") or obs[-1]["end"])
    moments = build_moments(vid, obs, states, groups, dur, c, step)
    write_jsonl(out / "corrected_observations.jsonl", [
        {"step": o["step"], "t": o["t"], "start": o["start"], "end": o["end"], "rule_state": s0, "rule_reason": why,
         "corrected_state": s1, "changed": s0 != s1, "activity_top1": o["activity"]["top1"],
         "activity_top1_rel": o["activity"]["top1_rel"], "activity_margin_ratio": o["activity"]["top1_margin_ratio"],
         "environment_top1": o["environment"]["top3"][0][0], "people_median": o["people"]["count_median"],
         "track_ids": o["people"]["track_ids"], "speech_share": o["audio"]["speech_share"], "shot_cuts": o["shot_cuts"]}
        for o, (s0, why), s1 in zip(obs, first, states)])
    write_json(out / "candidate_moments.json", {"video_id": vid, "params": c, "n_intervals": len(moments),
                                                "status_counts": dict(Counter(m["status"] for m in moments)), "moments": moments})
    write_json(out / "boundary_evidence.json", {"video_id": vid, "pooled_candidates": groups, "all_evidence": ev,
                                                "overrides_applied": overrides})
    lp = out / "correction_log.jsonl"
    lp.unlink(missing_ok=True)
    for e in clog + [{"rule": "boundary_decision", "t": g["t"], "accepted": g["accepted"], "decided_by": g["decided_by"],
                      "detail": g["reason"]} for g in groups]:
        append_jsonl(lp, e)
    report(vid, moments, groups, clog, c, out / "correction_report.md")
    try:
        plot(obs, states, groups, moments, dur, vdir / "reports" / "correction_timeline.png")
    except Exception as e:  # noqa: BLE001
        log.warning(f"timeline plot failed: {e}")
    log.info(f"{len(moments)} candidate intervals ({dict(Counter(m['status'] for m in moments))}); "
             f"{sum(g['accepted'] for g in groups)}/{len(groups)} pooled candidates accepted")


def report(vid, M, G, clog, c, path):
    d = np.array([m["duration_s"] for m in M if m["status"] == "moment"]) if M else np.array([])
    acc, rej = [g for g in G if g["accepted"]], [g for g in G if not g["accepted"]]
    L = [f"# Correction report: {vid}", "",
         f"- {len(M)} candidate intervals: {dict(Counter(m['status'] for m in M))}",
         (f"- 'moment' durations (s): min {d.min():g}, median {np.median(d):g}, mean {d.mean():.1f}, max {d.max():g}" if len(d) else "- no moments"),
         f"- pooled candidate changes: {len(G)} ({len(acc)} accepted, {len(rej)} rejected); human decisions: "
         f"{sum(g['decided_by'] == 'human' for g in G)}",
         f"- accepted by evidence: {dict(Counter('+'.join(g['types']) for g in acc).most_common(10))}",
         f"- rejected by evidence: {dict(Counter('+'.join(g['types']) for g in rej).most_common(10))}",
         f"- noise-floor corrections: {sum(1 for x in clog if x['rule'] == 'noise_floor')}; human activity overrides: "
         f"{sum(1 for x in clog if x['rule'] == 'human_override')}",
         f"- parameters: persist alone {c['persist_alone_s']} s, supported {c['persist_supported_s']} s, threshold {c['boundary_threshold']}, "
         f"weights {c['weights']}", "",
         "| id | span (s) | dur | status | activities (share of 2 s steps) | sub-events | environment | start | end |",
         "|---|---|---:|---|---|---|---|---|---|"]
    for m in M:
        acts = ", ".join(f"{a['state']} {a['share_of_steps']}" for a in m["activities"][:3])
        subs = "; ".join(f"{s['state']} {s['start_s']:g}-{s['end_s']:g}" for s in m["sub_events"][:4])
        env = m["environment"][0]["label"] if m["environment"] else "-"
        L.append(f"| {m['interval_id']} | {m['start_s']:g}-{m['end_s']:g} | {m['duration_s']:g} | {m['status']} | {acts} | {subs} | {env} | "
                 f"{'+'.join(m['start_boundary']['types'])} | {'+'.join(m['end_boundary']['types'])} |")
    path.write_text("\n".join(L), encoding="utf-8")


def plot(obs, states, G, M, dur, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(3, 1, figsize=(16, 5.5), sharex=True, gridspec_kw={"height_ratios": [1.2, .5, 1]})
    labs = sorted(set(states))
    cm = plt.get_cmap("tab20")
    col = {s: ("#dddddd" if s == UNCERTAIN else "#444444" if s == UNOBS else "#bbbbff" if s == NOPERSON else cm(i % 20))
           for i, s in enumerate(labs)}
    for o, s in zip(obs, states):
        ax[0].axvspan(o["start"], o["end"], color=col[s], lw=0)
    ax[0].legend(handles=[plt.Rectangle((0, 0), 1, 1, color=col[s]) for s in labs], labels=labs, fontsize=6, ncol=6,
                 loc="upper center", bbox_to_anchor=(.5, 1.5))
    ax[0].set_ylabel("corrected\nactivity", rotation=0, ha="right", fontsize=8)
    ax[1].step([o["t"] for o in obs], [o["people"]["count_median"] or 0 for o in obs], where="mid", color="k", lw=.7)
    ax[1].set_ylabel("persons", rotation=0, ha="right", fontsize=8)
    sc = {"moment": "tab:green", "ambiguous": "tab:orange", "unobservable": "tab:gray"}
    for m in M:
        ax[2].axvspan(m["start_s"], m["end_s"], ymin=.55, ymax=.95, color=sc[m["status"]], alpha=.8, ec="white", lw=.5)
    for g in G:
        ax[2].plot([g["t"]], [.3], marker="v" if g["accepted"] else "x", color=("purple" if g["decided_by"] == "human" else
                   "black" if g["accepted"] else "red"), ms=4)
    ax[2].set_ylim(0, 1)
    ax[2].set_yticks([])
    ax[2].set_ylabel("candidate\nintervals", rotation=0, ha="right", fontsize=8)
    for a_ in ax:
        for g in G:
            if g["accepted"]:
                a_.axvline(g["t"], color="black", lw=.5)
    ax[-1].set_xlim(0, dur)
    ax[-1].xaxis.set_major_formatter(lambda v, _p=None: f"{int(v // 60)}:{int(v % 60):02d}")
    ax[-1].set_xlabel("green = moment, orange = ambiguous, grey = unobservable; v accepted, x rejected, purple = reviewer decision")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
