"""Diagnostics for a Level-1 aggregation run (read-only; never rewrites the aggregation outputs).

    python -m level1.diagnose --video-id detvde-Chocolate_Ballerina_Founder_Chane_5b4f44

Writes data/level1/<id>/diagnostics/: diagnostic_report.md, interval_states.csv, suppressed_by_reason.csv,
timeline_diagnostics.png. Needs numpy + matplotlib only.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ATTRS = ["environment", "activity", "people_count", "object_density", "movement", "audio_state", "active_speakers"]
CANDIDATE = {"transient_state_change": "insufficient persistence (state returned)",
             "unconfirmed_change": "insufficient persistence (stayed in hysteresis band)",
             "unconfirmed_change_at_end": "insufficient persistence (video ended)"}
FEATURE = {"outweighed_component_change": "weak overall state difference (one component changed)",
           "shot_cut": "camera-only change (shot cut)",
           "transient_object": "object-count fluctuation (short-lived object class)",
           "isolated_person_detection": "detection noise (person seen < 1 s)",
           "person_arrival": "person arrival (secondary)", "person_departure": "person departure (secondary)",
           "temporary_occlusion": "temporary occlusion (secondary)", "hand_raised": "gesture (secondary)"}
DANCE_ACT = ["people dancing", "people performing music"]
DANCE_ENV = ["a stage or auditorium", "a gym or dance studio"]


def jl(p):
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def short(s, n=22):
    s = s or "-"
    for a in ("a video frame of ", "a ", "an ", "people "):
        s = s[len(a):] if s.startswith(a) else s
    return s if len(s) <= n else s[:n - 1] + "…"


def scene_classes(emb, thr=0.85):
    """Deterministic leader clustering of interval CLIP embeddings (time order)."""
    cents, lab = [], []
    for e in emb:
        if e is None:
            lab.append(-1)
            continue
        sims = [float(c @ e) for c in cents]
        if sims and max(sims) >= thr:
            k = int(np.argmax(sims))
            lab.append(k)
        else:
            cents.append(e)
            lab.append(len(cents) - 1)
    return lab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video-id", required=True)
    ap.add_argument("--root", default=".")
    a = ap.parse_args()
    d = Path(a.root) / "data" / "level1" / a.video_id
    out = d / "diagnostics"
    out.mkdir(exist_ok=True)
    S = jl(d / "atomic_state_summaries.jsonl")
    C = jl(d / "state_changes.jsonl")
    X = jl(d / "suppressed_changes.jsonl")
    G = json.loads((d / "stable_state_segments.json").read_text(encoding="utf-8"))
    meta = json.loads((d / "detect_meta.json").read_text(encoding="utf-8"))
    P = G["params"]
    acts, envs = meta["activities"], meta["environments"]
    xc = dict(np.load(d / "xclip_actions.npz")) if (d / "xclip_actions.npz").exists() else None
    ce = dict(np.load(d / "clip_env.npz")) if (d / "clip_env.npz").exists() else None
    n = len(S)
    seg_of = {i: g["segment"] for g in G["segments"] for i in g["intervals"]}

    # ---- 1. per-interval dominant states (+ scene class + speaker)
    emb = []
    for s in S:
        if ce is None:
            emb.append(None)
            continue
        m = (ce["times"] >= s["start_s"]) & (ce["times"] < s["end_s"])
        e = ce["emb"][m].mean(0) if m.any() else None
        emb.append(None if e is None else e / np.linalg.norm(e))
    scene = scene_classes(emb)
    rows = []
    for i, s in enumerate(S):
        pr, sec = s["primary"], s["secondary"]
        talk = sec.get("speaker_talk_s") or {}
        row = {"interval": i, "start_s": s["start_s"], "end_s": s["end_s"], "segment": seg_of.get(i)}
        for k in ATTRS:
            row[k] = pr[k].get("dominant", "unknown")
            row[k + "_p"] = pr[k].get("p", "")
        row["scene_class"] = f"S{scene[i]}" if scene[i] >= 0 else "unknown"
        row["dominant_speaker"] = max(talk, key=talk.get) if talk else "none"
        row["speakers_talk_s"] = " ".join(f"{k}:{v:g}" for k, v in talk.items())
        dist = pr["activity"].get("distribution", {})
        row["p_dancing"] = round(dist.get("people dancing", 0), 3)
        row["p_dance_or_music"] = round(sum(dist.get(x, 0) for x in DANCE_ACT), 3)
        edist = pr["environment"].get("distribution", {})
        row["p_stage_or_dance_env"] = round(sum(edist.get(x, 0) for x in DANCE_ENV), 3)
        c = s.get("compatibility") or {}
        row["compat_to_segment"] = c.get("compat_to_segment")
        row["compat_to_previous"] = c.get("compat_to_previous")
        for k, v in (c.get("components_to_segment") or {}).items():
            row[f"seg_{k}"] = v
        row["shot_cuts"] = len(sec["shot_cuts"])
        rows.append(row)
    with open(out / "interval_states.csv", "w", newline="", encoding="utf-8") as f:
        keys = []
        for r in rows:
            keys += [k for k in r if k not in keys]
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    # ---- 2. suppressed changes grouped by reason; candidate vs feature-level
    groups = defaultdict(list)
    for x in X:
        groups[CANDIDATE.get(x["type"]) or FEATURE.get(x["type"]) or x["type"]].append(x)
    n_cand = sum(len(v) for k, v in groups.items() if k in CANDIDATE.values())
    with open(out / "suppressed_by_reason.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["reason", "level", "type", "t", "intervals", "detail"])
        for k, v in sorted(groups.items()):
            for x in v:
                lvl = "candidate_boundary" if x["type"] in CANDIDATE else "feature_level"
                w.writerow([k, lvl, x["type"], x["t"], " ".join(map(str, x.get("intervals", []))),
                            json.dumps(x.get("detail") or x.get("changed") or x.get("components_to_previous") or "")[:300]])

    # ---- 3. suppressed changes persisting >= 2 intervals; dominant-label runs differing from the segment
    long_supp = [x for x in X if x["type"] in CANDIDATE and len(x.get("intervals", [])) >= 2]
    runs = []
    for g in G["segments"]:
        ids = g["intervals"]
        for k in ATTRS + ["scene_class"]:
            seg_dom = Counter(rows[i][k] for i in ids).most_common(1)[0][0]
            j = 0
            while j < len(ids):
                lab = rows[ids[j]][k]
                e = j
                while e + 1 < len(ids) and rows[ids[e + 1]][k] == lab:
                    e += 1
                if lab != seg_dom and lab != "unknown" and e - j + 1 >= 2:
                    runs.append({"segment": g["segment"], "attribute": k, "label": lab, "segment_dominant": seg_dom,
                                 "start_s": rows[ids[j]]["start_s"], "end_s": rows[ids[e]]["end_s"],
                                 "n_intervals": e - j + 1,
                                 "compat_to_segment": [rows[i]["compat_to_segment"] for i in ids[j:e + 1]]})
                j = e + 1

    # ---- 4. dance / performance evidence
    dance = [r for r in rows if r["activity"] in DANCE_ACT or r["p_dance_or_music"] >= 0.2
             or r["p_stage_or_dance_env"] >= 0.3 or r["environment"] in DANCE_ENV]
    clip_dance = []
    if xc is not None:
        j = acts.index("people dancing")
        top = xc["probs"].argmax(1)
        clip_dance = [(float(t), float(p[j])) for t, p, tp in zip(xc["times"], xc["probs"], top) if acts[tp] == "people dancing"]

    # ---- 6. within-interval conflicts (clip-level and frame-level dominants)
    conflicts = []
    for i, s in enumerate(S):
        info = {}
        if xc is not None:
            m = (xc["times"] >= s["start_s"]) & (xc["times"] < s["end_s"])
            if m.any():
                cnt = Counter(acts[k] for k in xc["probs"][m].argmax(1))
                info["activity_clip_dominants"] = dict(cnt)
        if ce is not None:
            m = (ce["times"] >= s["start_s"]) & (ce["times"] < s["end_s"])
            if m.any():
                info["environment_frame_dominants"] = dict(Counter(envs[k] for k in ce["probs"][m].argmax(1)))
        ad = s["primary"]["audio_state"].get("distribution", {})
        multi_act = len([v for v in info.get("activity_clip_dominants", {}).values() if v >= 2]) >= 2
        multi_env = len(info.get("environment_frame_dominants", {})) >= 2
        multi_aud = len([v for v in ad.values() if v >= 0.25]) >= 2
        if multi_act or multi_env or multi_aud:
            conflicts.append({"interval": i, "start_s": s["start_s"], "activity": multi_act, "environment": multi_env,
                              "audio": multi_aud, **info, "audio_distribution": ad,
                              "collapsed_to": {k: rows[i][k] for k in ("activity", "environment", "audio_state")}})

    # ---- 7. timeline plot
    plot(rows, C, X, P, out / "timeline_diagnostics.png", xc, acts)

    # ---- report
    btimes = ", ".join(f"{c['t']:g}s" for c in C)
    L = [f"# Level-1 diagnostics: {a.video_id}", "",
         f"{n} intervals, {len(G['segments'])} segments, {len(C)} accepted boundaries "
         f"({btimes}), {len(X)} suppressed records.", "",
         "## Suppressed records: candidate boundaries vs feature-level changes", "",
         f"- **Candidate segment boundaries** (segment compatibility fell below {P['tau_change']}, then did not persist): **{n_cand}**",
         f"- **Feature-level / secondary observations** (never boundary candidates): **{len(X) - n_cand}**", "",
         "| Reason | Level | Count |", "|---|---|---:|"]
    for k, v in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        L.append(f"| {k} | {'candidate' if k in CANDIDATE.values() else 'feature'} | {len(v)} |")
    L += ["", "No suppression category uses detector confidence: low-confidence gating is not implemented, "
          "so 'low confidence' has 0 records by construction.", "",
          "## Accepted boundaries", ""]
    for c in C:
        i = next(k for k, r in enumerate(rows) if r["start_s"] == c["t"])
        L.append(f"### {c['t']:g} s ({c['type']}, persisted {c['persisted_s']:g} s)")
        L.append(f"compat to segment {c['compat_to_segment']}, to previous {c['compat_to_previous_interval']}; "
                 f"components vs segment {c['components_to_segment']}")
        L += ["", "| interval | t | env | activity | people | audio | scene | speaker | compat seg |", "|---|---|---|---|---|---|---|---|---|"]
        for r in rows[max(0, i - 3):min(n, i + 3)]:
            L.append(f"| {r['interval']} | {r['start_s']:g} | {short(r['environment'])} | {short(r['activity'])} | "
                     f"{r['people_count']} | {r['audio_state']} | {r['scene_class']} | {r['dominant_speaker']} | {r['compat_to_segment']} |")
        L += ["", "Changed attributes: " + "; ".join(
            f"{x['attribute']} {x.get('from', '')} -> {x.get('to', '')} (sim {x.get('similarity', x.get('clip_cosine'))})"
            for x in c["changed"]), ""]
    L += ["## Candidate changes suppressed after persisting >= 2 intervals", ""]
    L += [f"- {x['t']:g} s, {x['type']}, intervals {x['intervals']}: {x['reason']}" for x in long_supp] or ["- none"]
    L += ["", "## Dominant-label runs (>= 2 intervals) that differ from their segment's dominant label", "",
          "| seg | attribute | label | segment dominant | span | n | compat to segment |", "|---|---|---|---|---|---:|---|"]
    L += [f"| {x['segment']} | {x['attribute']} | {short(x['label'], 30)} | {short(x['segment_dominant'], 30)} | "
          f"{x['start_s']:g}-{x['end_s']:g} | {x['n_intervals']} | {x['compat_to_segment']} |" for x in runs] or ["| none |"]
    L += ["", "## Dance / performance evidence", "",
          f"X-CLIP clips whose top label is 'people dancing': {len(clip_dance)}"
          + (f" (times: {', '.join(f'{t:.0f}' for t, _ in clip_dance[:60])})" if clip_dance else ""), "",
          "| interval | t | segment | activity | p dance+music | env | p stage/dance env | people | scene | compat seg | components vs segment |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in dance:
        comp = {k[4:]: r[k] for k in r if k.startswith("seg_")}
        L.append(f"| {r['interval']} | {r['start_s']:g} | {r['segment']} | {short(r['activity'])} | {r['p_dance_or_music']} | "
                 f"{short(r['environment'])} | {r['p_stage_or_dance_env']} | {r['people_count']} | {r['scene_class']} | "
                 f"{r['compat_to_segment']} | {comp} |")
    if not dance:
        L.append("| none | | | | | | | | | | |")
    L += ["", f"## Intervals with conflicting states collapsed into one label ({len(conflicts)} of {n})", "",
          "| interval | t | conflict in | activity clip tops | environment frame tops | audio dist | collapsed to |",
          "|---|---|---|---|---|---|---|"]
    for c in conflicts:
        L.append(f"| {c['interval']} | {c['start_s']:g} | {', '.join(k for k in ('activity', 'environment', 'audio') if c[k])} | "
                 f"{ {short(k): v for k, v in c.get('activity_clip_dominants', {}).items()} } | "
                 f"{ {short(k): v for k, v in c.get('environment_frame_dominants', {}).items()} } | {c['audio_distribution']} | "
                 f"{ {k: short(v) for k, v in c['collapsed_to'].items()} } |")
    L += ["", "## Per-interval dominant states", "",
          "| i | t | seg | environment | activity | people | objects | movement | audio | speakers | spk | scene | compat seg | compat prev |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['interval']} | {r['start_s']:g} | {r['segment']} | {short(r['environment'])} ({r['environment_p']}) | "
                 f"{short(r['activity'])} ({r['activity_p']}) | {r['people_count']} | {r['object_density']} | {r['movement']} | "
                 f"{r['audio_state']} | {r['active_speakers']} | {r['dominant_speaker']} | {r['scene_class']} | "
                 f"{r['compat_to_segment']} | {r['compat_to_previous']} |")
    (out / "diagnostic_report.md").write_text("\n".join(L), encoding="utf-8")
    print(f"wrote {out}: candidates {n_cand}, feature-level {len(X) - n_cand}, long suppressed {len(long_supp)}, "
          f"off-dominant runs {len(runs)}, dance rows {len(dance)}, conflict intervals {len(conflicts)}")


def plot(rows, C, X, P, path, xc, acts):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tracks = ["environment", "activity", "people_count", "audio_state", "scene_class", "dominant_speaker"]
    fig, axes = plt.subplots(len(tracks) + 2, 1, figsize=(16, 2.0 + 0.9 * (len(tracks) + 2) + 1.5), sharex=True,
                             gridspec_kw={"height_ratios": [1] * len(tracks) + [1.6, 1.6]})
    cmap = plt.get_cmap("tab20")
    T = rows[-1]["end_s"]
    for ax, k in zip(axes, tracks):
        labs = sorted({r[k] for r in rows})
        col = {l: cmap(i % 20) for i, l in enumerate(labs)}
        for r in rows:
            ax.axvspan(r["start_s"], r["end_s"], color=col[r[k]], lw=0)
        prev = None
        for r in rows:
            if r[k] != prev:
                ax.text(r["start_s"] + 1, 0.5, short(r[k], 16), fontsize=6, va="center", rotation=90)
            prev = r[k]
        ax.set_yticks([])
        ax.set_ylabel(k.replace("_", " "), rotation=0, ha="right", va="center", fontsize=8)
    ax = axes[len(tracks)]
    t = [(r["start_s"] + r["end_s"]) / 2 for r in rows]
    ax.plot(t, [r["compat_to_segment"] if r["compat_to_segment"] is not None else np.nan for r in rows], "-o", ms=3,
            label="compat to segment aggregate")
    ax.plot(t, [r["compat_to_previous"] if r["compat_to_previous"] is not None else np.nan for r in rows], "-", lw=1,
            alpha=0.7, label="compat to previous interval")
    for v, ls in ((P["tau_change"], "--"), (P["tau_keep"], ":"), (P["tau_strong"], "-.")):
        ax.axhline(v, color="grey", ls=ls, lw=0.8)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("compatibility", rotation=0, ha="right", va="center", fontsize=8)
    ax.legend(fontsize=7, loc="lower left")
    ax = axes[-1]
    if xc is not None:
        for name in ("people dancing", "people performing music", "people talking while seated"):
            if name in acts:
                ax.plot(xc["times"], xc["probs"][:, acts.index(name)], lw=0.8, label=f"X-CLIP p({name})")
        ax.legend(fontsize=7, loc="upper left")
    ax.set_ylabel("activity\nprobability", rotation=0, ha="right", va="center", fontsize=8)
    for a in axes:
        for c in C:
            a.axvline(c["t"], color="black", lw=2)
        for x in X:
            if x["type"] in CANDIDATE:
                a.axvline(x["t"], color="red", lw=1, ls="--")
    axes[-1].set_xlim(0, T)
    axes[-1].xaxis.set_major_formatter(lambda v, _p=None: f"{int(v // 60)}:{int(v % 60):02d}")
    axes[-1].set_xlabel("time (m:ss). Black = accepted boundary, red dashed = suppressed candidate boundary. "
                        "Grey lines: tau_change (--), tau_keep (:), tau_strong (-.)")
    axes[0].set_title("Level-1 atomic states (15 s intervals)", loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
