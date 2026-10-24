"""Semantic consolidation: for every candidate boundary the VLM answers merge / keep / uncertain.

Overlapping windows of `window_intervals` consecutive candidate intervals (stride 1) each judge the boundaries inside
them, so interior boundaries are judged twice. Reconciliation is deterministic: identical decisions are accepted;
any disagreement -> 'uncertain' (both kept in the audit log) and flagged for human review. 'uncertain' boundaries are
kept (no merge). Boundaries set by a human reviewer are never sent to the VLM. Boundary times are always the
existing candidate times (no new timestamps are invented).
"""
from __future__ import annotations

from .context_builder import boundary_text, interval_text, transcript_excerpt, video_context
from .frame_sampler import grab, spread
from .output_validator import ask, check_merge

RULES = """You are helping build a video moment retrieval dataset from a long TV/community video.
The video has been cut automatically into consecutive candidate intervals. For each listed boundary between two
neighbouring intervals, decide:
- "merge": both sides are parts of the SAME coherent event (e.g. one performance that alternates singing, dancing
  and applause; one interview where the camera cuts between speakers or framings; one scene continuing after a cut).
- "keep": the visible content or event genuinely changes (different activity, setting, segment of the programme,
  inserted clip, new topic of conversation clearly signalled).
- "uncertain": the evidence is insufficient to decide.
Do not merge only because the place or the people are the same. Do not split only because the camera framing
changes, someone gestures, or the automatic activity label changes. The detector evidence is automatic and can be
wrong: trust what you see in the frames and read in the transcript. Base every explanation on visible or spoken
evidence (cite frame times or transcript), in one or two sentences.
Answer with ONLY this JSON object:
{"decisions": [{"boundary": "<boundary id>", "decision": "merge|keep|uncertain", "explanation": "<evidence>"}]}"""


def windows(n: int, w: int):
    if n <= 1:
        return []
    if n <= w:
        return [list(range(n))]
    return [list(range(i, i + w)) for i in range(n - w + 1)]


def decide(vlm, cfg, video, cands, transcript, src, log):
    c = cfg["level2"]
    n = len(cands)
    bounds = {i: {"id": f"B{i}", "t": cands[i]["start_s"], "left": cands[i - 1]["interval_id"],
                  "right": cands[i]["interval_id"], "evidence": cands[i]["start_boundary"]} for i in range(1, n)}
    votes = {i: [] for i in bounds}
    audit = []
    for k, win in enumerate(windows(n, c["window_intervals"])):
        bids = [i for i in win[1:] if bounds[i]["evidence"].get("decided_by") != "human"]
        if not bids:
            continue
        content = [{"type": "text", "text": RULES}]
        vc = video_context(src, c["use_video_context"])
        if vc:
            content.append({"type": "text", "text": vc})
        for j, i in enumerate(win):
            m = cands[i]
            label = chr(ord("A") + j)
            content.append({"type": "text", "text": interval_text(m, label)})
            for t, img in grab(video, spread(m["start_s"], m["end_s"], c["frames_per_interval"]), c["frame_max_side"]):
                content.append({"type": "text", "text": f"{label} frame at {t:.1f} s:"})
                content.append({"type": "image", "image": img, "t": t})
        for i in bids:
            b = bounds[i]
            content.append({"type": "text", "text": boundary_text(b["id"], b["t"], b["evidence"]) +
                            f"\n  Transcript around {b['id']} (automatic captions, may contain errors): "
                            + transcript_excerpt(transcript, b["t"] - 20, b["t"] + 20, c["transcript_chars"] // 2)})
        content.append({"type": "text", "text": f"Decide boundaries: {', '.join(bounds[i]['id'] for i in bids)}."})
        obj, attempts = ask(vlm, content, c["max_new_tokens_merge"],
                            lambda o, ids=[bounds[i]["id"] for i in bids]: check_merge(o, ids), c["retries"],
                            f"merge_w{k:03d}")
        rec = {"window": k, "intervals": [cands[i]["interval_id"] for i in win], "boundaries": [bounds[i]["id"] for i in bids],
               "frames": [x.get("t") for x in content if x["type"] == "image"], "attempts": attempts, "valid": obj is not None}
        audit.append(rec)
        if obj is None:
            log.warning(f"window {k}: no valid decision after {len(attempts)} attempts")
            for i in bids:
                votes[i].append({"window": k, "decision": "uncertain", "explanation": "no valid model output", "invalid": True})
            continue
        for d in obj["decisions"]:
            i = int(d["boundary"][1:])
            votes[i].append({"window": k, "decision": d["decision"], "explanation": d["explanation"].strip()})
    final = []
    for i, b in bounds.items():
        if b["evidence"].get("decided_by") == "human":
            final.append({**_b(b), "decision": "keep", "decided_by": "human", "votes": [], "conflict": False,
                          "explanation": b["evidence"].get("reason"), "needs_review": False})
            continue
        vs = votes[i]
        ds = {v["decision"] for v in vs}
        if len(ds) == 1:
            d = ds.pop()
            final.append({**_b(b), "decision": d, "decided_by": "vlm", "votes": vs, "conflict": False,
                          "explanation": vs[0]["explanation"], "needs_review": d == "uncertain"})
        else:
            final.append({**_b(b), "decision": "uncertain", "decided_by": "reconciliation", "votes": vs, "conflict": bool(vs),
                          "explanation": "windows disagree: " + " | ".join(f"w{v['window']}: {v['decision']}" for v in vs),
                          "needs_review": True})
    return final, audit


def _b(b):
    return {"boundary": b["id"], "t": b["t"], "left_interval": b["left"], "right_interval": b["right"],
            "automatic_evidence": {k: b["evidence"].get(k) for k in ("types", "score", "reason", "decided_by")}}


def consolidate(vid, cands, decisions):
    """Group consecutive candidate intervals across 'merge' boundaries; keep/uncertain stay boundaries."""
    dmap = {d["right_interval"]: d for d in decisions}
    groups = [[cands[0]]]
    for m in cands[1:]:
        d = dmap.get(m["interval_id"])
        if d and d["decision"] == "merge":
            groups[-1].append(m)
        else:
            groups.append([m])
    out = []
    for k, g in enumerate(groups):
        sd, ed = dmap.get(g[0]["interval_id"]), dmap.get(cands[cands.index(g[-1]) + 1]["interval_id"]) if cands.index(g[-1]) + 1 < len(cands) else None
        out.append({"moment_id": f"{vid}_m{k:03d}", "video_id": vid, "start_s": g[0]["start_s"], "end_s": g[-1]["end_s"],
                    "duration_s": round(g[-1]["end_s"] - g[0]["start_s"], 3),
                    "source_intervals": [m["interval_id"] for m in g],
                    "merged_boundaries": [dmap[m["interval_id"]]["boundary"] for m in g[1:]],
                    "start_boundary": _edge(sd, "video start"), "end_boundary": _edge(ed, "video end"),
                    "needs_review": any(d and d["needs_review"] for d in (sd, ed))})
    return out


def _edge(d, default):
    if d is None:
        return {"decision": default}
    return {"boundary": d["boundary"], "t": d["t"], "decision": d["decision"], "decided_by": d["decided_by"],
            "explanation": d["explanation"], "automatic_evidence": d["automatic_evidence"]}
