"""Semantic consolidation: for every candidate boundary the VLM answers merge / keep / uncertain.

One call per boundary (prompt l2-v2), sent to the model in batches. The call shows the LEFT and RIGHT candidate intervals with frames concentrated
just before and after the cut, plus one frame from the interval before LEFT and after RIGHT as context, so
neighbouring calls overlap in context. The model writes its explanation before the decision. (l2-v1 judged two
boundaries per call by ID; the model systematically mixed up the IDs - 81 of 84 cross-window conflicts had the same
order - so boundaries are now judged one at a time, described as left/right, never by ID.)

Reconciliation stays deterministic: if a boundary has several votes (e.g. a re-run added one) and they disagree,
the boundary becomes 'uncertain' with every vote kept, and is flagged for review. 'uncertain' boundaries are kept
(no merge). Boundaries set by a human reviewer are never sent to the VLM. Boundary times are always the existing
candidate times.
"""
from __future__ import annotations

from .context_builder import boundary_text, interval_text, transcript_excerpt, video_context
from .frame_sampler import grab, spread
from .output_validator import ask_batch, check_single

RULES = """You are helping build a video moment retrieval dataset from a long TV/community video.
The video has been cut automatically into candidate intervals. Look at ONE cut: the LEFT interval ends where the
RIGHT interval begins. Decide:
- "merge": LEFT and RIGHT are parts of the SAME coherent event (e.g. one performance that alternates singing,
  dancing and applause; one interview where the camera cuts between speakers or framings; one scene after a cut).
- "keep": the visible content or event genuinely changes at the cut (different activity or setting, a different
  segment of the programme, an inserted clip starting or ending, a clearly signalled new topic).
- "uncertain": the evidence is insufficient.
Do not merge only because the place or the people are the same. Do not split only because the camera framing
changes, someone gestures, or the automatic activity label changes. The detector evidence is automatic and can be
wrong: trust what you see in the frames and read in the transcript.
First write a one- or two-sentence explanation citing what is visible or said just before and just after the
cut (frame times or transcript), THEN the decision that follows from it. If your explanation describes a change of
activity, setting or segment, the decision must be "keep".
Answer with ONLY this JSON object on one line:
{"explanation": "<evidence before and after the cut>", "decision": "merge|keep|uncertain"}"""


def boundary_frames(m_left, m_right, c):
    t = m_right["start_s"]
    n, f = c["boundary_frames_each_side"], c["boundary_focus_s"]
    left = spread(max(m_left["start_s"], t - f), t, n)
    right = spread(t, min(m_right["end_s"], t + f), n)
    return left, right


def boundary_content(cfg, video, cands, i, b, transcript, vc):
    c = cfg["level2"]
    n = len(cands)
    L, R = cands[i - 1], cands[i]
    content = [{"type": "text", "text": RULES}]
    if vc:
        content.append({"type": "text", "text": vc})
    if i - 2 >= 0 and c["context_frames"]:
        P = cands[i - 2]
        content.append({"type": "text", "text": "Context only - " + interval_text(P, "BEFORE")})
        for t, img in grab(video, spread(P["start_s"], P["end_s"], c["context_frames"]), c["frame_max_side"]):
            content += [{"type": "text", "text": f"BEFORE frame at {t:.1f} s:"}, {"type": "image", "image": img, "t": t}]
    lt, rt = boundary_frames(L, R, c)
    content.append({"type": "text", "text": interval_text(L, "LEFT")})
    for t, img in grab(video, lt, c["frame_max_side"]):
        content += [{"type": "text", "text": f"LEFT frame at {t:.1f} s:"}, {"type": "image", "image": img, "t": t}]
    content.append({"type": "text", "text": f"--- THE CUT IS AT {b['t']:g} s ---\n" + boundary_text("This cut", b["t"], b["evidence"])})
    content.append({"type": "text", "text": interval_text(R, "RIGHT")})
    for t, img in grab(video, rt, c["frame_max_side"]):
        content += [{"type": "text", "text": f"RIGHT frame at {t:.1f} s:"}, {"type": "image", "image": img, "t": t}]
    if i + 1 < n and c["context_frames"]:
        N = cands[i + 1]
        content.append({"type": "text", "text": "Context only - " + interval_text(N, "AFTER")})
        for t, img in grab(video, spread(N["start_s"], N["end_s"], c["context_frames"]), c["frame_max_side"]):
            content += [{"type": "text", "text": f"AFTER frame at {t:.1f} s:"}, {"type": "image", "image": img, "t": t}]
    content.append({"type": "text", "text": "Transcript around the cut (automatic captions, may contain errors): "
                    + transcript_excerpt(transcript, b["t"] - 20, b["t"] + 20, c["transcript_chars"] // 2)
                    + f"\nDecide whether LEFT and RIGHT belong to the same event across the cut at {b['t']:g} s."})
    return content


def decide(vlm, cfg, video, cands, transcript, src, log):
    """One call per boundary; all boundaries of the video are sent as batches (level2.batch_size per batch)."""
    c = cfg["level2"]
    n = len(cands)
    bounds = {i: {"id": f"B{i}", "t": cands[i]["start_s"], "left": cands[i - 1]["interval_id"],
                  "right": cands[i]["interval_id"], "evidence": cands[i]["start_boundary"]} for i in range(1, n)}
    votes = {i: [] for i in bounds}
    audit = []
    vc = video_context(src, c["use_video_context"])
    todo = [i for i, b in bounds.items() if b["evidence"].get("decided_by") != "human"]
    bs = int(c.get("batch_size", 64))
    for j in range(0, len(todo), bs):  # frames are decoded per chunk to bound memory
        chunk = todo[j:j + bs]
        items = [(boundary_content(cfg, video, cands, i, bounds[i], transcript, vc), c["max_new_tokens_merge"],
                  f"merge_{bounds[i]['id']}") for i in chunk]
        for i, (content, _, _), (obj, attempts) in zip(chunk, items, ask_batch(vlm, items, check_single, c["retries"])):
            b = bounds[i]
            audit.append({"boundary": b["id"], "t": b["t"], "frames": [x.get("t") for x in content if x["type"] == "image"],
                          "attempts": attempts, "valid": obj is not None})
            if obj is None:
                log.warning(f"{b['id']}: no valid decision after {len(attempts)} attempts")
                votes[i].append({"call": b["id"], "decision": "uncertain", "explanation": "no valid model output", "invalid": True})
            else:
                votes[i].append({"call": b["id"], "decision": obj["decision"], "explanation": obj["explanation"].strip()})
    return reconcile(bounds, votes), audit


def reconcile(bounds, votes):
    final = []
    for i, b in bounds.items():
        if b["evidence"].get("decided_by") == "human":
            final.append({**_b(b), "decision": "keep", "decided_by": "human", "votes": [], "conflict": False,
                          "explanation": b["evidence"].get("reason"), "needs_review": False})
            continue
        vs = votes.get(i, [])
        ds = {v["decision"] for v in vs}
        if len(ds) == 1:
            d = next(iter(ds))
            final.append({**_b(b), "decision": d, "decided_by": "vlm", "votes": vs, "conflict": False,
                          "explanation": vs[0]["explanation"], "needs_review": d == "uncertain"})
        else:
            final.append({**_b(b), "decision": "uncertain", "decided_by": "reconciliation", "votes": vs, "conflict": bool(vs),
                          "explanation": "votes disagree: " + " | ".join(f"{v['decision']}: {v['explanation'][:120]}" for v in vs),
                          "needs_review": True})
    return final


def _b(b):
    return {"boundary": b["id"], "t": b["t"], "left_interval": b["left"], "right_interval": b["right"],
            "automatic_evidence": {k: b["evidence"].get(k) for k in ("types", "score", "reason", "decided_by")}}


def consolidate(vid, cands, decisions):
    """Group consecutive candidate intervals across 'merge' boundaries; keep/uncertain stay boundaries."""
    dmap = {d["right_interval"]: d for d in decisions}
    groups = [[0]]
    for i in range(1, len(cands)):
        d = dmap.get(cands[i]["interval_id"])
        if d and d["decision"] == "merge":
            groups[-1].append(i)
        else:
            groups.append([i])
    out = []
    for k, g in enumerate(groups):
        sd = dmap.get(cands[g[0]]["interval_id"])
        ed = dmap.get(cands[g[-1] + 1]["interval_id"]) if g[-1] + 1 < len(cands) else None
        out.append({"moment_id": f"{vid}_m{k:03d}", "video_id": vid, "start_s": cands[g[0]]["start_s"],
                    "end_s": cands[g[-1]]["end_s"], "duration_s": round(cands[g[-1]]["end_s"] - cands[g[0]]["start_s"], 3),
                    "source_intervals": [cands[i]["interval_id"] for i in g],
                    "merged_boundaries": [dmap[cands[i]["interval_id"]]["boundary"] for i in g[1:]],
                    "start_boundary": _edge(sd, "video start"), "end_boundary": _edge(ed, "video end"),
                    "needs_review": any(d and d["needs_review"] for d in (sd, ed))})
    return out


def _edge(d, default):
    if d is None:
        return {"decision": default}
    return {"boundary": d["boundary"], "t": d["t"], "decision": d["decision"], "decided_by": d["decided_by"],
            "explanation": d["explanation"], "automatic_evidence": d["automatic_evidence"]}
