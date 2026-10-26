"""Run-length helpers and the minimal noise rule for 2 s sampling.

A state needs >= min_event_s (2 steps = 4 s) to count as an observed activity. A shorter run between two runs of
the same state is bridged (the activity continued); any other shorter run becomes 'uncertain' (detection noise).
Uncertain/unknown spans never create boundaries by themselves (rule D)."""
from __future__ import annotations

from .observation_filter import UNCERTAIN, UNOBS


def runs(states):
    out = []
    for k, s in enumerate(states):
        if out and out[-1][0] == s:
            out[-1][2] = k
        else:
            out.append([s, k, k])
    return out


def denoise(states, min_steps, log, times):
    st = list(states)
    R = runs(st)
    for i, (s, a, b) in enumerate(R):
        if b - a + 1 >= min_steps or s in (UNCERTAIN, UNOBS):
            continue
        prev_s = R[i - 1][0] if i else None
        next_s = R[i + 1][0] if i + 1 < len(R) else None
        new = prev_s if prev_s is not None and prev_s == next_s else UNCERTAIN
        for k in range(a, b + 1):
            st[k] = new
        log.append({"rule": "noise_floor", "t": times[a], "steps": b - a + 1, "from": s, "to": new,
                    "detail": "bridged between identical states" if new == prev_s else "run shorter than the noise floor"})
    return st


def certain_runs(states, uncertain=(UNCERTAIN,)):
    """Runs of certain states; uncertain gaps between two runs of the same state are bridged (rule D)."""
    R = [r for r in runs(states) if r[0] not in uncertain]
    merged = []
    for r in R:
        if merged and merged[-1][0] == r[0]:
            merged[-1][2] = r[2]
            merged[-1].append("bridged_uncertain_gap")
        else:
            merged.append(list(r))
    return merged
