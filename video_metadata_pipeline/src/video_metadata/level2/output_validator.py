"""Parse and validate VLM JSON. Invalid output is retried (bounded) and otherwise reported, never filled in."""
from __future__ import annotations

import json
import re

FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def extract_json(text: str):
    """Deterministic repair: strip code fences / prose, take the outermost {...}, drop trailing commas."""
    m = FENCE.search(text)
    if m:
        text = m.group(1)
    a, b = text.find("{"), text.rfind("}")
    if a < 0 or b <= a:
        raise ValueError("no JSON object in output")
    s = text[a:b + 1]
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return json.loads(re.sub(r",\s*([}\]])", r"\1", s))


DECISIONS = {"merge", "keep", "uncertain"}


def check_merge(obj, boundary_ids) -> list[str]:
    errs = []
    ds = obj.get("decisions") if isinstance(obj, dict) else None
    if not isinstance(ds, list):
        return ["missing 'decisions' list"]
    seen = set()
    for d in ds:
        if not isinstance(d, dict):
            errs.append("decision is not an object")
            continue
        if d.get("boundary") not in boundary_ids:
            errs.append(f"unknown boundary {d.get('boundary')!r}")
        if d.get("decision") not in DECISIONS:
            errs.append(f"decision must be one of {sorted(DECISIONS)}")
        if not isinstance(d.get("explanation"), str) or not d.get("explanation", "").strip():
            errs.append("each decision needs an explanation")
        seen.add(d.get("boundary"))
    missing = set(boundary_ids) - seen
    if missing:
        errs.append(f"missing decisions for {sorted(missing)}")
    return errs


META_FIELDS = {"description": str, "tags": list, "entities": list, "events": list, "relationships": list, "context": dict}


def check_metadata(obj) -> list[str]:
    if not isinstance(obj, dict):
        return ["output is not an object"]
    errs = [f"'{k}' must be {t.__name__}" for k, t in META_FIELDS.items() if not isinstance(obj.get(k), t)]
    ids = {e.get("id") for e in obj.get("entities", []) if isinstance(e, dict)}
    for e in obj.get("entities", []) if isinstance(obj.get("entities"), list) else []:
        if not isinstance(e, dict) or not e.get("id") or not e.get("type"):
            errs.append("entities need 'id' and 'type'")
    for ev in obj.get("events", []) if isinstance(obj.get("events"), list) else []:
        if not isinstance(ev, dict) or not ev.get("type"):
            errs.append("events need 'type'")
        elif any(p not in ids for p in ev.get("participants", []) or []):
            errs.append(f"event participants must be entity ids: {ev.get('participants')}")
    for rel in obj.get("relationships", []) if isinstance(obj.get("relationships"), list) else []:
        if not isinstance(rel, dict) or not all(rel.get(k) for k in ("subject", "predicate", "object")):
            errs.append("relationships need subject, predicate, object")
    return errs


def ask(vlm, content, max_new_tokens, checker, retries, tag):
    """Call, parse, validate; on failure append the errors and re-ask (bounded). Returns (obj or None, attempts)."""
    attempts = []
    cur = list(content)
    for i in range(retries + 1):
        out, cached = vlm(cur, max_new_tokens, tag=f"{tag}_try{i}")
        try:
            obj = extract_json(out)
            errs = checker(obj)
        except (ValueError, json.JSONDecodeError) as e:
            obj, errs = None, [f"invalid JSON: {e}"]
        attempts.append({"try": i, "cached": cached, "errors": errs, "raw_output": out[:4000]})
        if not errs:
            return obj, attempts
        cur = list(content) + [{"type": "text", "text": "Your previous answer was rejected: " + "; ".join(errs[:6]) +
                                ". Answer again with ONLY the JSON object in the required format."}]
    return None, attempts
