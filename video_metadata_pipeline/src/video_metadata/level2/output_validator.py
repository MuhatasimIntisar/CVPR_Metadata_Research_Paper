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


def check_single(obj) -> list[str]:
    if not isinstance(obj, dict):
        return ["output is not an object"]
    errs = []
    if obj.get("decision") not in DECISIONS:
        errs.append(f"'decision' must be one of {sorted(DECISIONS)}")
    if not isinstance(obj.get("explanation"), str) or not obj["explanation"].strip():
        errs.append("'explanation' must be a non-empty string")
    return errs


META_FIELDS = {"description": str, "tags": list, "entities": list, "actions": list, "relationships": list,
               "context": dict, "attributes": dict}
ENTITY_TYPES = {"person", "person_group", "object", "organization", "place", "animal", "text"}
EVIDENCE = {"visual", "transcript", "both"}


def check_metadata(obj) -> list[str]:
    """Schema v2 structure. (Relationship rule violations are removed by normalise() and logged, not re-asked.)"""
    if not isinstance(obj, dict):
        return ["output is not an object"]
    errs = [f"'{k}' must be {t.__name__}" for k, t in META_FIELDS.items() if not isinstance(obj.get(k), t)]
    if errs:
        return errs
    if not obj["description"].strip():
        errs.append("description is empty")
    ids = []
    for e in obj["entities"]:
        if not isinstance(e, dict) or not e.get("id") or e.get("type") not in ENTITY_TYPES:
            errs.append(f"entity needs id and type in {sorted(ENTITY_TYPES)}: {str(e)[:80]}")
            continue
        ids.append(e["id"])
        if e.get("evidence") not in EVIDENCE:
            errs.append(f"entity {e['id']}: evidence must be one of {sorted(EVIDENCE)}")
    if len(ids) != len(set(ids)):
        errs.append("entity ids must be unique")
    for a in obj["actions"]:
        if not isinstance(a, dict) or not a.get("type"):
            errs.append("each action needs a type")
        elif any(p not in ids for p in a.get("participants") or []):
            errs.append(f"action participants must be entity ids: {a.get('participants')}")
    for r in obj["relationships"]:
        if not isinstance(r, dict) or not all(r.get(k) for k in ("subject", "predicate", "object")):
            errs.append("relationships need subject, predicate, object")
    return errs


def ask(vlm, content, max_new_tokens, checker, retries, tag):
    """Call, parse, validate; on failure append the errors and re-ask (bounded). Returns (obj or None, attempts)."""
    return ask_batch(vlm, [(content, max_new_tokens, tag)], checker, retries)[0]


def ask_batch(vlm, items, checker, retries):
    """Batched ask: round 0 sends every item together; each later round re-asks only the items still invalid, with
    their errors appended (same prompts and cache tags as one-by-one asking). Returns [(obj or None, attempts)]."""
    res = [None] * len(items)
    attempts = [[] for _ in items]
    cur = {i: list(c) for i, (c, _, _) in enumerate(items)}
    for r in range(retries + 1):
        todo = sorted(cur)
        if not todo:
            break
        outs = vlm.batch([(cur[i], items[i][1], f"{items[i][2]}_try{r}") for i in todo])
        for i, (out, cached) in zip(todo, outs):
            try:
                obj = extract_json(out)
                errs = checker(obj)
            except (ValueError, json.JSONDecodeError) as e:
                obj, errs = None, [f"invalid JSON: {e}"]
            attempts[i].append({"try": r, "cached": cached, "errors": errs, "raw_output": out[:4000]})
            if not errs:
                res[i] = obj
                del cur[i]
            else:
                cur[i] = list(items[i][0]) + [{"type": "text", "text": "Your previous answer was rejected: " + "; ".join(errs[:6]) +
                                               ". Answer again with ONLY the JSON object in the required format."}]
    return [(res[i], attempts[i]) for i in range(len(items))]
