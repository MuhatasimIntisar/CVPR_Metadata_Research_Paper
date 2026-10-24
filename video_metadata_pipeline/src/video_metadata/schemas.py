"""Record schemas and a small dependency-free validator.

A schema is {field: (types, required)} where types is a type, tuple of types, or a nested schema dict
(for objects) or [schema] (for a list of objects). validate() returns a list of human-readable errors;
an empty list means valid.
"""
from __future__ import annotations

NUM = (int, float)
OPT_STR = (str, type(None))
OPT_NUM = (int, float, type(None))

SOURCE_METADATA = {
    "video_id": (str, True), "ia_identifier": (str, True), "source_url": (str, True), "title": (OPT_STR, True),
    "description": (OPT_STR, True), "subjects": (list, True), "creator": (OPT_STR, True), "language": (OPT_STR, True),
    "language_code": (OPT_STR, True), "publication_date": (OPT_STR, True), "duration_s": (OPT_NUM, True),
    "formats": (list, True), "rights": (dict, True), "collections": (list, True), "original_platform": (dict, True),
    "raw_metadata": (dict, True), "role": (str, True),
}

TRANSCRIPT_SEGMENT = {"start": (NUM, True), "end": (NUM, True), "text": (str, True)}
TRANSCRIPT = {
    "video_id": (str, True), "primary": ((dict, type(None)), True), "segments": ([TRANSCRIPT_SEGMENT], True),
    "words": (list, True), "tracks": (list, True), "alternates": (dict, True), "policy": (str, True),
    "role": (str, True),
}

CANDIDATE_MOMENT = {
    "interval_id": (str, True), "video_id": (str, True), "start_s": (NUM, True), "end_s": (NUM, True),
    "status": (str, True), "activities": (list, True), "sub_events": (list, True), "environment": (list, True),
    "start_boundary": (dict, True), "end_boundary": (dict, True), "uncertainty": (dict, True),
    "representative_frames_s": (list, True),
}


def _type_ok(v, t):
    if t is float or t == NUM:
        return isinstance(v, (int, float)) and not isinstance(v, bool)
    return isinstance(v, t)


def validate(obj, schema: dict, path: str = "") -> list[str]:
    errs = []
    if not isinstance(obj, dict):
        return [f"{path or 'record'}: expected object, got {type(obj).__name__}"]
    for key, (t, required) in schema.items():
        p = f"{path}.{key}" if path else key
        if key not in obj:
            if required:
                errs.append(f"{p}: missing")
            continue
        v = obj[key]
        if isinstance(t, dict):
            errs += validate(v, t, p)
        elif isinstance(t, list):
            if not isinstance(v, list):
                errs.append(f"{p}: expected list")
            else:
                for i, x in enumerate(v):
                    errs += validate(x, t[0], f"{p}[{i}]")
        elif not _type_ok(v, t):
            errs.append(f"{p}: expected {getattr(t, '__name__', t)}, got {type(v).__name__}")
    return errs


def check_time_bounds(items: list[dict], duration: float | None = None, start="start_s", end="end_s",
                      allow_overlap=False) -> list[str]:
    """Each item has start < end, inside [0, duration]; sorted and (optionally) non-overlapping."""
    errs, prev_end = [], None
    for i, it in enumerate(items):
        a, b = it.get(start), it.get(end)
        if a is None or b is None or not a < b:
            errs.append(f"item {i}: invalid span {a}..{b}")
            continue
        if a < 0 or (duration is not None and b > duration + 1e-3):
            errs.append(f"item {i}: span {a}..{b} outside 0..{duration}")
        if prev_end is not None and not allow_overlap and a < prev_end - 1e-6:
            errs.append(f"item {i}: overlaps previous (starts {a} < {prev_end})")
        prev_end = b if prev_end is None else max(prev_end, b)
    return errs
