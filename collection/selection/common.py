"""Shared helpers for the metadata-only corpus selection (collection/selection/0*_*.py).

Standard library + numpy / scipy / scikit-learn only. Every stage reads saved outputs of the previous stage and
writes its own folder under the configured output root; the discovery CSVs are never modified.
"""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import math
import re
import subprocess
import sys
import unicodedata
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]                      # collection/selection -> repository root
DEFAULT_CONFIG = HERE / "selection_config.json"
CATEGORIES = ["meeting", "discussion", "interview", "presentation", "news",
              "performance", "demonstration", "guide", "documentary", "event"]
STAGE_DIRS = {1: "01_consolidation", 2: "02_eligibility", 3: "03_profile", 4: "04_selection", 5: "05_final"}
CHOSEN = {"x", "y", "yes", "1", "true"}       # same markers as collection/ia_download.py

csv.field_size_limit(min(sys.maxsize, 2 ** 31 - 1))


class StageError(SystemExit):
    """Raised with an informative message; exits with status 2."""

    def __init__(self, msg):
        super().__init__(f"ERROR: {msg}")


# ------------------------------------------------------------------ config / paths
def load_config(path=None) -> dict:
    p = Path(path) if path else DEFAULT_CONFIG
    if not p.exists():
        raise StageError(f"config not found: {p}")
    with open(p, encoding="utf-8") as f:
        cfg = json.load(f)
    cfg["_path"] = str(p)
    return cfg


def rpath(p) -> Path:
    """Paths in the config are relative to the repository root (so commands work from any folder)."""
    p = Path(p)
    return p if p.is_absolute() else REPO / p


def stage_dir(cfg, n, out_root=None) -> Path:
    d = rpath(out_root or cfg["paths"]["output_root"]) / STAGE_DIRS[n]
    d.mkdir(parents=True, exist_ok=True)
    return d


def require(path: Path, what: str):
    if not Path(path).exists():
        raise StageError(f"missing {what}: {path} (run the previous stage first)")
    return Path(path)


# ------------------------------------------------------------------ csv / json io
def read_csv(path) -> list[dict]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def csv_header(path) -> list[str]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        return next(csv.reader(f), [])


def write_csv(path, rows, fields=None):
    """utf-8-sig so Excel on Windows shows accents; fields default to the union of keys in first-seen order."""
    rows = list(rows)
    if fields is None:
        fields = list(dict.fromkeys(k for r in rows for k in r))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: fmt(r.get(k, "")) for k in fields})
    return Path(path)


def fmt(v):
    if v is None:
        return ""
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, float):
        return f"{v:.6g}"
    if isinstance(v, (list, tuple)):
        return ";".join(map(str, v))
    return v


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)
    return Path(path)


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def validate_columns(rows_or_header, required, where):
    cols = set(rows_or_header[0].keys()) if rows_or_header and isinstance(rows_or_header[0], dict) else set(rows_or_header)
    missing = [c for c in required if c not in cols]
    if missing:
        raise StageError(f"{where}: missing required columns {missing}")


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_json(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def code_version() -> dict:
    """git commit of the repository (if available) plus a hash of the selection scripts themselves."""
    h = hashlib.sha256()
    for p in sorted(HERE.glob("*.py")) + [DEFAULT_CONFIG]:
        if p.exists():
            h.update(p.name.encode())
            h.update(p.read_bytes())
    out = {"selection_code_sha256": h.hexdigest(), "python": sys.version.split()[0]}
    for mod in ("numpy", "scipy", "sklearn"):
        try:
            out[f"{mod}_version"] = __import__(mod).__version__
        except Exception:  # noqa: BLE001
            out[f"{mod}_version"] = None
    try:
        out["git_commit"] = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True,
                                           text=True, timeout=10).stdout.strip() or None
        dirty = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain", "--", "collection/selection"],
                               capture_output=True, text=True, timeout=10).stdout.strip()
        out["selection_code_uncommitted_changes"] = bool(dirty)
    except Exception:  # noqa: BLE001  (git not installed)
        out["git_commit"] = None
    return out


def now_iso():
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


# ------------------------------------------------------------------ parsing / normalisation
def to_float(v, default=None):
    try:
        x = float(str(v).strip())
        return x if math.isfinite(x) else default
    except (TypeError, ValueError):
        return default


def full_date(s):
    s = str(s or "").strip()[:10]
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
        try:
            return dt.date.fromisoformat(s)
        except ValueError:
            return None
    return None


def year_of(s):
    m = re.match(r"(\d{4})", str(s or ""))
    return int(m.group(1)) if m else None


def fold(s) -> str:
    """lower-case, accents removed, '&' -> 'and', every non-alphanumeric run -> one space."""
    s = unicodedata.normalize("NFKD", str(s or "")).encode("ascii", "ignore").decode().lower()
    s = s.replace("&", " and ")
    return " ".join(re.sub(r"[^a-z0-9]+", " ", s).split())


def channel_key(creator, station) -> str:
    """Station identity: the IA creator (the uploading station's name), else the identifier prefix."""
    c = fold(creator)
    return f"c:{c}" if c else f"p:{fold(station) or 'unknown'}"


def split_list(v, sep=";"):
    if isinstance(v, (list, tuple, set)):
        return [str(x).strip() for x in v if str(x).strip()]
    return [x.strip() for x in str(v or "").split(sep) if x.strip()]


def truthy(v) -> bool:
    return str(v or "").strip().lower() in CHOSEN


# ------------------------------------------------------------------ category re-matching
def clean_term(t: str) -> str:
    return fold(str(t).strip().strip('"'))


def term_regex(term: str):
    """Whole-word / whole-phrase match on folded text ('fair' does not match 'affair')."""
    return re.compile(r"(?<![a-z0-9])" + re.escape(term) + r"(?![a-z0-9])")


def compile_categories(cat_cfg: dict) -> dict:
    """categories.json -> {category: {"terms": [(term, regex)], "subject_tags": [...], "exclude": [...]}}."""
    glob_ex = [clean_term(t) for t in cat_cfg.get("exclude_title", [])]
    out = {}
    for name, c in cat_cfg["categories"].items():
        terms = [clean_term(t) for t in c.get("terms", [])]
        tags = [clean_term(t) for t in c.get("subject_tags", [])]
        ex = glob_ex + [clean_term(t) for t in c.get("exclude", [])]
        out[name] = {"terms": [(t, term_regex(t)) for t in dict.fromkeys(terms) if t],
                     "subject_tags": [(t, term_regex(t)) for t in dict.fromkeys(tags) if t],
                     "exclude": [term_regex(t) for t in ex if t]}
    return out


def match_categories(title: str, subjects: str, compiled: dict) -> list[dict]:
    """Every category whose configured terms occur in the title or the IA subject tags.
    Returns evidence rows {category, term, field}. A keyword match is evidence, not verified content."""
    t, s = fold(title), fold(" ; ".join(split_list(subjects)))
    ev = []
    for cat, c in compiled.items():
        if any(rx.search(t) for rx in c["exclude"]):
            continue
        for term, rx in c["terms"]:
            if rx.search(t):
                ev.append({"category": cat, "term": term, "field": "title"})
            elif rx.search(s):
                ev.append({"category": cat, "term": term, "field": "subject"})
        for term, rx in c["subject_tags"]:
            if rx.search(s):
                ev.append({"category": cat, "term": term, "field": "subject"})
    return ev


def evidence_str(ev: list[dict]) -> str:
    return " | ".join(f"{e['category']}:{e['field']}:{e['term']}" for e in ev)


def parse_evidence(s: str) -> list[dict]:
    out = []
    for part in str(s or "").split(" | "):
        bits = part.split(":", 2)
        if len(bits) == 3:
            out.append({"category": bits[0], "field": bits[1], "term": bits[2]})
    return out


# ------------------------------------------------------------------ programme titles
MONTHS = ("january february march april may june july august september october november december "
          "jan feb mar apr jun jul aug sep sept oct nov dec").split()
WEEKDAYS = "monday tuesday wednesday thursday friday saturday sunday mon tue tues wed thu thur thurs fri sat sun".split()
EPISODE_RX = re.compile(r"(?<![a-z0-9])(?:e|ep|episode|episodio|eps|show|no|part|pt|vol|season)\s*\d+[a-z]?(?![a-z])"
                        r"|#\s*\d+")
NUMBER_RX = re.compile(r"(?<![a-z0-9])\d+(?:st|nd|rd|th)?(?![a-z0-9])")
DATE_WORDS_RX = re.compile(r"(?<![a-z0-9])(?:" + "|".join(MONTHS + WEEKDAYS) + r")(?![a-z0-9])")
EDGE_WORDS = {"of", "for", "on", "the", "and", "at", "in", "a", "to", "from", "with", "de", "del", "la", "el"}
SEP_RX = re.compile(r"\s+[-–—|:]\s+|\s*\|\s*|:\s+")


def title_key(title: str) -> str:
    """Programme-level title: episode markers, dates, weekdays and numbers removed (original title untouched)."""
    t = EPISODE_RX.sub(" ", fold(title))
    t = DATE_WORDS_RX.sub(" ", t)
    t = NUMBER_RX.sub(" ", t).split()
    while t and t[-1] in EDGE_WORDS:
        t.pop()
    while t and t[0] in EDGE_WORDS:
        t.pop(0)
    return " ".join(t)


def title_prefix(title: str) -> tuple[str, bool]:
    """(series prefix, has_episode_marker): text before an explicit episode marker, else the first title segment."""
    f = fold(title)
    m = EPISODE_RX.search(f)
    if m and m.start() > 0:
        return title_key(f[:m.start()]), True
    raw = SEP_RX.split(unicodedata.normalize("NFKD", str(title or "")).encode("ascii", "ignore").decode())
    return title_key(raw[0] if raw else title), False


def episode_tokens(title: str) -> tuple:
    """Numbers / episode markers / date words in a title: differing tokens mean different episodes."""
    f = fold(title)
    return tuple(EPISODE_RX.findall(f)) + tuple(NUMBER_RX.findall(f)) + tuple(DATE_WORDS_RX.findall(f))


# ------------------------------------------------------------------ geography (source metadata only)
US_STATES = {
    "alabama": "AL", "alaska": "AK", "arizona": "AZ", "arkansas": "AR", "california": "CA", "colorado": "CO",
    "connecticut": "CT", "delaware": "DE", "florida": "FL", "georgia": "GA", "hawaii": "HI", "idaho": "ID",
    "illinois": "IL", "indiana": "IN", "iowa": "IA", "kansas": "KS", "kentucky": "KY", "louisiana": "LA",
    "maine": "ME", "maryland": "MD", "massachusetts": "MA", "michigan": "MI", "minnesota": "MN",
    "mississippi": "MS", "missouri": "MO", "montana": "MT", "nebraska": "NE", "nevada": "NV",
    "new hampshire": "NH", "new jersey": "NJ", "new mexico": "NM", "new york": "NY", "north carolina": "NC",
    "north dakota": "ND", "ohio": "OH", "oklahoma": "OK", "oregon": "OR", "pennsylvania": "PA",
    "rhode island": "RI", "south carolina": "SC", "south dakota": "SD", "tennessee": "TN", "texas": "TX",
    "utah": "UT", "vermont": "VT", "virginia": "VA", "washington": "WA", "west virginia": "WV",
    "wisconsin": "WI", "wyoming": "WY", "district of columbia": "DC", "washington dc": "DC"}
CA_PROVINCES = {
    "alberta": "AB", "british columbia": "BC", "manitoba": "MB", "new brunswick": "NB",
    "newfoundland and labrador": "NL", "nova scotia": "NS", "ontario": "ON", "prince edward island": "PE",
    "quebec": "QC", "saskatchewan": "SK", "yukon": "YT", "northwest territories": "NT", "nunavut": "NU"}
ABBREV = {v: ("US", v) for v in US_STATES.values()} | {v: ("CA", v) for v in CA_PROVINCES.values()}


def regions_from_subjects(subjects: str) -> list[tuple[str, str]]:
    """(country, region) from IA subject tags that are exactly a US state / Canadian province name.
    ('washington' is ambiguous with the city; it is accepted only as a whole tag.)"""
    out = []
    for tag in split_list(subjects):
        t = fold(tag)
        if t in US_STATES:
            out.append(("US", US_STATES[t]))
        elif t in CA_PROVINCES:
            out.append(("CA", CA_PROVINCES[t]))
    return list(dict.fromkeys(out))


def region_from_creator(creator: str):
    """Inferred: a trailing ', ND' / ', Ohio' style suffix in the station name (e.g. 'City of West Fargo, ND')."""
    raw = str(creator or "")
    m = re.search(r",\s*([A-Za-z .]+)\s*$", raw)
    if not m:
        return None
    tail = m.group(1).strip()
    if tail.upper() in ABBREV and len(tail) == 2:
        return ABBREV[tail.upper()]
    f = fold(tail)
    if f in US_STATES:
        return ("US", US_STATES[f])
    if f in CA_PROVINCES:
        return ("CA", CA_PROVINCES[f])
    return None


def duration_band(minutes, edges) -> str:
    if minutes is None:
        return "unknown"
    lo = 0
    for e in edges:
        if minutes < e:
            return f"{lo:g}-{e:g}"
        lo = e
    return f"{lo:g}+"


def quarter_of(date_str):
    d = full_date(date_str)
    if d:
        return f"{d.year}Q{(d.month - 1) // 3 + 1}"
    y = year_of(date_str)
    return f"{y}" if y else "unknown"


def gini(values) -> float:
    xs = sorted(v for v in values if v > 0)
    n = len(xs)
    if not n:
        return 0.0
    cum = sum((i + 1) * x for i, x in enumerate(xs))
    return round((2 * cum) / (n * sum(xs)) - (n + 1) / n, 4)


def concentration(counts: dict) -> dict:
    tot = sum(counts.values())
    if not tot:
        return {"n": 0}
    shares = sorted((v / tot for v in counts.values()), reverse=True)
    ent = -sum(p * math.log(p) for p in shares if p > 0)
    return {"n_unique": len(counts), "total": tot, "hhi": round(sum(p * p for p in shares), 5),
            "top1_share": round(shares[0], 4), "top10_share": round(sum(shares[:10]), 4),
            "normalised_entropy": round(ent / math.log(len(shares)), 4) if len(shares) > 1 else 0.0,
            "gini": gini(counts.values())}


def log(msg):
    print(msg, flush=True)
