#!/usr/bin/env python3
"""
TVM collection: find Internet Archive community-media candidates per category, let a human pick,
then download only the picked videos with full metadata and captions.

Categories, their search terms and the collections searched are set in collection/categories.json
(edit it to add categories or terms; the category name becomes the corpus/<category>/ folder).

Step 1  search (no downloads)
  python collection/ia_download.py search --seed 2
  python collection/ia_download.py search --seed 3 --candidates 80 --per-station 5 --pool 1000 --categories meeting events
  python collection/ia_download.py search --stations detvde --from-date 2026-03 --to-date 2026-04 --min-minutes 10 --max-minutes 30
  -> collection/rounds/candidates.csv  (--candidates per category, default 25)
  Videos already downloaded in earlier rounds (identifiers in collection/rounds/*manifest*.csv) are skipped.

Step 2  look, refine, pick
  python collection/ia_download.py report collection/rounds/candidates.csv
      length, channels, dates, same-channel series (videos on consecutive days), title words, which terms hit
  python collection/ia_download.py around collection/rounds/candidates.csv --days 2
      for the rows marked x: more videos from the same channel within +-2 days -> candidates_around.csv
  Open candidates.csv (Excel is fine), put an x in the 'select' column for the videos you want, save as CSV.
  Columns: title, duration_min, station (channel), creator, date, weekday, neighbours_1d (other listed videos
  from the same channel within a day), matched_terms (which search terms hit), subjects (the item's IA tags).

Step 3  download (only the rows marked x; asks for confirmation)
  python collection/ia_download.py download collection/rounds/candidates.csv

Output of step 3
  corpus/<category>/<short id>/   (corpus/ is git-ignored; short names avoid Windows' 260-char path limit)
      video.mp4            smallest suitable MP4 derivative under --max-mb
      metadata.json        full IA metadata API response (item fields + file list)
      captions*.srt/.vtt   captions if the item has any
      thumb.jpg            item thumbnail
      files.json           identifier + local name -> original archive file name
  collection/rounds/manifest.csv  the selected rows (next to the candidates csv; rename both to roundN_*.csv)

Candidate rules
  * the configured collections, movies only, youtube-* mirror IDs and excluded title words left out; any language
  * not access-restricted, has an MP4 under --max-mb, duration within [--min-min, --max-min]
  * dated within [--from-date, --to-date] (default 2024-2026); uses the item's date field,
    falling back to its upload date when no date is given (see the date_source column)
  * at most --per-station items per channel (the item's creator) per category, and at most --max-term-share of a
    category from one search term, for variety; --channels / --stations pick specific channels (then no channel
    cap); --captions-only keeps only captioned videos; length/date/channel/captions/term share can be set per category
  * an item is listed under the first category it matches (no duplicates across categories)
  * shuffled with --seed, so reruns with the same seed give the same list

Stdlib only (Python 3.8+).
"""
import argparse
import calendar
import csv
import datetime as dt
import hashlib
import json
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from types import SimpleNamespace
import urllib.error
import urllib.parse
import urllib.request

UA = "tvm-collection/0.2 (academic research dataset; metadata + selected downloads)"
SEARCH = "https://archive.org/advancedsearch.php"
META = "https://archive.org/metadata/"
DL = "https://archive.org/download/"
DELAY_S = 1.0

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG = os.path.join(HERE, "categories.json")
DEFAULT_ROUNDS = os.path.join(HERE, "rounds")


def as_date(s, end=False):
    """'2025' / '2025-03' / '2025-03-31' -> a date (first or last day of the period)."""
    m = re.fullmatch(r"(\d{4})(?:-(\d{2}))?(?:-(\d{2}))?", str(s).strip())
    if not m:
        raise SystemExit(f"bad date {s!r} (use YYYY, YYYY-MM or YYYY-MM-DD)")
    y, mo = int(m.group(1)), int(m.group(2) or (12 if end else 1))
    d = int(m.group(3)) if m.group(3) else (calendar.monthrange(y, mo)[1] if end else 1)
    return dt.date(y, mo, d)


def full_date(s):
    try:
        return dt.date.fromisoformat(str(s)[:10]) if re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(s)[:10]) else None
    except ValueError:
        return None


def rules_for(c, args):
    """Per-category settings from categories.json, falling back to the command-line values.
    --stations on the command line wins over a category's own list."""
    stations = [x.lower() for x in (getattr(args, "stations", None) or c.get("stations") or [])]
    channels = [" ".join(x.lower().split()) for x in (getattr(args, "channels", None) or c.get("channels") or [])]
    return SimpleNamespace(
        min_s=60 * float(c.get("min_minutes", args.min_minutes)), max_s=60 * float(c.get("max_minutes", args.max_minutes)),
        from_d=as_date(c.get("from_date", args.from_date)), to_d=as_date(c.get("to_date", args.to_date), end=True),
        stations=stations, channels=channels, max_mb=int(c.get("max_mb", args.max_mb)),
        captions_only=bool(getattr(args, "captions_only", False) or c.get("require_captions", False)))


def load_categories(path):
    with open(path, encoding="utf-8") as f:
        cfg = json.load(f)
    for name, c in cfg["categories"].items():
        if not c.get("terms") and not c.get("subject_tags"):
            raise SystemExit(f"{path}: category {name!r} needs 'terms' or 'subject_tags'")
    return cfg


MP4_PREF = ["h.264", "MPEG4", "h.264 IA", "512Kb MPEG4"]
CAPTION_EXT = (".srt", ".vtt")
CANDIDATE_FIELDS = ["select", "category", "title", "duration_min", "station", "creator", "date", "weekday",
                    "neighbours_1d", "date_source", "video_mb", "has_captions", "language", "matched_terms", "subjects",
                    "found_via", "description", "details_url", "identifier", "video_file", "video_md5", "captions"]


# ---------------------------------------------------------------- http
def get(url, params=None, as_json=True, retries=3):
    if params:
        url = f"{url}?{urllib.parse.urlencode(params, doseq=True)}"
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                body = r.read()
            time.sleep(DELAY_S)
            return json.loads(body) if as_json else body
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
            if attempt == retries - 1:
                print(f"  ! {e} ({url[:110]})", file=sys.stderr)
                return None
            time.sleep(2 * (attempt + 1))


def download(url, dest, expect_size=None, expect_md5=None, retries=4):
    if os.path.exists(dest) and expect_size and os.path.getsize(dest) == expect_size:
        return True
    tmp = dest + ".part"
    print("Downloading from:", url)
    for attempt in range(1, retries + 1):
        md5 = hashlib.md5()
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        try:
            with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
                if r.geturl() != url:
                    print("Redirected to:   ", r.geturl())
                total, done = int(r.headers.get("Content-Length") or expect_size or 0), 0
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
                    md5.update(chunk)
                    done += len(chunk)
                    if total:
                        print(f"\r    {os.path.basename(dest)[:50]:50} {done / total:6.1%}", end="", flush=True)
            print()
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
            print(f"\n  ! attempt {attempt}/{retries} failed: {e}", file=sys.stderr)
            if attempt < retries:
                time.sleep(5 * attempt)
                continue
            return False
        if expect_md5 and md5.hexdigest() != expect_md5:
            print(f"  ! md5 mismatch for {dest} (attempt {attempt}/{retries})", file=sys.stderr)
            if attempt < retries:
                continue
            return False
        os.replace(tmp, dest)
        return True
    return False


# ---------------------------------------------------------------- helpers
def to_seconds(v):
    """Parse IA durations: '158.02', '00:02:38', '2:38', '158 seconds', '26 min'."""
    if v is None:
        return None
    s = str(v[0] if isinstance(v, list) else v).strip().lower()
    if re.fullmatch(r"\d+(\.\d+)?", s):
        return float(s)
    if re.fullmatch(r"\d+(:\d{1,2}){1,2}(\.\d+)?", s):
        sec = 0.0
        for part in s.split(":"):
            sec = sec * 60 + float(part)
        return sec
    m = re.match(r"(\d+(?:\.\d+)?)\s*(s|sec|second|m|min|minute|h|hr|hour)", s)
    if m:
        mult = {"s": 1, "m": 60, "h": 3600}[m.group(2)[0]]
        return float(m.group(1)) * mult
    return None


def channel_of(row):
    """Channel = the item's creator (the uploading station's name); identifier prefix only when it has none.
    (Prefixes are unreliable: one station can upload under many prefixes, several stations can share one.)"""
    return " ".join(str(row.get("creator") or "").lower().split()) or (row.get("station") or station_of(row["identifier"]))


def station_of(identifier):
    m = re.match(r"^([A-Za-z0-9]+)[-_]", identifier)
    return m.group(1).lower() if m else identifier[:8].lower()


def first(v):
    return v[0] if isinstance(v, list) and v else (v or "")


def pick_video(files, max_mb):
    mp4s = [f for f in files if f.get("name", "").lower().endswith(".mp4") and f.get("size")]
    def rank(f):
        fmt = f.get("format", "")
        return (MP4_PREF.index(fmt) if fmt in MP4_PREF else len(MP4_PREF), int(f["size"]))
    for f in sorted(mp4s, key=rank):
        if int(f["size"]) <= max_mb * 1024 * 1024:
            return f
    return None


def duration_of(meta, files, video):
    for v in (video.get("length") if video else None, meta.get("runtime"),
              *(f.get("length") for f in files if f.get("source") == "original")):
        sec = to_seconds(v)
        if sec:
            return sec
    return None


# ---------------------------------------------------------------- selection
def base_query(cfg, rules):
    q = f"collection:({' OR '.join(cfg['collections'])}) AND mediatype:movies AND NOT identifier:youtube-*"
    if rules.stations:
        q += f" AND identifier:({' OR '.join(st + '*' for st in rules.stations)})"
    if getattr(rules, "channels", None):
        q += " AND creator:(" + " OR ".join(f'"{ch}"' for ch in rules.channels) + ")"
    span = f"[{rules.from_d.isoformat()} TO {rules.to_d.isoformat()}]"
    # date = content/air date; publicdate = upload date, used when an item has no date
    return f"({q}) AND (date:{span} OR publicdate:{span})"


def candidates(cfg, cat, rules, pool):
    c = cfg["categories"][cat]
    terms, tags = c.get("terms", []), c.get("subject_tags", [])
    parts = []
    if terms:
        parts.append(f"title:({' OR '.join(terms)})")
    if terms or tags:
        parts.append(f"subject:({' OR '.join(terms + tags)})")
    q = f"{base_query(cfg, rules)} AND ({' OR '.join(parts)})"
    excl = cfg.get("exclude_title", []) + c.get("exclude", [])
    if excl:
        q += f" AND NOT title:({' OR '.join(excl)})"
    return run_query(q, pool)


def run_query(q, pool):
    data = get(SEARCH, {"q": q, "fl[]": ["identifier"], "rows": pool, "sort[]": "downloads desc", "output": "json"})
    docs = (data or {}).get("response", {}).get("docs", [])
    return [d["identifier"] for d in docs], (data or {}).get("response", {}).get("numFound", 0)


def matched(meta, c):
    """Which configured terms/tags actually occur in the title or the IA subject tags (shown to the picker)."""
    title = str(first(meta.get("title"))).lower()
    subj = meta.get("subject", [])
    subj = " | ".join(map(str, subj if isinstance(subj, list) else [subj])).lower()
    hits = [t.strip('"') for t in c.get("terms", []) if t.strip('"').lower() in title or t.strip('"').lower() in subj]
    hits += [t.strip('"') for t in c.get("subject_tags", []) if t.strip('"').lower() in subj]
    return "; ".join(dict.fromkeys(hits))


def previous_ids(rounds_dir):
    ids = set()
    if os.path.isdir(rounds_dir):
        for n in os.listdir(rounds_dir):
            if "manifest" in n and n.endswith(".csv"):
                with open(os.path.join(rounds_dir, n), encoding="utf-8-sig") as f:
                    ids |= {r["identifier"] for r in csv.DictReader(f) if r.get("identifier")}
    return ids


def evaluate(identifier, rules):
    if rules.stations and station_of(identifier) not in rules.stations:
        return None, "other station"
    item = get(META + urllib.parse.quote(identifier))
    if not item or "metadata" not in item:
        return None, "no metadata"
    meta, files = item["metadata"], item.get("files", [])
    if str(meta.get("access-restricted-item", "")).lower() == "true" or item.get("is_dark"):
        return None, "restricted"
    if getattr(rules, "channels", None) and " ".join(str(first(meta.get("creator"))).lower().split()) not in rules.channels:
        return None, "other channel"
    if getattr(rules, "captions_only", False) and not any(f.get("name", "").lower().endswith(CAPTION_EXT) for f in files):
        return None, "no captions"
    date, date_src = item_date(meta)
    fd = full_date(date)
    if not date or (fd and not rules.from_d <= fd <= rules.to_d) or \
            (not fd and not rules.from_d.year <= int(date[:4]) <= rules.to_d.year):
        return None, f"date {date or 'unknown'}"
    video = pick_video(files, rules.max_mb)
    if not video:
        return None, f"no mp4 under {rules.max_mb} MB"
    sec = duration_of(meta, files, video)
    if sec is None:
        return None, "unknown duration"
    if not (rules.min_s <= sec <= rules.max_s):
        return None, f"duration {sec / 60:.1f} min"
    return {"item": item, "meta": meta, "files": files, "video": video, "seconds": sec,
            "date": date, "date_src": date_src}, "ok"


def make_row(ident, cat, c, res, found_via=""):
    v, meta = res["video"], res["meta"]
    caps = [f["name"] for f in res["files"] if f.get("name", "").lower().endswith(CAPTION_EXT)]
    subj = meta.get("subject", [])
    subj = subj if isinstance(subj, list) else [subj]
    lang = meta.get("language", [])
    fd = full_date(res["date"])
    return {
        "select": "", "category": cat,
        "title": str(first(meta.get("title")))[:150],
        "duration_min": f"{res['seconds'] / 60:.1f}",
        "station": station_of(ident), "creator": str(first(meta.get("creator")))[:60],
        "date": res["date"], "weekday": fd.strftime("%a") if fd else "", "neighbours_1d": "",
        "date_source": res["date_src"],
        "video_mb": f"{int(v['size']) / 1048576:.1f}",
        "has_captions": "yes" if caps else "",
        "language": "; ".join(map(str, lang if isinstance(lang, list) else [lang])),
        "matched_terms": matched(meta, c) if c else "",
        "subjects": "; ".join(map(str, subj[:10]))[:200],
        "found_via": found_via,
        "description": clean_text(meta.get("description"), 300),
        "details_url": f"https://archive.org/details/{ident}",
        "identifier": ident, "video_file": v["name"], "video_md5": v.get("md5", ""),
        "captions": "|".join(caps),
    }


def mark_neighbours(rows, days=1):
    """neighbours_1d = other listed videos from the same channel dated within +-days (series / consecutive days)."""
    by = defaultdict(list)
    for r in rows:
        fd = full_date(r.get("date"))
        if fd:
            by[channel_of(r)].append((fd, r))
    for lst in by.values():
        for fd, r in lst:
            r["neighbours_1d"] = str(sum(1 for g, x in lst if x is not r and abs((g - fd).days) <= days))
    return rows


def item_date(meta):
    """Content date if present, otherwise upload date. Returns (YYYY-MM-DD or YYYY, source)."""
    for key in ("date", "publicdate", "addeddate"):
        m = re.match(r"(\d{4})(-\d{2}-\d{2})?", str(first(meta.get(key))).strip())
        if m:
            return m.group(0), ("aired" if key == "date" else "uploaded")
    return "", ""


def search(args):
    """Collect a varied candidate pool per category (no downloads)."""
    cfg = load_categories(args.config)
    cats = args.categories or list(cfg["categories"])
    unknown = [c for c in cats if c not in cfg["categories"]]
    if unknown:
        raise SystemExit(f"unknown categories {unknown}; defined in {args.config}: {list(cfg['categories'])}")
    rng = random.Random(args.seed)
    used_ids = set() if args.allow_repeats else previous_ids(args.out)
    if used_ids:
        print(f"skipping {len(used_ids)} videos already downloaded in earlier rounds")
    rows = []
    for cat in cats:
        c = cfg["categories"][cat]
        rules = rules_for(c, args)
        need = int(c.get("candidates", args.candidates))
        cap = need if (rules.stations or rules.channels) else args.per_station  # chosen channels are not capped
        term_cap = max(1, int(round(float(c.get("max_term_share", args.max_term_share)) * need)))
        per_term = Counter()
        print(f"\n=== {cat}  ({rules.min_s / 60:g}-{rules.max_s / 60:g} min, {rules.from_d} to {rules.to_d}"
              + (f", channels {', '.join(rules.stations)}" if rules.stations else "") + ")", flush=True)
        per_station = {}
        ids, total = candidates(cfg, cat, rules, args.pool)
        rng.shuffle(ids)
        print(f"  {total:,} matches, checking up to {len(ids)} for {need} candidates", flush=True)
        got = 0
        for ident in ids:
            if got >= need:
                break
            if ident in used_ids:
                continue
            res, why = evaluate(ident, rules)
            if not res:
                if args.verbose:
                    print(f"     skip {ident[:70]}: {why}")
                continue
            row = make_row(ident, cat, c, res)
            st, term = channel_of(row), (row["matched_terms"].split(";")[0].strip() or "-")
            if per_station.get(st, 0) >= cap or per_term[term] >= term_cap:
                if args.verbose:
                    print(f"     skip {ident[:70]}: {'channel' if per_station.get(st, 0) >= cap else 'term'} quota full")
                continue
            used_ids.add(ident)
            per_station[st] = per_station.get(st, 0) + 1
            per_term[term] += 1
            got += 1
            rows.append(row)
            print(f"     {got:>3}. {rows[-1]['title'][:62]}  ({rows[-1]['duration_min']} min, {st}, {rows[-1]['date']})", flush=True)
        if got < need:
            print(f"  ! only {got}/{need} candidates for {cat}; raise --pool, --per-station or --max-term-share, widen "
                  f"the length or date range, or add terms in {os.path.basename(args.config)}")
    return mark_neighbours(rows)


def around(args):
    """For every picked row: more videos from the same channel dated within +-days (series, consecutive days)."""
    with open(args.csv, encoding="utf-8-sig") as f:
        listed = list(csv.DictReader(f))
    seeds = [r for r in listed if (r.get("select") or "").strip().lower() in CHOSEN]
    if not seeds:
        raise SystemExit("No rows selected. Put an x in the 'select' column of the videos to expand from.")
    cfg = load_categories(args.config)
    used = {r["identifier"] for r in listed} | (set() if args.allow_repeats else previous_ids(os.path.dirname(os.path.abspath(args.csv))))
    rows = []
    for sd in seeds:
        d = full_date(sd.get("date"))
        st = sd.get("station") or station_of(sd["identifier"])
        creator = " ".join(str(sd.get("creator") or "").lower().split())
        if not d:
            print(f"  - {sd['identifier'][:60]}: no exact date, skipped")
            continue
        rules = SimpleNamespace(min_s=60 * args.min_minutes, max_s=60 * args.max_minutes, max_mb=args.max_mb,
                                captions_only=args.captions_only,
                                from_d=d - dt.timedelta(days=args.days), to_d=d + dt.timedelta(days=args.days),
                                stations=[] if creator else [st], channels=[creator] if creator else [])
        ids, total = run_query(base_query(cfg, rules), args.pool)
        print(f"\n=== {creator or st} around {d} (+-{args.days} d), from {sd['title'][:50]}: {total} items", flush=True)
        got = 0
        for ident in sorted(ids):
            if got >= args.per_item:
                break
            if ident in used:
                continue
            res, why = evaluate(ident, rules)
            if not res:
                if args.verbose:
                    print(f"     skip {ident[:70]}: {why}")
                continue
            used.add(ident)
            got += 1
            rows.append(make_row(ident, sd.get("category", ""), cfg["categories"].get(sd.get("category", "")), res,
                                 found_via=sd["identifier"]))
            print(f"     {got:>3}. {rows[-1]['date']} {rows[-1]['title'][:60]}  ({rows[-1]['duration_min']} min)", flush=True)
    return mark_neighbours(rows)


STOP = set("the a an and or of in on at to for with from by is are be this that it its as - & | live new part ep episode "
           "video show 2023 2024 2025 2026 de la el y en los las del".split())


def report(args):
    """Summary of candidate/manifest CSVs to steer the next search: length, channels, dates, series, title words."""
    rows = []
    for p in args.csv:
        with open(p, encoding="utf-8-sig") as f:
            rows += list(csv.DictReader(f))
    if args.selected:
        rows = [r for r in rows if (r.get("select") or "").strip().lower() in CHOSEN]
    if not rows:
        raise SystemExit("no rows")
    cfg = load_categories(args.config) if os.path.exists(args.config) else {"categories": {}}
    groups = [("ALL", rows)] + [(c, [r for r in rows if r.get("category") == c]) for c in dict.fromkeys(r.get("category") for r in rows)]
    for name, rs in groups:
        mins = sorted(float(r["duration_min"]) for r in rs if r.get("duration_min"))
        print(f"\n=== {name}: {len(rs)} videos")
        if mins:
            b = Counter("<10" if m < 10 else "10-20" if m < 20 else "20-30" if m < 30 else "30-60" if m < 60 else "60+" for m in mins)
            print(f"  length   min {mins[0]:.0f} / median {mins[len(mins) // 2]:.0f} / max {mins[-1]:.0f} min;  "
                  + ", ".join(f"{k} min: {b[k]}" for k in ("<10", "10-20", "20-30", "30-60", "60+") if b[k]))
        ch = Counter(channel_of(r) for r in rs)
        name_of = {channel_of(r): (r.get("creator") or channel_of(r)) for r in rs}
        print(f"  channels {len(ch)} distinct; " + ", ".join(f"{name_of[k][:30]}: {n}" for k, n in ch.most_common(10)))
        if rs and ch.most_common(1)[0][1] / len(rs) > 0.2:
            print(f"  ! one channel is {ch.most_common(1)[0][1] / len(rs):.0%} of this group")
        caps = sum((r.get("has_captions") or "") == "yes" for r in rs)
        print(f"  captions {caps}/{len(rs)} have captions (the pipeline's transcript comes from them)")
        ds = sorted(d for d in (full_date(r.get("date")) for r in rs) if d)
        if ds:
            ym = Counter(d.strftime("%Y-%m") for d in ds)
            print(f"  dates    {ds[0]} to {ds[-1]};  busiest months " + ", ".join(f"{k}: {n}" for k, n in ym.most_common(6)))
        if name == "ALL":
            by = defaultdict(list)
            for r in rs:
                d = full_date(r.get("date"))
                if d:
                    by[channel_of(r)].append(d)
            runs = []
            for st, lst in by.items():
                lst = sorted(set(lst))
                cur = [lst[0]]
                for d in lst[1:]:
                    if (d - cur[-1]).days <= args.days:
                        cur.append(d)
                    else:
                        runs.append((st, cur)) if len(cur) > 1 else None
                        cur = [d]
                runs.append((st, cur)) if len(cur) > 1 else None
            runs.sort(key=lambda x: -len(x[1]))
            print(f"  series   same channel, dates <= {args.days} day(s) apart: "
                  + ("; ".join(f"{st} {r[0]}..{r[-1]} ({len(r)})" for st, r in runs[:10]) or "none"))
        words = Counter(w for r in rs for w in set(re.findall(r"[a-záéíóúñ]+", r.get("title", "").lower()))
                        if w not in STOP and len(w) > 2)
        print("  title    " + ", ".join(f"{w} {n}" for w, n in words.most_common(15)))
        mt = Counter(t.strip() for r in rs for t in (r.get("matched_terms") or "").split(";") if t.strip())
        if mt:
            print("  matched  " + ", ".join(f"{t} {n}" for t, n in mt.most_common(12)))
            if rs and mt.most_common(1)[0][1] / len(rs) > 0.35:
                print(f"  ! '{mt.most_common(1)[0][0]}' matches {mt.most_common(1)[0][1] / len(rs):.0%} of this group")
        c = cfg["categories"].get(name)
        if c and mt:  # only csvs written by this version record matched terms
            unused = [t.strip('"') for t in c.get("terms", []) if t.strip('"') not in mt]
            if unused:
                print(f"  unused terms (matched nothing here): {', '.join(unused)}")
    top = [r.get("creator") for r in rows if r.get("creator")]
    top = [k for k, _ in Counter(top).most_common(3)]
    print("\nNext-search ideas:")
    if top:
        print("  more from the top channels:   python collection/ia_download.py search --channels "
              + " ".join(f'"{k}"' for k in top) + " --seed <n>")
    print(f"  series around picked videos:  python collection/ia_download.py around <csv with x marks> --days 2")
    print(f"  other lengths / dates:        --min-minutes 10 --max-minutes 30 --from-date 2025-01 --to-date 2025-06")


def clean_text(v, limit):
    s = " ".join(map(str, v)) if isinstance(v, list) else str(v or "")
    s = re.sub(r"<[^>]+>", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s[:limit] + ("..." if len(s) > limit else "")


# ---------------------------------------------------------------- download
def short_dir(ident, keep=40):
    """Short, unique folder name (Windows paths are limited to 260 characters)."""
    base = re.sub(r"[^A-Za-z0-9_-]", "_", ident[:keep]).rstrip("_-")
    return f"{base}_{hashlib.md5(ident.encode()).hexdigest()[:6]}"


def local_name(name, video_file):
    """video.mp4 / captions.en.asr.srt / thumb.jpg instead of the long archive file names."""
    if name == video_file:
        return "video" + os.path.splitext(name)[1].lower()
    if name == "__ia_thumb.jpg":
        return "thumb.jpg"
    stem = os.path.splitext(video_file)[0]
    if name.startswith(stem):
        return "captions" + name[len(stem):]
    return "captions_" + hashlib.md5(name.encode()).hexdigest()[:6] + os.path.splitext(name)[1]


def fetch_item(row, item, out):
    ident = row["identifier"]
    d = os.path.join(out, row["category"], short_dir(ident))
    os.makedirs(d, exist_ok=True)
    if item is None:
        item = get(META + urllib.parse.quote(ident)) or {}
    with open(os.path.join(d, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(item, f, ensure_ascii=False, indent=2)
    by_name = {f["name"]: f for f in item.get("files", [])}
    wanted = [row["video_file"]] + [c for c in row["captions"].split("|") if c]
    if "__ia_thumb.jpg" in by_name:
        wanted.append("__ia_thumb.jpg")
    ok, mapping = True, {}
    for name in wanted:
        meta = by_name.get(name, {})
        url = DL + urllib.parse.quote(ident) + "/" + urllib.parse.quote(name)
        local = local_name(name, row["video_file"])
        mapping[local] = name
        size = int(meta["size"]) if meta.get("size") else None
        ok &= download(url, os.path.join(d, local), size, meta.get("md5"))
    with open(os.path.join(d, "files.json"), "w", encoding="utf-8") as f:
        json.dump({"identifier": ident, "details_url": f"https://archive.org/details/{ident}",
                   "files": mapping}, f, ensure_ascii=False, indent=2)
    print(f"    -> {d}")
    return ok


CHOSEN = {"x", "y", "yes", "1", "true"}


def write_candidates(rows, out):
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, "candidates.csv")
    # utf-8-sig so Excel on Windows shows Spanish accents correctly
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=CANDIDATE_FIELDS)
        w.writeheader()
        w.writerows(rows)
    return path


def cmd_search(args):
    rows = search(args)
    path = write_candidates(rows, args.out)
    print("\nCandidates per category:")
    for cat in dict.fromkeys(r["category"] for r in rows):
        print(f"  {cat:12} {sum(1 for r in rows if r['category'] == cat):>4}")
    print(f"\nWrote {len(rows)} candidates to {path}")
    print("Next: open it, put an x in the 'select' column for the videos you want, save, then run:")
    print(f"  python {os.path.basename(sys.argv[0])} download {path}")


def cmd_around(args):
    rows = around(args)
    out = os.path.join(os.path.dirname(os.path.abspath(args.csv)), "candidates_around.csv")
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=CANDIDATE_FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {len(rows)} candidates to {out} (found_via = the picked video they came from)")


def cmd_download(args):
    with open(args.csv, encoding="utf-8-sig") as f:
        rows = [r for r in csv.DictReader(f) if (r.get("select") or "").strip().lower() in CHOSEN]
    if not rows:
        print("No rows selected. Put an x in the 'select' column first.")
        return
    print("Selected per category:")
    for cat in dict.fromkeys(r["category"] for r in rows):
        k = sum(1 for r in rows if r["category"] == cat)
        print(f"  {cat:10} {k}{'' if k == args.expect else f'   (expected {args.expect})'}")
    total_mb = sum(float(r.get("video_mb") or 0) for r in rows)
    out = args.out
    if not args.yes:
        ans = input(f"\nDownload {len(rows)} videos (~{total_mb:.0f} MB) into {out}? [y/N] ").strip().lower()
        if ans not in ("y", "yes"):
            print("Cancelled.")
            return
    failed = []
    for row in rows:
        print(f"  {row['category']:9} {row['identifier'][:70]}")
        if not fetch_item(row, None, out):
            failed.append(row["identifier"])
    man = os.path.join(os.path.dirname(os.path.abspath(args.csv)), "manifest.csv")
    with open(man, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=CANDIDATE_FIELDS)
        w.writeheader()
        w.writerows({k: r.get(k, "") for k in CANDIDATE_FIELDS} for r in rows)
    print(f"\nDone. {len(rows) - len(failed)}/{len(rows)} complete. Manifest: {man}")
    if failed:
        print("Failed (rerun the same command to resume):", *failed, sep="\n  ")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def rules_args(p, per_category=True):
        note = " (a category in the config can override it)" if per_category else ""
        p.add_argument("--min-minutes", "--min-min", type=float, default=1.0, help="shortest video, minutes" + note)
        p.add_argument("--max-minutes", "--max-min", type=float, default=30.0, help="longest video, minutes" + note)
        p.add_argument("--max-mb", type=int, default=300, help="largest video file, MB" + note)
        p.add_argument("--pool", type=int, default=200, help="search hits to check per category / per picked video")
        p.add_argument("--config", default=DEFAULT_CONFIG, help="categories, search terms and collections (JSON)")
        p.add_argument("--allow-repeats", action="store_true", help="do not skip videos downloaded in earlier rounds")
        p.add_argument("-v", "--verbose", action="store_true", help="show why candidates were skipped")

    s = sub.add_parser("search", help="find candidates and write candidates.csv (no downloads)")
    rules_args(s)
    s.add_argument("-o", "--out", default=DEFAULT_ROUNDS, help="where candidates.csv is written")
    s.add_argument("--candidates", type=int, default=25, help="candidates per category (a category can override it)")
    s.add_argument("--per-station", type=int, default=3, help="max candidates per channel per category (not applied with --stations/--channels)")
    s.add_argument("--max-term-share", type=float, default=1.0,
                   help="max share of a category from one search term, e.g. 0.3 (a category can set max_term_share); 1 = no limit")
    s.add_argument("--channels", nargs="+", help='only these channels, by name as in the creator column, e.g. "DETV" "Nutmeg TV"')
    s.add_argument("--captions-only", action="store_true", help="only videos with captions (a category can set require_captions)")
    s.add_argument("--categories", nargs="+", help="only search these categories (default: all in the config)")
    s.add_argument("--stations", nargs="+", help="only these identifier prefixes (e.g. detvde cctv17vt); --channels is usually better")
    s.add_argument("--from-date", default="2024", help="earliest date: YYYY, YYYY-MM or YYYY-MM-DD")
    s.add_argument("--to-date", default="2026", help="latest date: YYYY, YYYY-MM or YYYY-MM-DD")
    s.add_argument("--seed", type=int, default=42)

    a = sub.add_parser("around", help="more videos from the same channel within +-N days of each picked row")
    a.add_argument("csv", help="a candidates/manifest csv with x marks on the videos to expand from")
    rules_args(a, per_category=False)
    a.add_argument("--days", type=int, default=2, help="+- days around each picked video's date")
    a.add_argument("--per-item", type=int, default=10, help="max new candidates per picked video")
    a.add_argument("--captions-only", action="store_true", help="only videos with captions")

    r = sub.add_parser("report", help="summarise candidate/manifest csvs (length, channel, date, series, title words)")
    r.add_argument("csv", nargs="+")
    r.add_argument("--selected", action="store_true", help="only rows marked x")
    r.add_argument("--days", type=int, default=1, help="max gap for a 'series' of same-channel videos")
    r.add_argument("--config", default=DEFAULT_CONFIG)

    d = sub.add_parser("download", help="download rows marked in the 'select' column")
    d.add_argument("csv", help="candidates.csv with your selections")
    d.add_argument("-o", "--out", default=os.path.join(os.path.dirname(HERE), "corpus"),
                   help="where the videos go (default: corpus/ at the repository root)")
    d.add_argument("--expect", type=int, default=4, help="expected picks per category (warning only)")
    d.add_argument("-y", "--yes", action="store_true", help="skip the confirmation prompt")

    args = ap.parse_args()
    {"search": cmd_search, "around": cmd_around, "report": report, "download": cmd_download}[args.cmd](args)


if __name__ == "__main__":
    main()
