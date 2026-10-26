#!/usr/bin/env python3
"""
TVM pilot: find Internet Archive community-media candidates per category, let a human pick,
then download only the picked videos with full metadata and captions.

Categories: interview, meeting, event, guide, museum

Step 1  search (no downloads)
  python collection/ia_download.py search
  python collection/ia_download.py search --candidates 40 --spanish-candidates 15 --max-min 20
  -> collection/rounds/candidates.csv  (25 English + 10 Spanish per category by default)

Step 2  pick
  Open candidates.csv (Excel is fine), put an x in the 'select' column for the videos you want,
  e.g. 4 per category, and save as CSV.

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
  * collection:community_media, movies only, youtube-* mirror IDs excluded
  * not access-restricted, has an MP4 under --max-mb, duration within [--min-min, --max-min]
  * dated within [--from-year, --to-year] (default 2024-2026); uses the item's date field,
    falling back to its upload date when no date is given (see the date_source column)
  * at most --per-station items per station prefix (e.g. "cctv17vt-") per category, for variety
  * shuffled with --seed, so reruns with the same seed give the same list

Stdlib only (Python 3.8+).
"""
import argparse
import csv
import hashlib
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

UA = "tvm-pilot/0.1 (academic research; small pilot download)"
SEARCH = "https://archive.org/advancedsearch.php"
META = "https://archive.org/metadata/"
DL = "https://archive.org/download/"
DELAY_S = 0.5

BASE_Q = "collection:community_media AND mediatype:movies AND NOT identifier:youtube-*"
EXCLUDE = 'NOT title:(trailer OR promo OR teaser OR "full schedule" OR "bulletin board")'
# The IA language field is sparse here, so Spanish is detected from the language field, Spanish
# title words, or a "Spanish"/"en Español" marker in the title. Phrases where "Spanish" is the topic
# rather than the language are excluded.
SPANISH_MARK = '(spanish OR "español" OR espanol OR "en español" OR "en espanol")'
SPANISH_TOPIC = ('("spanish flu" OR "spanish immersion" OR "spanish american" OR "spanish class" '
                 'OR "learn spanish" OR "spanish speaking" OR "spanish club" OR "spanish 1" OR "spanish 2" '
                 'OR "spanish fork" OR "spanish springs" OR "spanish trail")')
SPANISH_HINTS = ("spanish", "español", "espanol")

# en: English title/subject terms. es: Spanish title terms (also count as a Spanish signal).
CATEGORIES = {
    "interview": {
        "en": ['interview', '"talk show"', '"conversation with"', '"in conversation"', '"sits down"'],
        "es": ['entrevista', 'conversando', 'platica', 'charla'],
    },
    "meeting": {
        "en": ['"city council"', '"council meeting"', 'selectboard', '"board of supervisors"',
               '"planning commission"', '"committee meeting"', '"school board"', '"town meeting"'],
        "es": ['concejo', '"consejo municipal"', '"cabildo"', '"sesión"', '"reunión"', '"junta"'],
    },
    "event": {
        "en": ['parade', 'festival', 'concert', 'tournament', 'football', 'basketball', 'hockey',
               'soccer', 'baseball', '"live music"'],
        "es": ['desfile', 'concierto', 'fútbol', 'futbol', 'fiesta', 'festival', 'torneo', 'baile'],
    },
    "guide": {
        "en": ['"how to"', 'tips', 'guide', 'tutorial', 'safety', 'cooking', 'recipe', '"what to know"'],
        "es": ['cómo', 'como', 'consejos', 'receta', 'guía', 'seguridad', 'cocina'],
    },
    "museum": {
        "en": ['museum', 'exhibit', 'exhibition', 'gallery', '"art show"', 'curator', 'sculpture'],
        "es": ['museo', 'exposición', 'exhibición', 'galería', 'arte'],
    },
}

MP4_PREF = ["h.264", "MPEG4", "h.264 IA", "512Kb MPEG4"]
CAPTION_EXT = (".srt", ".vtt")
CANDIDATE_FIELDS = ["select", "category", "lang_group", "title", "duration_min", "video_mb", "has_captions",
                    "station", "date", "date_source", "description", "details_url", "identifier", "video_file", "video_md5",
                    "captions"]


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


def station_of(identifier):
    m = re.match(r"^([A-Za-z0-9]+)[-_]", identifier)
    return m.group(1).lower() if m else identifier[:8].lower()


def first(v):
    return v[0] if isinstance(v, list) and v else (v or "")


def is_spanish(meta, cat):
    langs = meta.get("language", [])
    lang = " ".join(map(str, langs if isinstance(langs, list) else [langs])).lower()
    title = str(first(meta.get("title"))).lower()
    if any(k in lang for k in ("spa", "span", "esp")):
        return True
    topic = [p.strip('"') for p in re.findall(r'"[^"]+"', SPANISH_TOPIC)]
    if any(k in title for k in SPANISH_HINTS) and not any(p in title for p in topic):
        return True
    es_words = [t.strip('"') for t in CATEGORIES[cat]["es"]]
    return any(re.search(rf"\b{re.escape(w)}\b", title) for w in es_words if w not in ("festival", "como"))


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
def candidates(cat, spanish, pool, y0, y1):
    en = " OR ".join(CATEGORIES[cat]["en"])
    es = " OR ".join(CATEGORIES[cat]["es"])
    if spanish:
        match = (f"(title:({es}) OR (title:({en}) AND title:{SPANISH_MARK}) "
                 f"OR ((title:({en}) OR subject:({en})) AND language:(spa OR spanish))) "
                 f"AND NOT title:{SPANISH_TOPIC}")
    else:
        match = (f"(title:({en}) OR subject:({en})) "
                 f"AND NOT title:{SPANISH_MARK} AND NOT language:(spa OR spanish)")
    years = f"[{y0}-01-01 TO {y1}-12-31]"
    # date = content/air date; publicdate = upload date, used when an item has no date
    q = f"({BASE_Q}) AND {match} AND {EXCLUDE} AND (date:{years} OR publicdate:{years})"
    data = get(SEARCH, {"q": q, "fl[]": ["identifier"], "rows": pool, "sort[]": "downloads desc",
                        "output": "json"})
    docs = (data or {}).get("response", {}).get("docs", [])
    total = (data or {}).get("response", {}).get("numFound", 0)
    return [d["identifier"] for d in docs], total


def evaluate(identifier, args):
    item = get(META + urllib.parse.quote(identifier))
    if not item or "metadata" not in item:
        return None, "no metadata"
    meta, files = item["metadata"], item.get("files", [])
    if str(meta.get("access-restricted-item", "")).lower() == "true" or item.get("is_dark"):
        return None, "restricted"
    date, date_src = item_date(meta)
    if not date or not (args.from_year <= int(date[:4]) <= args.to_year):
        return None, f"year {date[:4] or 'unknown'}"
    video = pick_video(files, args.max_mb)
    if not video:
        return None, f"no mp4 under {args.max_mb} MB"
    sec = duration_of(meta, files, video)
    if sec is None:
        return None, "unknown duration"
    if not (args.min_min * 60 <= sec <= args.max_min * 60):
        return None, f"duration {sec / 60:.1f} min"
    return {"item": item, "meta": meta, "files": files, "video": video, "seconds": sec,
            "date": date, "date_src": date_src}, "ok"


def item_date(meta):
    """Content date if present, otherwise upload date. Returns (YYYY-MM-DD or YYYY, source)."""
    for key in ("date", "publicdate", "addeddate"):
        m = re.match(r"(\d{4})(-\d{2}-\d{2})?", str(first(meta.get(key))).strip())
        if m:
            return m.group(0), ("aired" if key == "date" else "uploaded")
    return "", ""


def search(args):
    """Collect a large, varied candidate pool per category and language group (no downloads)."""
    rng = random.Random(args.seed)
    used_ids, rows = set(), []
    for cat in args.categories:
        print(f"\n=== {cat}", flush=True)
        per_station = {}
        quotas = [("en", False, args.candidates), ("es", True, args.spanish_candidates)]
        for group, spanish, need in quotas:
            if need <= 0:
                continue
            ids, total = candidates(cat, spanish, args.pool, args.from_year, args.to_year)
            rng.shuffle(ids)
            print(f"  [{group}] {total:,} matches, checking up to {len(ids)} for {need} candidates", flush=True)
            got = 0
            for ident in ids:
                if got >= need:
                    break
                st = station_of(ident)
                if ident in used_ids or per_station.get(st, 0) >= args.per_station:
                    continue
                res, why = evaluate(ident, args)
                if not res:
                    if args.verbose:
                        print(f"     skip {ident[:70]}: {why}")
                    continue
                if spanish != is_spanish(res["meta"], cat):
                    continue
                used_ids.add(ident)
                per_station[st] = per_station.get(st, 0) + 1
                got += 1
                v = res["video"]
                caps = [f["name"] for f in res["files"] if f.get("name", "").lower().endswith(CAPTION_EXT)]
                rows.append({
                    "select": "", "category": cat, "lang_group": group,
                    "title": str(first(res["meta"].get("title")))[:150],
                    "duration_min": f"{res['seconds'] / 60:.1f}",
                    "video_mb": f"{int(v['size']) / 1048576:.1f}",
                    "has_captions": "yes" if caps else "",
                    "station": st, "date": res["date"], "date_source": res["date_src"],
                    "description": clean_text(res["meta"].get("description"), 300),
                    "details_url": f"https://archive.org/details/{ident}",
                    "identifier": ident, "video_file": v["name"], "video_md5": v.get("md5", ""),
                    "captions": "|".join(caps),
                })
                print(f"     {got:>3}. {rows[-1]['title'][:70]}  ({rows[-1]['duration_min']} min)", flush=True)
            if got < need:
                print(f"  ! only {got}/{need} {group} candidates for {cat}; "
                      f"raise --pool or --max-min, or edit CATEGORIES")
    return rows


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
    print("\nCandidates per category (en / es):")
    for cat in args.categories:
        en = sum(1 for r in rows if r["category"] == cat and r["lang_group"] == "en")
        es = sum(1 for r in rows if r["category"] == cat and r["lang_group"] == "es")
        print(f"  {cat:10} {en:>3} / {es:>3}")
    print(f"\nWrote {len(rows)} candidates to {path}")
    print("Next: open it, put an x in the 'select' column for the videos you want, save, then run:")
    print(f"  python {os.path.basename(sys.argv[0])} download {path}")


def cmd_download(args):
    with open(args.csv, encoding="utf-8-sig") as f:
        rows = [r for r in csv.DictReader(f) if (r.get("select") or "").strip().lower() in CHOSEN]
    if not rows:
        print("No rows selected. Put an x in the 'select' column first.")
        return
    print("Selected per category:")
    for cat in CATEGORIES:
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

    s = sub.add_parser("search", help="find candidates and write candidates.csv (no downloads)")
    s.add_argument("-o", "--out", default=os.path.join("collection", "rounds"), help="where candidates.csv is written")
    s.add_argument("--candidates", type=int, default=25, help="English candidates per category")
    s.add_argument("--spanish-candidates", type=int, default=10, help="Spanish candidates per category")
    s.add_argument("--per-station", type=int, default=3, help="max candidates per station per category")
    s.add_argument("--min-min", type=float, default=1.0, help="min duration (minutes)")
    s.add_argument("--max-min", type=float, default=30.0, help="max duration (minutes)")
    s.add_argument("--max-mb", type=int, default=300, help="max video file size (MB)")
    s.add_argument("--pool", type=int, default=200, help="search hits to check per category/language")
    s.add_argument("--categories", nargs="+", choices=list(CATEGORIES), default=list(CATEGORIES),
                   help="only search these categories")
    s.add_argument("--from-year", type=int, default=2024, help="earliest year (inclusive)")
    s.add_argument("--to-year", type=int, default=2026, help="latest year (inclusive)")
    s.add_argument("--seed", type=int, default=42)
    s.add_argument("-v", "--verbose", action="store_true", help="show why candidates were skipped")

    d = sub.add_parser("download", help="download rows marked in the 'select' column")
    d.add_argument("csv", help="candidates.csv with your selections")
    d.add_argument("-o", "--out", default="corpus", help="where the videos go (default: corpus/)")
    d.add_argument("--expect", type=int, default=4, help="expected picks per category (warning only)")
    d.add_argument("-y", "--yes", action="store_true", help="skip the confirmation prompt")

    args = ap.parse_args()
    (cmd_search if args.cmd == "search" else cmd_download)(args)


if __name__ == "__main__":
    main()
