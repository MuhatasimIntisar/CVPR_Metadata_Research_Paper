#!/usr/bin/env python3
"""
Find example videos per content category across the chosen local/community TV sources.
Search only: queries metadata endpoints / listing pages, never downloads video files.

Sources (priority order)
  1  ia_community_media   IA collection:community_media (youtube-* mirrors excluded)
  2  denver_open_media    IA collection:denveropenmedia
  3  cctv_burlington      IA items uploaded by CCTV / Town Meeting TV (identifier cctv17vt-)
  4  okto_tv              okto.tv Oktothek listing, titles + series slug matched (German terms)
  5  dorf_tv              dorftv.at video listing, titles matched (German terms)
  6  swindon_viewpoint    swindonviewpoint.com WordPress REST search
  7  video_volunteers     videovolunteers.org WordPress REST search
  8  cinemata             cinemata.org MediaCMS search API
  9  ia_tv_news           IA collection:tvarchive, excluding CNN/Fox/MSNBC (local stations, C-SPAN, etc.)
  10 stanford_cable_news  IA tvarchive items for CNN/Fox News/MSNBC 2010+ (the Stanford analyzer corpus)

Usage
  python find_category_examples.py                    # 3 examples per source x category
  python find_category_examples.py -n 5 -o out        # 5 each, write out.json / out.csv
  python find_category_examples.py --sources okto_tv dorf_tv --max-pages 60

Notes
  * Okto and dorf tv have no usable search URL, so their listings are paged (newest first)
    and filtered by title keywords. Raise --max-pages for more coverage.
  * IA TV News / Stanford items are usually stream/borrow only, not downloadable. Verify per item.

Stdlib only (Python 3.8+).
"""
import argparse
import csv
import html
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from html.parser import HTMLParser

USER_AGENT = "video-retrieval-dataset-research/0.2 (academic, metadata search only)"
DELAY_S = 1.0  # polite gap between requests

# ---------------------------------------------------------------- categories
#   ia_terms : IA Lucene terms (title/subject[/description])
#   en_terms : plain keyword searches (WordPress, Cinemata)
#   de_terms : lowercase substrings matched against German listing titles (Okto, dorf tv)
CATEGORIES = {
    "news": {
        "label": "News: reports, breaking news, weather, daily updates",
        "ia_terms": ['news', 'newscast', 'weather', '"daily update"', '"news update"', 'breaking', 'headlines'],
        "en_terms": ["news", "report", "weather"],
        "de_terms": ["nachrichten", "news", "aktuell", "bericht", "wetter", "wochenschau", "magazin"],
    },
    "public_affairs": {
        "label": "Public Affairs: committee meetings, councils, civic debates",
        "ia_terms": ['"city council"', 'selectboard', 'committee', '"board meeting"', 'commission',
                     'debate', '"town meeting"', '"candidate forum"', 'hearing'],
        "en_terms": ["council", "panchayat", "election", "debate", "meeting"],
        "de_terms": ["gemeinderat", "stadtrat", "landtag", "politik", "wahl", "diskussion",
                     "debatte", "podium", "bürgerversammlung"],
    },
    "news_adjacent": {
        "label": "News-Adjacent: interviews, talk shows, lifestyle",
        "ia_terms": ['interview', '"talk show"', 'conversation', 'cooking', 'lifestyle', 'podcast'],
        "en_terms": ["interview", "talk", "conversation"],
        "de_terms": ["interview", "im gespräch", "gespräch", "talk", "kochen", "zu gast"],
    },
    "documentary": {
        "label": "Documentary: local history, retrospectives, investigative",
        "ia_terms": ['documentary', 'history', 'retrospective', 'investigation', 'investigative', 'heritage'],
        "en_terms": ["documentary", "history", "investigation"],
        "de_terms": ["doku", "dokumentation", "geschichte", "porträt", "portrait", "reportage", "rückblick"],
    },
    "event_coverage": {
        "label": "Event Coverage: exhibitions, parades, festivals",
        "ia_terms": ['parade', 'festival', 'exhibition', 'exhibit', 'museum', 'carnival', 'fair', 'celebration'],
        "en_terms": ["festival", "parade", "exhibition", "celebration"],
        "de_terms": ["festival", "ausstellung", "eröffnung", "vernissage", "umzug", "parade",
                     "fest", "museum", "opening", "preisverleihung"],
    },
}

# ---------------------------------------------------------------- sources
BIG3 = "identifier:(CNNW_* OR FOXNEWSW_* OR MSNBCW_*)"
SOURCES = {
    "ia_community_media": {"kind": "ia",
        "q": "collection:community_media AND mediatype:movies AND NOT identifier:youtube-*"},
    "denver_open_media": {"kind": "ia", "q": "collection:denveropenmedia AND mediatype:movies"},
    "cctv_burlington": {"kind": "ia", "q": "identifier:cctv17vt* AND mediatype:movies"},
    "okto_tv": {"kind": "listing",
        "list_url": "https://www.okto.tv/de/oktothek/newest_episodes", "page_start": 1,
        "link_re": r"^(?:https://www\.okto\.tv)?/de/sendereihe/[^/]+/video/[^/]+/[^/?#]+$",
        "base": "https://www.okto.tv", "match_url": True},
    "dorf_tv": {"kind": "listing",
        "list_url": "https://www.dorftv.at/videos", "page_start": 0,
        "link_re": r"^(?:https://www\.dorftv\.at)?/video/\d+/?$",
        "base": "https://www.dorftv.at", "match_url": False},
    "swindon_viewpoint": {"kind": "wp", "base": "https://swindonviewpoint.com", "subtype": "vimeo-video"},
    "video_volunteers": {"kind": "wp", "base": "https://www.videovolunteers.org", "subtype": "post"},
    "cinemata": {"kind": "cinemata", "base": "https://cinemata.org"},
    "ia_tv_news": {"kind": "ia", "q": f"collection:tvarchive AND NOT {BIG3}",
        "fields": ["title", "subject", "description"]},
    "stanford_cable_news": {"kind": "ia", "q": f"collection:tvarchive AND {BIG3}",
        "fields": ["title", "subject", "description"]},
}
ALL_SOURCES = list(SOURCES)

IA_SEARCH = "https://archive.org/advancedsearch.php"
IA_FIELDS = ["identifier", "title", "date", "description", "subject", "collection", "language"]
ROW_FIELDS = ["source", "category", "title", "date", "url", "id", "description",
              "subjects", "collections", "language", "matched"]


# ---------------------------------------------------------------- helpers
def fetch(url, params=None, as_json=True, retries=3):
    if params:
        url = f"{url}?{urllib.parse.urlencode(params, doseq=True)}"
    accept = "application/json" if as_json else "text/html"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": accept})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                body = r.read().decode("utf-8", errors="replace")
            time.sleep(DELAY_S)
            return json.loads(body) if as_json else body
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
            if attempt == retries - 1:
                print(f"    ! request failed: {e} ({url[:120]})", file=sys.stderr)
                return None
            time.sleep(2 ** attempt * 2)


def clean(text, limit=200):
    if text is None:
        return ""
    if isinstance(text, list):
        text = " ".join(str(t) for t in text)
    text = re.sub(r"<[^>]+>", " ", html.unescape(str(text)))
    text = re.sub(r"\s+", " ", text).strip()
    return text[:limit] + ("..." if len(text) > limit else "")


def as_list(v):
    if v is None:
        return []
    return v if isinstance(v, list) else [v]


def row(source, cat, **kw):
    r = {k: "" for k in ROW_FIELDS}
    r.update(source=source, category=cat, **kw)
    return r


# ---------------------------------------------------------------- Internet Archive
def search_ia(name, cfg, cat_key, n):
    terms = " OR ".join(CATEGORIES[cat_key]["ia_terms"])
    fields = cfg.get("fields", ["title", "subject"])
    match = " OR ".join(f"{f}:({terms})" for f in fields)
    q = f"({cfg['q']}) AND ({match})"
    data = fetch(IA_SEARCH, {"q": q, "fl[]": IA_FIELDS, "rows": n, "page": 1,
                             "sort[]": "date desc", "output": "json"})
    if not data:
        return [], None
    resp = data.get("response", {})
    out = [row(name, cat_key,
               title=clean(d.get("title"), 150),
               date=str(d.get("date", ""))[:10],
               url=f"https://archive.org/details/{d['identifier']}",
               id=d["identifier"],
               description=clean(d.get("description")),
               subjects="; ".join(str(s) for s in as_list(d.get("subject")))[:200],
               collections="; ".join(as_list(d.get("collection"))),
               language="; ".join(str(x) for x in as_list(d.get("language"))))
           for d in resp.get("docs", [])]
    return out, resp.get("numFound")


# ---------------------------------------------------------------- WordPress
def search_wp(name, cfg, cat_key, n):
    seen, out = set(), []
    for term in CATEGORIES[cat_key]["en_terms"]:
        if len(out) >= n:
            break
        hits = fetch(f"{cfg['base']}/wp-json/wp/v2/search",
                     {"search": term, "type": "post", "subtype": cfg["subtype"], "per_page": n})
        for h in hits or []:
            if h["id"] in seen or len(out) >= n:
                continue
            seen.add(h["id"])
            self_link = (h.get("_links", {}).get("self") or [{}])[0].get("href")
            detail = (fetch(self_link) if self_link else None) or {}
            body = (detail.get("excerpt") or {}).get("rendered") or (detail.get("content") or {}).get("rendered")
            out.append(row(name, cat_key, title=clean(h.get("title"), 150),
                           date=str(detail.get("date", ""))[:10], url=h.get("url"),
                           id=str(h["id"]), description=clean(body), matched=term))
    return out, None


# ---------------------------------------------------------------- Cinemata (MediaCMS)
def search_cinemata(name, cfg, cat_key, n):
    seen, out, total = set(), [], 0
    for term in CATEGORIES[cat_key]["en_terms"]:
        if len(out) >= n:
            break
        data = fetch(f"{cfg['base']}/api/v1/search", {"q": term})
        if not data:
            continue
        total += data.get("count", 0)
        for m in data.get("results", []):
            tok = m.get("friendly_token")
            if m.get("media_type") != "video" or tok in seen or len(out) >= n:
                continue
            seen.add(tok)
            out.append(row(name, cat_key, title=clean(m.get("title"), 150),
                           date=str(m.get("add_date", ""))[:10],
                           url=m.get("url") or f"{cfg['base']}/view?m={tok}", id=tok,
                           description=clean(m.get("summary") or m.get("description")),
                           subjects="; ".join(c.get("title", "") for c in m.get("categories_info") or []),
                           collections=m.get("author_name", ""),
                           language="; ".join(c.get("title", "") for c in m.get("media_country_info") or []),
                           matched=term))
    return out, total or None


# ---------------------------------------------------------------- HTML listings (Okto, dorf tv)
class LinkCollector(HTMLParser):
    def __init__(self, link_re):
        super().__init__()
        self.link_re = re.compile(link_re)
        self.stack, self.links = [], {}

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            href = dict(attrs).get("href") or ""
            href = href.split("?")[0].split("#")[0]
            self.stack.append([href, []] if self.link_re.match(href) else None)
        elif tag == "img" and self.stack and self.stack[-1]:
            alt = dict(attrs).get("alt")
            if alt:
                self.stack[-1].append(alt)  # fallback title if the link has no text

    def handle_data(self, data):
        for item in self.stack:
            if item:
                item[1].append(data)

    def handle_endtag(self, tag):
        if tag == "a" and self.stack:
            item = self.stack.pop()
            if item:
                text = re.sub(r"\s+", " ", " ".join(item[1])).strip() or " ".join(item[2:])
                if len(text) > len(self.links.get(item[0], "")):
                    self.links[item[0]] = text
                else:
                    self.links.setdefault(item[0], text)


_listing_cache = {}


def crawl_listing(name, cfg, max_pages):
    if name in _listing_cache:
        return _listing_cache[name]
    items = {}
    print(f"  crawling up to {max_pages} listing pages...", flush=True)
    for i in range(max_pages):
        page = cfg["page_start"] + i
        body = fetch(cfg["list_url"], {"page": page}, as_json=False)
        if not body:
            break
        p = LinkCollector(cfg["link_re"])
        p.feed(body)
        new = 0
        for href, text in p.links.items():
            url = href if href.startswith("http") else cfg["base"] + href
            if url not in items:
                items[url] = text
                new += 1
            elif len(text) > len(items[url]):
                items[url] = text
        if new == 0:
            break  # past the last page or layout changed
    print(f"  {len(items)} videos indexed", flush=True)
    if not items:
        print("  ! no video links parsed; the page layout may have changed (check link_re)", file=sys.stderr)
    _listing_cache[name] = items
    return items


def search_listing(name, cfg, cat_key, n, max_pages):
    items = crawl_listing(name, cfg, max_pages)
    out, total = [], 0
    terms = CATEGORIES[cat_key]["de_terms"]
    for url, title in items.items():
        hay = title.lower()
        if cfg.get("match_url"):
            hay += " " + urllib.parse.unquote(url).lower().replace("-", " ").replace("_", " ")
        hit = next((t for t in terms if t in hay), None)
        if not hit:
            continue
        total += 1
        if len(out) < n:
            out.append(row(name, cat_key, title=clean(title, 150) or url.rsplit("/", 1)[-1],
                           url=url, id=url.rstrip("/").rsplit("/", 2)[-2 if cfg.get("match_url") else -1],
                           language="de", matched=hit))
    return out, total


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-n", "--per-category", type=int, default=3, help="examples per source x category")
    ap.add_argument("-o", "--out", default="category_examples", help="output basename (.json and .csv)")
    ap.add_argument("--sources", nargs="+", choices=ALL_SOURCES, default=ALL_SOURCES)
    ap.add_argument("--max-pages", type=int, default=30, help="listing pages to crawl for Okto / dorf tv")
    args = ap.parse_args()

    results, matrix = [], {}
    for src in args.sources:
        cfg = SOURCES[src]
        print(f"\n=== {src}", flush=True)
        for cat_key in CATEGORIES:
            kind = cfg["kind"]
            if kind == "ia":
                rows, total = search_ia(src, cfg, cat_key, args.per_category)
            elif kind == "wp":
                rows, total = search_wp(src, cfg, cat_key, args.per_category)
            elif kind == "cinemata":
                rows, total = search_cinemata(src, cfg, cat_key, args.per_category)
            else:
                rows, total = search_listing(src, cfg, cat_key, args.per_category, args.max_pages)
            matrix[(src, cat_key)] = len(rows)
            results.extend(rows)
            total_s = f" of ~{total:,} matches" if isinstance(total, int) else ""
            print(f"  [{cat_key}] {len(rows)} found{total_s}")
            for r in rows:
                print(f"     - {r['title'][:80]} ({r['date'] or 'n/d'})  {r['url']}")

    cats = list(CATEGORIES)
    w = max(len(s) for s in args.sources)
    print("\nCoverage (examples found):")
    print(" " * (w + 2) + "  ".join(f"{c[:14]:>14}" for c in cats))
    for s in args.sources:
        print(f"{s:<{w}}  " + "  ".join(f"{matrix[(s, c)]:>14}" for c in cats))
    gaps = [(s, c) for (s, c), k in matrix.items() if k == 0]
    if gaps:
        print("\nGaps (no example found; adjust terms in CATEGORIES or check manually):")
        for s, c in gaps:
            print(f"  {s} / {c}")

    with open(f"{args.out}.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    with open(f"{args.out}.csv", "w", newline="", encoding="utf-8") as f:
        wtr = csv.DictWriter(f, fieldnames=ROW_FIELDS)
        wtr.writeheader()
        wtr.writerows(results)
    print(f"\nWrote {len(results)} rows to {args.out}.json and {args.out}.csv")


if __name__ == "__main__":
    main()
